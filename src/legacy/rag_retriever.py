"""Dual-RAG retriever for MFT verification arbitrage.

Supports two collections:
1. static_news_index — Official_Corpus: historical news articles for fact-checking
2. social_stream_index — Social_Velocity_Stream: real-time social media posts
   (filtered to post timestamps < T₁ to prevent look-ahead bias)

Legacy single-index RAGRetriever is preserved for backward compatibility.
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import pickle
import re
import time

import numpy as np
import pandas as pd
from src.domain_adapter import DomainAdapter, get_adapter

# Legacy entity list preserved for the RAG corpus builder (always finance)
ENTITY_LIST = DomainAdapter("finance").entities

# Build regex pattern for entity matching
_ENTITY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(e) for e in sorted(ENTITY_LIST, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def extract_entity(text):
    """Extract entity name from text using active domain adapter.

    Returns matched entity (canonical form) or domain-specific fallback.
    Uses DomainAdapter so health domain returns health entities, etc.
    """
    return get_adapter().extract_entity(text)


def _keyword_overlap(query, text):
    """Simple keyword overlap score between query and text."""
    q_words = set(w.lower() for w in query.split() if len(w) > 3)
    t_words = set(w.lower() for w in text.split() if len(w) > 3)
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


class RAGRetriever:
    """Retrieval-Augmented Generation retriever for financial news (legacy).

    Builds an embedding index over a financial news corpus for fast
    cosine-similarity retrieval of documents relevant to a headline.
    """

    CORPUS_PATH = "./input/rag_corpus/corpus.json"
    CACHE_DIR = "./output/rag_cache"
    INDEX_PATH = os.path.join(CACHE_DIR, "embedding_index.npy")
    METADATA_PATH = os.path.join(CACHE_DIR, "metadata.pkl")
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model = None
        self.corpus = []
        self.embeddings = None
        self._loaded = False

    def build_corpus(self, max_docs=5000):
        """Filter financial_news_dataset.csv for entity mentions, save to corpus."""
        os.makedirs(os.path.dirname(self.CORPUS_PATH), exist_ok=True)

        if os.path.exists(self.CORPUS_PATH):
            self.corpus = json.load(open(self.CORPUS_PATH))
            print(f"[RAG] Loaded existing corpus: {len(self.corpus)} docs from {self.CORPUS_PATH}")
            return self.corpus

        print("[RAG] Building corpus from financial_news_dataset.csv...")
        reader = pd.read_csv(
            "./input/financial_news_dataset.csv",
            encoding="ISO-8859-1",
            chunksize=5000,
            usecols=["text"]
        )

        seen = set()
        docs = []
        for chunk in reader:
            for text in chunk["text"].dropna():
                text = str(text).strip()
                if len(text) < 100:
                    continue
                entities = _ENTITY_PATTERN.findall(text)
                if entities:
                    dedup_key = text[:200]
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        docs.append({"text": text, "entities": list(set(e.lower() for e in entities))})
            if len(docs) >= max_docs:
                break

        self.corpus = docs[:max_docs]
        with open(self.CORPUS_PATH, "w") as f:
            json.dump(self.corpus, f, indent=2)
        print(f"[RAG] Corpus saved: {len(self.corpus)} docs to {self.CORPUS_PATH}")
        return self.corpus

    def _load_model(self):
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"[RAG] Loading embedding model: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print("[RAG] Model loaded.")

    def build_index(self):
        if not self.corpus:
            self.build_corpus()
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._load_model()

        print(f"[RAG] Embedding {len(self.corpus)} documents...")
        texts = [doc["text"] for doc in self.corpus]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)

        np.save(self.INDEX_PATH, self.embeddings)
        with open(self.METADATA_PATH, "wb") as f:
            pickle.dump(self.corpus, f)
        print(f"[RAG] Embedding index saved to {self.CACHE_DIR} "
              f"({self.embeddings.shape[0]} docs, {self.embeddings.shape[1]} dims)")
        self._loaded = True

    def load_cache(self):
        if self._loaded:
            return True
        if os.path.exists(self.INDEX_PATH) and os.path.exists(self.METADATA_PATH):
            self.embeddings = np.load(self.INDEX_PATH)
            with open(self.METADATA_PATH, "rb") as f:
                self.corpus = pickle.load(f)
            self._loaded = True
            print(f"[RAG] Loaded cached index: {len(self.corpus)} docs, {self.embeddings.shape[1]} dims")
            return True
        return False

    def ensure_index(self):
        if not self.load_cache():
            self.build_corpus()
            self.build_index()

    def retrieve(self, headline, top_k=3):
        """Retrieve top-k relevant corpus documents for a headline.

        Returns list of (doc_text, similarity_score) tuples.
        """
        if not self._loaded or self.embeddings is None:
            return [], 0.0

        entity = extract_entity(headline)
        if entity == "UNKNOWN":
            return [], 0.0

        self._load_model()
        start = time.time()
        query_emb = self.model.encode([headline], show_progress_bar=False)
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.1:
                results.append((self.corpus[idx]["text"][:1500], round(float(scores[idx]), 4)))

        elapsed = (time.time() - start) * 1000
        return results, elapsed


# ── Social Stream Retriever (Phase 3.1) ─────────────────────────

class SocialStreamRetriever:
    """Retrieves social media posts timestamped before a cutoff (T₁).

    Indexes the social_stream.csv and retrieves posts relevant to a
    headline/entity, filtered to posts before T₁ to prevent look-ahead
    bias in the LLM evaluation at T₁.
    """

    def __init__(self, social_stream_path="./output/social_stream.csv",
                 t1_cutoff_s=5.0, top_k=5):
        """
        Args:
            social_stream_path: Path to social_stream.csv
            t1_cutoff_s: Only retrieve posts with timestamp < this value (default: T₁=5s)
            top_k: Number of social posts to return per query
        """
        self.social_stream_path = social_stream_path
        self.t1_cutoff_s = t1_cutoff_s
        self.top_k = top_k
        self._posts_df = None
        self._loaded = False

    def load(self):
        """Load social stream CSV into memory."""
        if self._loaded:
            return True
        if not os.path.exists(self.social_stream_path):
            print(f"[SocialRAG] Social stream not found at {self.social_stream_path}")
            return False
        self._posts_df = pd.read_csv(self.social_stream_path)
        # Ensure numeric types
        self._posts_df["timestamp_seconds"] = pd.to_numeric(
            self._posts_df["timestamp_seconds"], errors="coerce"
        )
        self._loaded = True
        print(f"[SocialRAG] Loaded {len(self._posts_df)} posts from {self.social_stream_path}")
        return True

    def retrieve_for_event(self, event_id, top_k=None):
        """Retrieve social posts for a specific event_id before T₁ cutoff.

        Args:
            event_id: Event ID string (e.g., EVT-00042)
            top_k: Override default top_k

        Returns:
            list of (post_text, timestamp, post_type) tuples
        """
        if not self._loaded:
            if not self.load():
                return []

        k = top_k or self.top_k
        event_posts = self._posts_df[
            (self._posts_df["event_id"] == event_id) &
            (self._posts_df["timestamp_seconds"] < self.t1_cutoff_s)
        ]

        if event_posts.empty:
            return []

        results = []
        for _, row in event_posts.iterrows():
            results.append((
                str(row["text"])[:500],
                float(row["timestamp_seconds"]),
                str(row["post_type"]),
            ))

        # Sort by timestamp descending (most recent first up to T₁)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def retrieve_by_entity(self, headline, top_k=None):
        """Retrieve social posts matching the entity in the headline, before T₁.

        Cross-event retrieval: finds social posts from other events involving
        the same entity, filtered to posts before T₁. This gives the LLM
        a pattern baseline for early social reactions to this entity.

        Args:
            headline: The headline text to extract entity from
            top_k: Override default top_k

        Returns:
            list of (post_text, timestamp, event_id, post_type) tuples
        """
        if not self._loaded:
            if not self.load():
                return []

        k = top_k or self.top_k
        entity = extract_entity(headline)
        if entity == "UNKNOWN":
            return []

        # Find posts matching this entity, before T₁ cutoff
        entity_lower = entity.lower()
        matched = self._posts_df[
            (self._posts_df["timestamp_seconds"] < self.t1_cutoff_s) &
            (self._posts_df["entity"].str.lower().str.contains(entity_lower, na=False) |
             self._posts_df["text"].str.lower().str.contains(entity_lower, na=False))
        ]

        if matched.empty:
            return []

        # Score by keyword overlap with headline, sort, return top-k
        scored = []
        for _, row in matched.iterrows():
            overlap = _keyword_overlap(headline, str(row["text"]))
            # Boost debunk posts slightly (they're more informative)
            type_boost = 0.1 if row["post_type"] in ("skepticism", "debunk") else 0.0
            scored.append((
                overlap + type_boost,
                str(row["text"])[:500],
                float(row["timestamp_seconds"]),
                str(row["event_id"]),
                str(row["post_type"]),
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s[1], s[2], s[3], s[4]) for s in scored[:k]]
        return results

    def summarize_velocity(self, event_id):
        """Summarize the social media velocity for an event at T₁.

        Returns dict with counts of each post type before T₁.
        Useful for determining debunk vs amplification velocity.
        """
        if not self._loaded:
            if not self.load():
                return {}

        event_posts = self._posts_df[
            (self._posts_df["event_id"] == event_id) &
            (self._posts_df["timestamp_seconds"] < self.t1_cutoff_s)
        ]

        if event_posts.empty:
            return {"total_before_t1": 0}

        type_counts = event_posts["post_type"].value_counts().to_dict()
        return {
            "total_before_t1": int(len(event_posts)),
            "type_counts": {str(k): int(v) for k, v in type_counts.items()},
            "has_debunk": "debunk" in type_counts or "skepticism" in type_counts,
            "has_amplification": "amplification" in type_counts or "confirmation" in type_counts,
        }


# ── Dual RAG Orchestrator (Phase 3.1) ───────────────────────────

class DualRAGRetriever:
    """Orchestrates retrieval from both static news and social stream indices.

    Wraps RAGRetriever (news corpus) and SocialStreamRetriever (social posts)
    to provide unified context for the LLM evaluation at T₁.
    """

    def __init__(self, news_retriever=None, social_retriever=None,
                 social_stream_path="./output/social_stream.csv",
                 t1_cutoff_s=5.0, news_top_k=3, social_top_k=5):
        """
        Args:
            news_retriever: Existing RAGRetriever instance (or None to create)
            social_retriever: Existing SocialStreamRetriever (or None to create)
            social_stream_path: Path to social_stream.csv
            t1_cutoff_s: Social post timestamp cutoff (T₁)
            news_top_k: Number of news articles to retrieve
            social_top_k: Number of social posts to retrieve
        """
        self.news_retriever = news_retriever or RAGRetriever()
        self.social_retriever = social_retriever or SocialStreamRetriever(
            social_stream_path=social_stream_path,
            t1_cutoff_s=t1_cutoff_s,
        )
        self.news_top_k = news_top_k
        self.social_top_k = social_top_k

    def ensure_index(self):
        """Ensure both indices are loaded."""
        self.news_retriever.ensure_index()
        self.social_retriever.load()

    def retrieve(self, headline, event_id=None):
        """Retrieve context from both indices for an LLM evaluation.

        Args:
            headline: The headline text to verify
            event_id: Optional event_id for targeted social retrieval

        Returns:
            dict with keys:
                news_context: list of (text, score) tuples
                social_context: list of (text, timestamp, post_type) tuples
                social_velocity: dict with velocity summary
                retrieval_latency_ms: total retrieval time
        """
        start = time.time()

        # Static news retrieval
        news_results, news_latency = self.news_retriever.retrieve(headline, top_k=self.news_top_k)

        # Social stream retrieval
        if event_id:
            social_results = self.social_retriever.retrieve_for_event(event_id, top_k=self.social_top_k)
            social_velocity = self.social_retriever.summarize_velocity(event_id)
        else:
            social_results = self.social_retriever.retrieve_by_entity(headline, top_k=self.social_top_k)
            social_velocity = {}

        elapsed = (time.time() - start) * 1000

        return {
            "news_context": news_results,
            "social_context": social_results,
            "social_velocity": social_velocity,
            "retrieval_latency_ms": round(elapsed, 2),
            "news_latency_ms": round(news_latency, 2),
        }

    def format_context_for_prompt(self, dual_context, max_news_chars=2000, max_social_chars=1500):
        """Format dual context into a structured string for the LLM prompt.

        Args:
            dual_context: Output from retrieve()
            max_news_chars: Max characters for news context
            max_social_chars: Max characters for social context

        Returns:
            str: Formatted context string for the prompt
        """
        parts = []

        # News context section
        news = dual_context.get("news_context", [])
        if news:
            news_text = "\n---\n".join(doc[:max_news_chars // max(len(news), 1)] for doc, _score in news)
            parts.append(f"[Verifiable Facts — News Corpus]\n{news_text}")

        # Social context section
        social = dual_context.get("social_context", [])
        if social:
            social_lines = []
            for item in social:
                if len(item) >= 3:
                    text, ts, ptype = item[:3]
                elif len(item) >= 2:
                    text, ts = item[:2]
                    ptype = "post"
                else:
                    text, ts, ptype = item[0], 0, "post"
                social_lines.append(f"[{ptype}] t={ts:.1f}s: {text[:300]}")
            social_text = "\n".join(social_lines)
            parts.append(f"[Social Media Consensus — Posts before T1]\n{social_text}")

        # Velocity summary
        velocity = dual_context.get("social_velocity", {})
        if velocity and velocity.get("total_before_t1", 0) > 0:
            tc = velocity.get("type_counts", {})
            parts.append(f"[Social Velocity]\n"
                         f"Posts before T1: {velocity['total_before_t1']}\n"
                         f"Breakdown: {json.dumps(tc)}")

        return "\n\n".join(parts) if parts else "No relevant context found."
