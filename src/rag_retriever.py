"""RAG retriever for financial news authenticity verification.

Builds a searchable corpus from financial news, embeds with sentence-transformers,
and retrieves relevant context for a given headline via cosine similarity.
"""

import os
import re
import json
import time
import pickle
import numpy as np
import pandas as pd

ENTITY_LIST = [
    "Apple", "Microsoft", "Amazon", "Alphabet", "Meta", "Tesla", "NVIDIA",
    "JPMorgan", "Wells Fargo", "Goldman Sachs", "ExxonMobil", "Chevron",
    "Pfizer", "Moderna", "Disney", "Ford", "Toyota", "Walmart", "Home Depot",
    "Nike", "FedEx", "Visa", "Mastercard", "Netflix", "Oracle", "IBM",
    "Intel", "Adobe", "Salesforce", "Verizon",
    # Major S&P 500 additions
    "Berkshire Hathaway", "UnitedHealth", "Johnson & Johnson", "Procter & Gamble",
    "Coca-Cola", "PepsiCo", "McDonald's", "Boeing", "Caterpillar", "3M",
    "American Express", "Honeywell", "Merck", "AbbVie", "Cisco",
    "AT&T", "Comcast", "NextEra Energy", "Bank of America", "Citigroup",
    "Morgan Stanley", "Charles Schwab", "BlackRock", "Thermo Fisher",
    "Accenture", "Uber", "AMD", "Micron", "Qualcomm", "Broadcom",
]

# Build regex pattern for entity matching
_ENTITY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(e) for e in sorted(ENTITY_LIST, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def extract_entity(text):
    """Extract entity name from text using regex matching.

    Returns matched entity (canonical form from ENTITY_LIST) or 'UNKNOWN'.
    """
    match = _ENTITY_PATTERN.search(text)
    if match:
        raw = match.group(1)
        # Find canonical form (preserving original capitalization from ENTITY_LIST)
        for e in ENTITY_LIST:
            if e.lower() == raw.lower():
                return e
        return raw.title()
    return "UNKNOWN"


class RAGRetriever:
    """Retrieval-Augmented Generation retriever for financial news.

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

        # Check if corpus already exists
        if os.path.exists(self.CORPUS_PATH):
            self.corpus = json.load(open(self.CORPUS_PATH))
            print(f"[RAG] Loaded existing corpus: {len(self.corpus)} docs from {self.CORPUS_PATH}")
            return self.corpus

        # Read financial news dataset, filter by entity mentions
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
                    dedup_key = text[:200]  # first 200 chars for dedup
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        docs.append({"text": text, "entities": list(set(e.lower() for e in entities))})
            if len(docs) >= max_docs:
                break

        # Trim to max_docs
        self.corpus = docs[:max_docs]

        with open(self.CORPUS_PATH, "w") as f:
            json.dump(self.corpus, f, indent=2)
        print(f"[RAG] Corpus saved: {len(self.corpus)} docs to {self.CORPUS_PATH}")
        return self.corpus

    def _load_model(self):
        """Lazy-load sentence-transformers model."""
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer
        print(f"[RAG] Loading embedding model: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print("[RAG] Model loaded.")

    def build_index(self):
        """Embed corpus documents and save to cache."""
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

        print(f"[RAG] Embedding index saved to {self.CACHE_DIR} ({self.embeddings.shape[0]} docs, {self.embeddings.shape[1]} dims)")
        self._loaded = True

    def load_cache(self):
        """Load pre-computed embeddings from cache."""
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
        """Build index if cache missing, otherwise load from cache."""
        if not self.load_cache():
            self.build_corpus()
            self.build_index()

    def retrieve(self, headline, top_k=3):
        """Retrieve top-k relevant corpus documents for a headline.

        Returns list of (doc_text, similarity_score) tuples.
        """
        if not self._loaded or self.embeddings is None:
            return []

        entity = extract_entity(headline)
        if entity == "UNKNOWN":
            return []

        self._load_model()

        start = time.time()
        query_emb = self.model.encode([headline], show_progress_bar=False)
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.1:  # similarity threshold
                results.append((self.corpus[idx]["text"][:1500], round(float(scores[idx]), 4)))

        elapsed = (time.time() - start) * 1000
        return results, elapsed
