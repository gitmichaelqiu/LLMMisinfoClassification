"""RAG (Retrieval-Augmented Generation) verifier.

Retrieves relevant context from a knowledge corpus before calling the LLM,
grounding verification in external evidence rather than parametric knowledge alone.

Provides:
- DenseRetriever: sentence-transformer embeddings + cosine similarity
- SparseRetriever: TF-IDF vectorization + cosine similarity
- RAGVerifier: orchestrates retrieval → context formatting → LLM → parse

All retrievers return (doc_id, doc_text, relevance_score) tuples for
evidence traceability. The RAGVerifier records evidence IDs, retrieval
hits, relevance scores, and per-stage latency in metadata.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from src.llm_clients import LLMClient, create_client
from src.prompts import RAG_SYSTEM, RAG_USER, format_user_prompt
from src.schemas import VerificationItem, VerificationResult, VerifierConfig
from src.verifier_single_shot import SingleShotVerifier

# ── Retrievers ─────────────────────────────────────────────────────


class SparseRetriever:
    """TF-IDF based sparse retriever.

    Builds a TF-IDF index over the corpus, then retrieves top-k documents
    by cosine similarity. Tracks document IDs for evidence provenance.

    Attributes:
        corpus: List of (doc_id, text) tuples.
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        corpus: List[Tuple[str, str]],
        top_k: int = 3,
    ):
        """Initialise the TF-IDF index.

        Args:
            corpus: List of (doc_id, claim_text) tuples to search over.
            top_k: Default number of documents to retrieve.
        """
        self.corpus = corpus
        self.top_k = top_k
        self._build_index()

    def _build_index(self) -> None:
        """Build TF-IDF index from corpus texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [text for _, text in self.corpus]
        if not texts:
            self._vectorizer = None
            self._tfidf_matrix = np.empty((0, 0))
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Retrieve top-k documents relevant to the query.

        Args:
            query: Search query (usually the claim text).
            k: Number of results (defaults to self.top_k).

        Returns:
            List of (doc_id, doc_text, relevance_score) tuples,
            sorted descending by score. Empty list if no matches above zero.
        """
        k = k or self.top_k
        if not self.corpus or self._vectorizer is None or self._tfidf_matrix.size == 0:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = (self._tfidf_matrix @ query_vec.T).toarray().flatten()

        top_indices = np.argsort(scores)[::-1][:k]
        results: List[Tuple[str, str, float]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc_id, doc_text = self.corpus[idx]
            results.append((doc_id, doc_text, float(scores[idx])))
        return results


class DenseRetriever:
    """Dense embedding retriever using sentence-transformers.

    Encodes the corpus into dense vectors and retrieves top-k documents
    by cosine similarity. Falls back to TF-IDF features for the query
    encoding if sentence-transformers is unavailable.

    Attributes:
        corpus: List of (doc_id, text) tuples.
        top_k: Default number of results to return.
    """

    def __init__(
        self,
        corpus: List[Tuple[str, str]],
        top_k: int = 3,
    ):
        """Initialise the embedding index.

        Args:
            corpus: List of (doc_id, claim_text) tuples to search over.
            top_k: Default number of documents to retrieve.
        """
        self.corpus = corpus
        self.top_k = top_k
        self._cached_embeddings: Optional[np.ndarray] = None
        self._build_index()

    def _build_index(self) -> None:
        """Build dense embedding index over corpus texts."""
        texts = [text for _, text in self.corpus]
        if not texts:
            self._cached_embeddings = np.array([])
            return

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._cached_embeddings = model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            self._dim = self._cached_embeddings.shape[1]
        except ImportError:
            # Fallback: use TF-IDF vectors as dense-ish features
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(stop_words="english", max_features=5000)
            self._cached_embeddings = vec.fit_transform(texts).toarray().astype(np.float64)
            self._dim = self._cached_embeddings.shape[1]
            # Keep vectorizer for query encoding
            self._fallback_vec = vec

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Retrieve top-k documents by embedding cosine similarity.

        Args:
            query: Search query (usually the claim text).
            k: Number of results (defaults to self.top_k).

        Returns:
            List of (doc_id, doc_text, relevance_score) tuples,
            sorted descending by score. Empty list if no matches above zero.
        """
        k = k or self.top_k
        if not self.corpus or self._cached_embeddings is None or len(self._cached_embeddings) == 0:
            return []
        if not query.strip():
            return []

        # Encode query
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        except ImportError:
            vec = getattr(self, "_fallback_vec", None)
            if vec is None:
                return self.corpus[:k]  # bare fallback: return first k
            query_vec = vec.transform([query]).toarray().astype(np.float64)[0]

        # Cosine similarity
        emb_norms = np.linalg.norm(self._cached_embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0 or (emb_norms == 0).all():
            return []

        sims = (self._cached_embeddings @ query_vec) / (emb_norms * query_norm + 1e-10)
        sims = np.nan_to_num(sims, nan=0.0)

        top_indices = np.argsort(sims)[::-1][:k]
        results: List[Tuple[str, str, float]] = []
        for idx in top_indices:
            if sims[idx] <= 0:
                continue
            doc_id, doc_text = self.corpus[idx]
            results.append((doc_id, doc_text, float(sims[idx])))
        return results


# ── RAG Verifier ───────────────────────────────────────────────────


class RAGVerifier:
    """Retrieval-Augmented Generation verifier.

    Workflow:
    1. Retrieve relevant documents from the knowledge corpus.
    2. Format and truncate context to fit LLM window limits.
    3. Call the LLM with the RAG prompt (system + user with context).
    4. Parse the response and attach retrieval metadata.

    Records in each VerificationResult:
    - evidence_ids: which corpus documents were retrieved.
    - retrieval_hits: number of documents retrieved (may be < top_k).
    - relevance_scores: similarity scores for each hit.
    - retrieval_latency_s / llm_latency_s: per-stage timing.
    - context_length_chars: size of the formatted context.

    Supports configurable retriever types:
      - "dense": sentence-transformer embedding similarity.
      - "sparse": TF-IDF cosine similarity.

    Attributes:
        corpus: Knowledge corpus as list of VerificationItem objects.
        retriever: Active retriever instance (DenseRetriever or SparseRetriever).
        client: LLM client instance.
        config: Verifier configuration.
    """

    def __init__(
        self,
        corpus: List[VerificationItem],
        config: Optional[VerifierConfig] = None,
        client: Optional[LLMClient] = None,
        retriever_type: str = "dense",
        top_k: int = 3,
    ):
        """Initialise the RAG verifier.

        Args:
            corpus: Knowledge corpus as VerificationItem list. The claim_text
                of each item is indexed for retrieval.
            config: Verifier configuration (retriever_type is overridden).
            client: LLM client (defaults to MockClient).
            retriever_type: "dense" or "sparse".
            top_k: Number of documents to retrieve per query.

        Raises:
            ValueError: If retriever_type is not recognised.
        """
        self.corpus = corpus
        self.config = config or VerifierConfig(
            retriever_type=retriever_type,
            extra={"top_k": top_k},
        )
        self.client = client or create_client(mock=True)
        self.retriever_type = retriever_type
        self.top_k = top_k

        # Build corpus tuples once
        corpus_tuples: List[Tuple[str, str]] = [
            (item.id, item.claim_text) for item in corpus
        ]

        if retriever_type == "dense":
            self.retriever = DenseRetriever(corpus_tuples, top_k=top_k)
        elif retriever_type == "sparse":
            self.retriever = SparseRetriever(corpus_tuples, top_k=top_k)
        else:
            raise ValueError(
                f"Unknown retriever_type: {retriever_type}. Options: 'dense', 'sparse'"
            )

        self._parser = SingleShotVerifier(config=self.config, client=self.client)

    def verify(self, item: VerificationItem) -> VerificationResult:
        """Verify a claim using retrieval-augmented generation.

        Args:
            item: The claim to verify.

        Returns:
            VerificationResult with:
            - evidence: retrieved document excerpts prepended to LLM evidence.
            - metadata: retriever stats, latency breakdown, evidence trace.
        """
        # 1. Retrieve
        retrieval_start = time.time()
        hits = self.retriever.retrieve(item.claim_text, k=self.top_k)
        retrieval_latency = time.time() - retrieval_start

        # 2. Format context with truncation
        context = self._format_context(hits)

        # 3. Generate LLM verdict with context
        llm_start = time.time()
        user_prompt = format_user_prompt(
            RAG_USER,
            claim_text=item.claim_text,
            context=context,
        )
        raw_output = self.client.generate(
            system_prompt=RAG_SYSTEM,
            user_prompt=user_prompt,
            config=self.config,
        )
        llm_latency = time.time() - llm_start
        total_latency = retrieval_latency + llm_latency

        # 4. Parse
        result = self._parser._parse_response(raw_output, item.id, total_latency)

        # 5. Attach retrieval evidence and metadata
        evidence_ids = [doc_id for doc_id, _, _ in hits]
        relevance_scores = [score for _, _, score in hits]

        # Prepend retrieved evidence to LLM-generated evidence
        retrieval_evidence: List[str] = [
            f"[Retrieved] doc={doc_id} score={score:.3f}: {text[:120]}"
            for doc_id, text, score in hits
        ]
        result.evidence = retrieval_evidence + result.evidence

        result.metadata.update(
            {
                "retriever_type": self.retriever_type,
                "top_k": self.top_k,
                "retrieval_hits": len(hits),
                "evidence_ids": evidence_ids,
                "relevance_scores": relevance_scores,
                "retrieval_latency_s": round(retrieval_latency, 4),
                "llm_latency_s": round(llm_latency, 4),
                "context_length_chars": len(context),
            }
        )

        return result

    def verify_batch(
        self,
        items: List[VerificationItem],
    ) -> List[VerificationResult]:
        """Verify multiple claims sequentially.

        Args:
            items: List of claims to verify.

        Returns:
            List of VerificationResult objects.
        """
        return [self.verify(item) for item in items]

    # ── Context helpers ──────────────────────────────────────────

    def _format_context(
        self,
        hits: List[Tuple[str, str, float]],
        max_chars: int = 2000,
    ) -> str:
        """Format retrieved documents into a single context string.

        Each document is prefixed with its ID and relevance score.
        Truncates early entries and drops entries that would exceed
        max_chars to stay within LLM context windows.

        Args:
            hits: List of (doc_id, doc_text, score) tuples, sorted
                  descending by score.
            max_chars: Maximum total character length of the context.

        Returns:
            Formatted context string (empty string if no hits).
        """
        parts: List[str] = []
        running_total = 0

        for doc_id, doc_text, score in hits:
            entry = f"[Document {doc_id}] (relevance={score:.3f})\n{doc_text}\n"
            entry_len = len(entry)

            if running_total + entry_len > max_chars:
                remaining = max_chars - running_total
                if remaining > 50:
                    parts.append(entry[:remaining])
                break

            parts.append(entry)
            running_total += entry_len

        return "".join(parts)
