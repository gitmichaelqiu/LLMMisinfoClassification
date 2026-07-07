"""Retrieval-Augmented Generation (RAG) verifier.

Retrieves relevant context from a knowledge corpus before calling the LLM,
grounding verification in external evidence rather than parametric knowledge alone.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.llm_clients import LLMClient, create_client
from src.prompts import RAG_SYSTEM, RAG_USER, format_user_prompt
from src.schemas import VerificationItem, VerificationResult, VerifierConfig
from src.verifier_single_shot import SingleShotVerifier


class DenseRetriever:
    """Simple dense retriever using sentence embeddings and cosine similarity."""

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self._embeddings = None

    def _lazy_encode(self):
        if self._embeddings is None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("all-MiniLM-L6-v2")
                self._embeddings = model.encode(self.corpus, show_progress_bar=False)
            except ImportError:
                raise ImportError("sentence-transformers required for dense retriever")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve top-k most relevant corpus documents.

        Args:
            query: The query string (claim text).
            k: Number of documents to retrieve.

        Returns:
            List of relevant document strings.
        """
        self._lazy_encode()
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], show_progress_bar=False)[0]
        except ImportError:
            return self.corpus[:k]

        scores = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.corpus[i] for i in top_indices]


class SparseRetriever:
    """Simple sparse (TF-IDF) retriever."""

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.corpus = corpus
        self._tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve top-k documents by TF-IDF similarity."""
        query_vec = self.vectorizer.transform([query])
        scores = np.asarray((self._tfidf_matrix @ query_vec.T).toarray().flatten())
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.corpus[i] for i in top_indices]


class RAGVerifier:
    """Retrieval-augmented verifier.

    Retrieves relevant context from a corpus, then passes it to the LLM
    alongside the claim for evidence-grounded verification.

    Attributes:
        retriever: DenseRetriever or SparseRetriever instance.
        client: LLM client.
        config: Verifier configuration.
    """

    def __init__(
        self,
        corpus: Optional[list[str]] = None,
        retriever_type: str = "dense",
        config: Optional[VerifierConfig] = None,
        client: Optional[LLMClient] = None,
    ):
        self.config = config or VerifierConfig(retriever_type=retriever_type)
        self.client = client or create_client(mock=True)
        self._parser = SingleShotVerifier(config=self.config, client=self.client)
        self.corpus = corpus or []

        if retriever_type == "dense":
            self.retriever = DenseRetriever(self.corpus) if self.corpus else None
        elif retriever_type == "sparse":
            self.retriever = SparseRetriever(self.corpus) if self.corpus else None
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    def verify(self, item: VerificationItem) -> VerificationResult:
        """Verify a claim using RAG — retrieve context, then call LLM.

        Args:
            item: The claim to verify.

        Returns:
            VerificationResult with evidence from the corpus.
        """
        retrieve_start = time.time()
        if self.retriever and self.corpus:
            retrieved = self.retriever.retrieve(item.claim_text, k=3)
            context = "\n\n".join(retrieved)
        else:
            context = item.context or ""
        retrieve_time = time.time() - retrieve_start

        user_prompt = format_user_prompt(
            RAG_USER,
            claim_text=item.claim_text,
            context=context,
        )

        llm_start = time.time()
        raw_output = self.client.generate(
            system_prompt=RAG_SYSTEM,
            user_prompt=user_prompt,
            config=self.config,
        )
        llm_time = time.time() - llm_start

        result = self._parser._parse_response(
            raw_output, item.id, retrieve_time + llm_time
        )
        result.metadata["retrieve_time_s"] = round(retrieve_time, 3)
        result.metadata["llm_time_s"] = round(llm_time, 3)
        result.metadata["retriever_type"] = self.config.retriever_type
        result.evidence = [f"Retrieved context ({self.config.retriever_type}): {len(context)} chars"]

        return result

    def verify_batch(self, items: list[VerificationItem]) -> list[VerificationResult]:
        return [self.verify(item) for item in items]
