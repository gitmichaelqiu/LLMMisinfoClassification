"""TF-IDF retriever for Retrieval-Augmented Generation."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import VerificationItem

TOP_K = 5
MAX_EVIDENCE_CHARS = 2000


def build_retriever(
    corpus: list[VerificationItem],
) -> tuple[TfidfVectorizer, np.ndarray, list[str]]:
    """Fit a TF-IDF vectorizer on the corpus and return (vec, tfidf_matrix, texts)."""
    texts = [it.claim_text for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    return vec, vec.fit_transform(texts), texts


def _retrieve(
    claim: str,
    vec: TfidfVectorizer,
    tfidf: np.ndarray,
    texts: list[str],
    top_k: int = TOP_K,
) -> str:
    """Retrieve top-k most similar documents via cosine similarity."""
    qv = vec.transform([claim])
    sims = cosine_similarity(qv, tfidf).flatten()
    parts: list[str] = []
    for idx in np.argsort(sims)[::-1][:top_k]:
        if sims[idx] > 0:
            parts.append(f"[Doc {idx}] (rel={sims[idx]:.3f})\n{texts[idx]}\n")
    return "".join(parts)[:MAX_EVIDENCE_CHARS]


def make_user_prompt(
    claim_text: str,
    rag_vec=None,
    rag_tfidf=None,
    rag_texts=None,
) -> str:
    """Build the user prompt for Single-Shot / Voting (paper appendix format)."""
    if rag_vec is not None:
        evidence = _retrieve(claim_text, rag_vec, rag_tfidf, rag_texts)
        if evidence:
            return (
                f"Claim to verify:\n{claim_text}\n\n"
                f"Retrieved Evidence:\n{evidence}\n"
                f"Is this claim REAL or FAKE?"
            )
    return f"Claim to verify:\n{claim_text}\n\nIs this claim REAL or FAKE?"


def make_moa_user_prompt(
    claim_text: str,
    rag_vec=None,
    rag_tfidf=None,
    rag_texts=None,
) -> str:
    """Build the MoA user prompt. Differs from SS/Voting in header and wording."""
    if rag_vec is not None:
        evidence = _retrieve(claim_text, rag_vec, rag_tfidf, rag_texts)
        if evidence:
            return (
                f"Claim to analyze:\n{claim_text}\n\n"
                f"Retrieved Evidence. Use them to support your argument:\n{evidence}"
            )
    return f"Claim to analyze:\n{claim_text}"
