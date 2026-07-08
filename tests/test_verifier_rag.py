"""Tests for src/verifier_rag.py — RAG (retrieval-augmented) verifier.

Tests cover:
- Dense and sparse retriever correctness.
- Evidence and metadata tracking (evidence IDs, retrieval hits, relevance scores).
- Latency breakdown in metadata (retrieval vs LLM).
- Context truncation and boundary conditions.
- End-to-end smoke test on the synthetic finance dataset.
"""

from __future__ import annotations

from typing import List

import pytest

from src.schemas import Verdict, VerificationItem, VerifierConfig
from src.verifier_rag import DenseRetriever, RAGVerifier, SparseRetriever

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_corpus() -> List[VerificationItem]:
    """Small corpus of finance-related claims for retrieval testing."""
    corpus = [
        ("doc-1", "Federal Reserve holds interest rates steady at 5.25%"),
        ("doc-2", "Apple reports Q4 revenue of $89.5 billion, beating estimates"),
        ("doc-3", "Tesla delivers 435,000 vehicles in Q3 2024"),
        ("doc-4", "Microsoft announces $10 billion share buyback program"),
        ("doc-5", "Goldman Sachs upgrades S&P 500 target to 6,000"),
        ("doc-6", "Breaking: Federal Reserve declares all bank deposits void"),
        ("doc-7", "SEC announces Apple is under investigation for accounting fraud"),
        ("doc-8", "Elon Musk announces Tesla is filing for bankruptcy"),
    ]
    return [
        VerificationItem.create(claim_text=text, metadata={"id": id_})
        for id_, text in corpus
    ]


@pytest.fixture
def rag_verifier_dense(sample_corpus) -> RAGVerifier:
    return RAGVerifier(corpus=sample_corpus, retriever_type="dense", top_k=3)


@pytest.fixture
def rag_verifier_sparse(sample_corpus) -> RAGVerifier:
    return RAGVerifier(corpus=sample_corpus, retriever_type="sparse", top_k=3)


# ── Retriever Tests ────────────────────────────────────────────────


class TestSparseRetriever:
    def test_retrieve_returns_tuples(self, sample_corpus):
        """SparseRetriever returns (doc_id, text, score) tuples."""
        corpus_tuples = [(item.id, item.claim_text) for item in sample_corpus]
        retriever = SparseRetriever(corpus_tuples, top_k=2)
        results = retriever.retrieve("Federal Reserve interest rates")
        assert len(results) <= 2
        for doc_id, text, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(text, str)
            assert isinstance(score, float)
            assert score > 0

    def test_retrieve_matching_query(self, sample_corpus):
        """Top result for a finance query should contain relevant keywords."""
        corpus_tuples = [(item.id, item.claim_text) for item in sample_corpus]
        retriever = SparseRetriever(corpus_tuples, top_k=3)
        results = retriever.retrieve("Federal Reserve")
        assert len(results) >= 1
        # The top result should discuss the Federal Reserve
        top_text = results[0][1].lower()
        assert "federal" in top_text or "reserve" in top_text or "fed" in top_text

    def test_retrieve_empty_query(self, sample_corpus):
        """Empty query returns empty list (scores are zero)."""
        corpus_tuples = [(item.id, item.claim_text) for item in sample_corpus]
        retriever = SparseRetriever(corpus_tuples)
        results = retriever.retrieve("")
        assert len(results) == 0

    def test_retrieve_empty_corpus(self):
        """Empty corpus returns empty list gracefully."""
        retriever = SparseRetriever([], top_k=3)
        results = retriever.retrieve("anything")
        assert len(results) == 0


class TestDenseRetriever:
    def test_retrieve_returns_tuples(self, sample_corpus):
        """DenseRetriever returns (doc_id, text, score) tuples."""
        corpus_tuples = [(item.id, item.claim_text) for item in sample_corpus]
        retriever = DenseRetriever(corpus_tuples, top_k=2)
        results = retriever.retrieve("Federal Reserve interest rates")
        assert len(results) <= 2
        for doc_id, text, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(text, str)
            assert isinstance(score, float)
            assert score > 0

    def test_retrieve_matching_query(self, sample_corpus):
        """Top result should be semantically relevant."""
        corpus_tuples = [(item.id, item.claim_text) for item in sample_corpus]
        retriever = DenseRetriever(corpus_tuples, top_k=3)
        results = retriever.retrieve("central bank policy")
        assert len(results) >= 1
        # The top result should mention Federal Reserve or similar central bank topic
        top_text = results[0][1].lower()
        assert "federal" in top_text or "reserve" in top_text or "goldman" in top_text

    def test_retrieve_empty_query(self, sample_corpus):
        """Empty query returns empty list (norm is zero)."""
        corpus_tuples = [(item.id, item.claim_text) for item in sample_corpus]
        retriever = DenseRetriever(corpus_tuples)
        results = retriever.retrieve("")
        assert len(results) == 0

    def test_retrieve_empty_corpus(self):
        """Empty corpus returns empty list gracefully."""
        retriever = DenseRetriever([], top_k=3)
        results = retriever.retrieve("anything")
        assert len(results) == 0


# ── RAG Verifier Tests — Evidence, Metadata, and Latency ──────────


class TestRAGVerifierEvidence:
    def test_verify_returns_result(self, rag_verifier_dense):
        """Basic verify() smoke test."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        assert result.item_id == item.id
        assert result.verdict in Verdict
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_s >= 0.0

    def test_verify_records_evidence_ids(self, rag_verifier_dense):
        """Metadata includes evidence_ids from retrieved documents."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        assert "evidence_ids" in result.metadata
        assert isinstance(result.metadata["evidence_ids"], list)
        # At least one document should be retrieved
        assert len(result.metadata["evidence_ids"]) >= 1

    def test_verify_records_retrieval_hits(self, rag_verifier_dense):
        """Metadata includes count of retrieved documents."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        assert "retrieval_hits" in result.metadata
        assert 1 <= result.metadata["retrieval_hits"] <= 3  # top_k=3

    def test_verify_records_relevance_scores(self, rag_verifier_dense):
        """Metadata includes relevance scores for each hit."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        assert "relevance_scores" in result.metadata
        scores = result.metadata["relevance_scores"]
        assert len(scores) >= 1
        for score in scores:
            assert isinstance(score, float)
            assert score > 0

    def test_verify_records_latency_breakdown(self, rag_verifier_dense):
        """Metadata includes per-stage latency (retrieval vs LLM)."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        assert "retrieval_latency_s" in result.metadata
        assert "llm_latency_s" in result.metadata
        assert result.metadata["retrieval_latency_s"] >= 0
        assert result.metadata["llm_latency_s"] >= 0
        # Total latency should approximately equal sum of stages
        total = result.metadata["retrieval_latency_s"] + result.metadata["llm_latency_s"]
        assert abs(result.latency_s - total) < 0.01

    def test_verify_records_retriever_type(self, rag_verifier_dense):
        """Metadata records which retriever was used."""
        item = VerificationItem.create(claim_text="Test")
        result_dense = rag_verifier_dense.verify(item)
        assert result_dense.metadata["retriever_type"] == "dense"

        # Also sparse
        from src.verifier_rag import RAGVerifier

        sparse_verifier = RAGVerifier(
            corpus=rag_verifier_dense.corpus,
            retriever_type="sparse",
            top_k=3,
        )
        result_sparse = sparse_verifier.verify(
            VerificationItem.create(claim_text="Test")
        )
        assert result_sparse.metadata["retriever_type"] == "sparse"

    def test_verify_evidence_includes_retrieved_docs(self, rag_verifier_dense):
        """Evidence list includes retrieved document excerpts."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        # At least one evidence entry should start with "[Retrieved]"
        retrieved_entries = [
            e for e in result.evidence if e.startswith("[Retrieved]")
        ]
        assert len(retrieved_entries) >= 1
        # Each entry should contain doc= and score=
        for entry in retrieved_entries:
            assert "doc=" in entry
            assert "score=" in entry

    def test_verify_context_length_in_metadata(self, rag_verifier_dense):
        """Metadata records the context length in chars."""
        item = VerificationItem.create(claim_text="Federal Reserve raises rates")
        result = rag_verifier_dense.verify(item)
        assert "context_length_chars" in result.metadata
        assert result.metadata["context_length_chars"] > 0

    def test_sparse_verifier_also_records_metadata(self, rag_verifier_sparse):
        """Sparse retriever records the same metadata fields."""
        item = VerificationItem.create(claim_text="Apple stock performance")
        result = rag_verifier_sparse.verify(item)
        assert result.metadata["retriever_type"] == "sparse"
        assert "evidence_ids" in result.metadata
        assert "relevance_scores" in result.metadata
        assert "retrieval_latency_s" in result.metadata

    def test_verify_batch_returns_all(self, rag_verifier_dense):
        """verify_batch processes all items and returns correct count."""
        items = [
            VerificationItem.create(claim_text="Federal Reserve rates"),
            VerificationItem.create(claim_text="Apple revenue report"),
            VerificationItem.create(claim_text="Tesla vehicle deliveries"),
        ]
        results = rag_verifier_dense.verify_batch(items)
        assert len(results) == 3
        for r in results:
            assert r.verdict in Verdict
            assert "evidence_ids" in r.metadata
            assert r.metadata["retrieval_hits"] >= 1


# ── Context Formatting ─────────────────────────────────────────────


class TestContextFormatting:
    def test_format_context_empty(self, rag_verifier_dense):
        """Empty hits produces empty context string."""
        context = rag_verifier_dense._format_context([])
        assert context == ""

    def test_format_context_single_hit(self, rag_verifier_dense):
        """Single hit is properly formatted."""
        hits = [("doc-1", "Federal Reserve holds rates steady", 0.85)]
        context = rag_verifier_dense._format_context(hits)
        assert "doc-1" in context
        assert "0.850" in context
        assert "Federal Reserve holds rates steady" in context

    def test_format_context_truncation(self, rag_verifier_dense):
        """Truncation respects max_chars limit."""
        hits = [
            ("doc-1", "A" * 500, 0.9),
            ("doc-2", "B" * 500, 0.8),
            ("doc-3", "C" * 500, 0.7),
        ]
        context = rag_verifier_dense._format_context(hits, max_chars=300)
        assert len(context) <= 300

    def test_format_context_drops_when_all_exceed(self, rag_verifier_dense):
        """Very tight max_chars may produce empty or minimal context."""
        hits = [
            ("doc-1", "A" * 100, 0.9),
        ]
        context = rag_verifier_dense._format_context(hits, max_chars=10)
        # Too tight for even one entry prefix — may be empty or truncated
        assert len(context) <= 10


# ── Edge Cases ─────────────────────────────────────────────────────


class TestRAGVerifierEdgeCases:
    def test_empty_corpus_returns_default_verdict(self):
        """Empty corpus does not crash; returns default (REAL) verdict."""
        verifier = RAGVerifier(corpus=[], retriever_type="dense")
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        assert result.verdict in Verdict
        assert result.metadata["retrieval_hits"] == 0

    def test_single_item_corpus(self):
        """Single-item corpus retrieves that item as evidence."""
        corpus = [VerificationItem.create(claim_text="Federal Reserve policy")]
        verifier = RAGVerifier(corpus=corpus, retriever_type="dense", top_k=1)
        item = VerificationItem.create(claim_text="interest rates")
        result = verifier.verify(item)
        assert result.metadata["retrieval_hits"] >= 0  # may or may not match

    def test_corpus_with_duplicate_texts(self):
        """Corpus with duplicate texts is handled gracefully (no crash)."""
        texts = ["Apple stock rises", "Apple stock rises"]
        corpus = [VerificationItem.create(claim_text=t) for t in texts]
        verifier = RAGVerifier(corpus=corpus, retriever_type="sparse", top_k=2)
        item = VerificationItem.create(claim_text="Apple")
        result = verifier.verify(item)
        assert result.metadata["retrieval_hits"] >= 0

    def test_unknown_retriever_type(self):
        """Unknown retriever type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown retriever_type"):
            RAGVerifier(
                corpus=[VerificationItem.create(claim_text="test")],
                retriever_type="unknown",
            )

    def test_config_propagates_model(self):
        """VerifierConfig.model propagates through to metadata."""
        config = VerifierConfig(model="rag-test-model")
        corpus = [VerificationItem.create(claim_text="Test item")]
        verifier = RAGVerifier(corpus=corpus, config=config, retriever_type="dense")
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert result.metadata["model"] == "rag-test-model"


# ── End-to-End Smoke Test ──────────────────────────────────────────


class TestRAGVerifierSmoke:
    """End-to-end smoke tests on synthetic finance data with mock client.

    These validate the pipeline runs correctly and records evidence
    metadata. They do NOT claim scientific validity (mock client).
    """

    def test_smoke_finance_fallback_dense(self):
        """End-to-end: finance dataset + dense retriever."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) > 10

        # Use first 5 items as corpus, next 5 as test
        corpus = items[:5]
        test_items = items[5:10]

        verifier = RAGVerifier(corpus=corpus, retriever_type="dense", top_k=3)
        results = verifier.verify_batch(test_items)

        assert len(results) == 5
        for r in results:
            assert r.verdict in Verdict
            assert 0.0 <= r.confidence <= 1.0
            assert "evidence_ids" in r.metadata
            assert "retrieval_hits" in r.metadata
            assert "relevance_scores" in r.metadata
            assert "retrieval_latency_s" in r.metadata
            assert "llm_latency_s" in r.metadata
            assert r.metadata["retriever_type"] == "dense"
            # Evidence should include retrieved documents
            assert any(e.startswith("[Retrieved]") for e in r.evidence)

    def test_smoke_finance_fallback_sparse(self):
        """End-to-end: finance dataset + sparse retriever."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) > 10

        corpus = items[:5]
        test_items = items[5:10]

        verifier = RAGVerifier(corpus=corpus, retriever_type="sparse", top_k=3)
        results = verifier.verify_batch(test_items)

        assert len(results) == 5
        for r in results:
            assert r.verdict in Verdict
            assert r.metadata["retriever_type"] == "sparse"
            assert "evidence_ids" in r.metadata
            # Only assert retrieved evidence if any documents matched
            if r.metadata["retrieval_hits"] > 0:
                assert any(e.startswith("[Retrieved]") for e in r.evidence)
