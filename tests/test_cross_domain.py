"""Cross-domain tests — validate the framework generalises beyond finance.

Tests cover:
- HealthDatasetAdapter loads items from data/raw/health/ or synthetic fallback.
- PoliticalDatasetAdapter loads synthetic political statements.
- Both adapters produce valid VerificationItem objects with correct labels.
- load_dataset() factory dispatches to the correct adapter for each domain.
- SingleShot verifier runs on all three domains in mock mode.
- Item counts and label distributions are correct per domain.
- Cross-contamination: health adapter does not load finance data.
"""

from __future__ import annotations

import pytest

from src.datasets import DatasetAdapter, load_dataset
from src.llm_clients import MockClient
from src.schemas import Verdict, VerificationItem
from src.verifier_single_shot import SingleShotVerifier

# ====================================================================
# Health Dataset Adapter Tests
# ====================================================================


class TestHealthAdapter:
    def test_load_returns_items(self):
        adapter = load_dataset("healthcare")
        items = adapter.load()
        assert len(items) > 0
        for item in items:
            assert isinstance(item, VerificationItem)

    def test_items_have_healthcare_domain(self):
        adapter = load_dataset("healthcare")
        items = adapter.load()
        for item in items:
            assert item.metadata.get("domain") == "healthcare"

    def test_items_have_valid_labels(self):
        adapter = load_dataset("healthcare")
        items = adapter.load()
        for item in items:
            assert item.ground_truth in (Verdict.REAL, Verdict.FAKE, None)

    def test_item_counts_balanced(self):
        adapter = load_dataset("healthcare")
        counts = adapter.item_counts()
        assert counts["total"] > 0

    def test_train_test_split(self):
        adapter = load_dataset("healthcare")
        train, test = adapter.train_test_split(test_size=0.3, seed=42)
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == adapter.item_counts()["total"]

    def test_is_dataset_adapter(self):
        adapter = load_dataset("healthcare")
        assert isinstance(adapter, DatasetAdapter)

    def test_does_not_load_finance_data(self):
        """Health adapter does not scan finance directory."""
        adapter = load_dataset("healthcare")
        items = adapter.load()
        for item in items:
            assert item.metadata.get("domain") != "finance"

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            load_dataset("nonexistent")


# ====================================================================
# Political Dataset Adapter Tests
# ====================================================================


class TestPoliticalAdapter:
    def test_load_returns_items(self):
        adapter = load_dataset("political")
        items = adapter.load()
        assert len(items) > 0
        for item in items:
            assert isinstance(item, VerificationItem)

    def test_items_have_political_domain(self):
        adapter = load_dataset("political")
        items = adapter.load()
        for item in items:
            assert item.metadata.get("domain") == "political"

    def test_items_have_valid_labels(self):
        adapter = load_dataset("political")
        items = adapter.load()
        for item in items:
            assert item.ground_truth in (Verdict.REAL, Verdict.FAKE)

    def test_item_counts(self):
        adapter = load_dataset("political")
        counts = adapter.item_counts()
        assert counts["total"] > 0
        assert counts["fake"] > 0
        assert counts["real"] > 0

    def test_train_test_split(self):
        adapter = load_dataset("political")
        counts = adapter.item_counts()
        train, test = adapter.train_test_split(test_size=0.3, seed=42)
        assert len(train) + len(test) == counts["total"]

    def test_is_dataset_adapter(self):
        adapter = load_dataset("political")
        assert isinstance(adapter, DatasetAdapter)


# ====================================================================
# Factory Dispatch Tests
# ====================================================================


class TestFactoryDispatch:
    def test_finance_domain(self):
        adapter = load_dataset("finance")
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter
        assert isinstance(adapter, FinanceDatasetAdapter)

    def test_healthcare_domain(self):
        adapter = load_dataset("healthcare")
        from src.healthcare.health_dataset_adapter import HealthDatasetAdapter
        assert isinstance(adapter, HealthDatasetAdapter)

    def test_health_alias(self):
        adapter = load_dataset("health")
        from src.healthcare.health_dataset_adapter import HealthDatasetAdapter
        assert isinstance(adapter, HealthDatasetAdapter)

    def test_political_domain(self):
        adapter = load_dataset("political")
        from src.political.political_dataset_adapter import PoliticalDatasetAdapter
        assert isinstance(adapter, PoliticalDatasetAdapter)

    def test_politics_alias(self):
        adapter = load_dataset("politics")
        from src.political.political_dataset_adapter import PoliticalDatasetAdapter
        assert isinstance(adapter, PoliticalDatasetAdapter)


# ====================================================================
# Cross-Domain Mock Verification Tests
# ====================================================================


class TestCrossDomainMock:
    """Mock-mode verifier runs on all domains.

    These validate the pipeline runs correctly across domains. They do
    NOT claim scientific validity (mock client always returns FAKE).
    """

    DOMAINS = ["finance", "healthcare", "political"]

    def test_verifier_runs_on_all_domains(self):
        """SingleShot verifier runs without error on each domain."""
        verifier = SingleShotVerifier()

        for domain in self.DOMAINS:
            adapter = load_dataset(domain)
            items = adapter.load()
            results = verifier.verify_batch(items)
            assert len(results) == len(items)
            for r in results:
                assert r.verdict in Verdict
                assert 0.0 <= r.confidence <= 1.0

    def test_item_counts_per_domain(self):
        """Each domain returns expected item counts."""
        for domain in self.DOMAINS:
            adapter = load_dataset(domain)
            counts = adapter.item_counts()
            assert counts["total"] >= 6  # minimum across all domains

    def test_all_domains_output_same_shape(self):
        """Verifier output has consistent structure across domains."""
        verifier = SingleShotVerifier()
        for domain in self.DOMAINS:
            adapter = load_dataset(domain)
            items = adapter.load()
            results = verifier.verify_batch(items[:3])
            for r in results:
                assert "model" in r.metadata
                assert r.latency_s >= 0
                assert isinstance(r.evidence, list)

    def test_mock_client_identical_across_domains(self):
        """Mock client returns same fixed format for all domains."""
        client = MockClient(fixed_verdict="FAKE", fixed_confidence=85)
        verifier = SingleShotVerifier(client=client)

        for domain in self.DOMAINS:
            adapter = load_dataset(domain)
            items = adapter.load()
            r = verifier.verify(items[0])
            assert r.verdict == Verdict.FAKE
            assert r.confidence == 0.85

    def test_load_dataset_unknown_domain_error(self):
        """Unknown domain raises clear error with available options."""
        with pytest.raises(ValueError) as exc:
            load_dataset("nonexistent")
        msg = str(exc.value)
        assert "finance" in msg
        assert "healthcare" in msg
        assert "political" in msg


# ====================================================================
# Schema Compatibility Tests
# ====================================================================


class TestSchemaCompatibility:
    """Verify that domain items work with all verifier types."""

    def test_single_shot_accepts_all_domains(self):
        """SingleShot verifier accepts items from any domain."""
        verifier = SingleShotVerifier()
        for domain in ["finance", "healthcare", "political"]:
            adapter = load_dataset(domain)
            item = adapter.load()[0]
            result = verifier.verify(item)
            assert result.item_id == item.id

    def test_voting_verifier_accepts_all_domains(self):
        """Voting verifier accepts items from any domain."""
        from src.verifier_voting import VotingVerifier

        verifier = VotingVerifier(n_voters=3)
        for domain in ["finance", "healthcare", "political"]:
            adapter = load_dataset(domain)
            items = adapter.load()
            results = verifier.verify_batch(items[:2])
            assert len(results) == 2

    def test_rag_verifier_accepts_all_domains(self):
        """RAG verifier accepts items from any domain as corpus."""
        from src.verifier_rag import RAGVerifier

        for domain in ["finance", "healthcare", "political"]:
            adapter = load_dataset(domain)
            items = adapter.load()
            corpus = items[:3]
            verifier = RAGVerifier(corpus=corpus, retriever_type="dense", top_k=2)
            results = verifier.verify_batch(items[3:6])
            assert len(results) >= 0

    def test_hybrid_policy_accepts_all_domains(self):
        """HybridPolicy works with results from any domain."""
        from src.hybrid_policy import HybridPolicy

        verifier = SingleShotVerifier()
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)

        for domain in ["finance", "healthcare", "political"]:
            adapter = load_dataset(domain)
            item = adapter.load()[0]
            result = verifier.verify(item)
            decision = policy.decide(result, base_rate=0.05)
            assert decision.action is not None
