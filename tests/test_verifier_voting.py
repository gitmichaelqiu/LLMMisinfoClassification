"""Tests for src/verifier_voting.py — voting ensemble verifier."""

from __future__ import annotations

from src.schemas import Verdict, VerificationItem, VerifierConfig
from src.verifier_voting import VotingVerifier


class TestVotingVerifier:
    def test_verify_returns_result(self):
        verifier = VotingVerifier(n_voters=3)
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        assert result.item_id == item.id
        assert result.verdict in Verdict
        assert 0.0 <= result.confidence <= 1.0

    def test_verify_batch_returns_all(self):
        verifier = VotingVerifier(n_voters=3)
        items = [
            VerificationItem.create(claim_text="A"),
            VerificationItem.create(claim_text="B"),
        ]
        results = verifier.verify_batch(items)
        assert len(results) == 2

    def test_majority_aggregation(self):
        """All mock voters return FAKE → majority is FAKE."""
        verifier = VotingVerifier(n_voters=5)
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert result.verdict == Verdict.FAKE  # MockClient always says FAKE
        assert result.metadata["n_voters"] == 5

    def test_unanimous_agreement(self):
        """All identical → unanimous resolves to that verdict."""
        verifier = VotingVerifier(n_voters=3, aggregation="unanimous")
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert result.verdict == Verdict.FAKE  # all mock say FAKE

    def test_metadata_includes_agreement_rate(self):
        verifier = VotingVerifier(n_voters=3)
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert "agreement_rate" in result.metadata
        assert result.metadata["agreement_rate"] == 1.0  # all mock identical

    def test_custom_n_voters(self):
        verifier = VotingVerifier(n_voters=7)
        assert verifier.n_voters == 7

    def test_config_propagates_model(self):
        config = VerifierConfig(model="custom-model")
        verifier = VotingVerifier(config=config, n_voters=3)
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert result.metadata["model"] == "custom-model"

    def test_smoke_on_finance_fallback(self):
        """End-to-end smoke test on synthetic finance data with mock client."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) == 10

        verifier = VotingVerifier(n_voters=3)
        results = verifier.verify_batch(items)

        assert len(results) == 10
        for r in results:
            assert r.verdict in Verdict
            assert 0.0 <= r.confidence <= 1.0
            assert "n_voters" in r.metadata
