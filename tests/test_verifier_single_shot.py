"""Tests for src/verifier_single_shot.py — single-shot verifier."""

from __future__ import annotations

from src.llm_clients import MockClient
from src.schemas import Verdict, VerificationItem, VerifierConfig
from src.verifier_single_shot import SingleShotVerifier


class TestSingleShotVerifier:
    def test_verify_returns_result(self):
        verifier = SingleShotVerifier()
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        assert result.item_id == item.id
        assert result.verdict in Verdict
        assert 0.0 <= result.confidence <= 1.0
        assert result.latency_s >= 0.0

    def test_verify_with_context(self):
        verifier = SingleShotVerifier()
        item = VerificationItem.create(
            claim_text="Some news",
            context="Background context.",
        )
        result = verifier.verify(item)
        assert result.item_id == item.id

    def test_verify_batch_returns_all(self):
        verifier = SingleShotVerifier()
        items = [
            VerificationItem.create(claim_text="Claim A"),
            VerificationItem.create(claim_text="Claim B"),
            VerificationItem.create(claim_text="Claim C"),
        ]
        results = verifier.verify_batch(items)
        assert len(results) == 3
        assert all(r.item_id == items[i].id for i, r in enumerate(results))

    def test_mock_client_deterministic(self):
        client = MockClient(fixed_verdict="FAKE", fixed_confidence=85)
        verifier = SingleShotVerifier(client=client)
        item = VerificationItem.create(claim_text="Test")
        r1 = verifier.verify(item)
        r2 = verifier.verify(item)
        assert r1.verdict == r2.verdict == Verdict.FAKE
        assert r1.confidence == r2.confidence == 0.85

    def test_custom_system_prompt(self):
        verifier = SingleShotVerifier(
            system_prompt="You are a test verifier. Output only: Verdict: REAL",
        )
        item = VerificationItem.create(claim_text="Anything")
        verifier.verify(item)  # smoke: does not crash
        # MockClient ignores system prompt, so verdict is still FAKE

    def test_metadata_includes_model(self):
        config = VerifierConfig(model="test-model")
        verifier = SingleShotVerifier(config=config)
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert result.metadata["model"] == "test-model"

    def test_smoke_on_finance_fallback(self):
        """End-to-end smoke test on synthetic finance data with mock client."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) > 10

        verifier = SingleShotVerifier()
        results = verifier.verify_batch(items[:10])

        assert len(results) == 10
        for r in results:
            assert r.verdict in Verdict
            assert 0.0 <= r.confidence <= 1.0

        # Don't assert on accuracy — mock is not a real classifier.
        # This smoke test validates the pipeline runs end-to-end.


class TestParseResponse:
    """Unit tests for _parse_response."""

    def _parse(self, raw: str) -> tuple:
        verifier = SingleShotVerifier()
        result = verifier._parse_response(raw, "test-1", 0.5)
        return result.verdict, result.confidence, result.evidence, result.latency_s

    def test_parses_fake_verdict(self):
        raw = "Verdict: FAKE\nConfidence: 90\nFlags: [contradiction]\nReasoning: It is fake."
        verdict, conf, evidence, latency = self._parse(raw)
        assert verdict == Verdict.FAKE
        assert conf == 0.9
        assert latency == 0.5

    def test_parses_real_verdict(self):
        raw = "Verdict: REAL\nConfidence: 85\nFlags: [none]\nReasoning: Looks authentic."
        verdict, conf, *_ = self._parse(raw)
        assert verdict == Verdict.REAL
        assert conf == 0.85

    def test_parses_escalate(self):
        raw = "Verdict: ESCALATE\nConfidence: 40\nFlags: [inconsistency]\nReasoning: Unclear."
        verdict, *_ = self._parse(raw)
        assert verdict == Verdict.ESCALATE

    def test_defaults_on_malformed(self):
        raw = "Some random output without verdict tags."
        verdict, conf, evidence, _ = self._parse(raw)
        assert verdict == Verdict.REAL  # default
        assert conf == 0.5  # default
        assert evidence == []  # no flags/reasoning parsed

    def test_confidence_clamped(self):
        raw = "Verdict: FAKE\nConfidence: 150\nFlags: []\nReasoning: x"
        _, conf, *_ = self._parse(raw)
        assert conf == 1.0

    def test_flags_and_reasoning_extracted(self):
        raw = "Verdict: FAKE\nConfidence: 75\nFlags: [contradiction, implausibility]\nReasoning: Two red flags found."
        _, _, evidence, _ = self._parse(raw)
        assert any("contradiction, implausibility" in e for e in evidence)
        assert any("Two red flags found" in e for e in evidence)
