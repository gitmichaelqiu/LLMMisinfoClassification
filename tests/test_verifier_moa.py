"""Tests for src/verifier_moa.py — MoA debate verifier with degeneracy diagnostics.

Mock mode: MockClient always returns the same fixed verdict regardless of
the prompt and role. This means Believer (pro-REAL) and Skeptic (pro-FAKE)
produce IDENTICAL output — the debate architecture is not exercised.

These tests verify:
1. Basic plumbing works (types, metadata, batch processing)
2. Degeneracy is DETECTED and DOCUMENTED (always-FAKE, always-REAL, zero
   information gain beyond base rate)
3. Degeneracy diagnostics compute correctly

ALL RESULTS IN THIS FILE ARE IMPLEMENTATION TESTS — NOT SCIENTIFIC FINDINGS.
Real MoA evaluation requires diverse LLM responses from actual API calls
where Believer and Skeptic genuinely disagree.
"""

from __future__ import annotations

from src.llm_clients import MockClient
from src.metrics import classification_metrics
from src.schemas import Verdict, VerificationItem, VerifierConfig
from src.verifier_moa import MoAVerifier


class TestMoAVerifierPlumbing:
    """Basic plumbing: does the verifier wire up and produce results?"""

    def test_verify_returns_result(self):
        """verify() returns a properly typed VerificationResult."""
        verifier = MoAVerifier()
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        assert result.item_id == item.id
        assert result.verdict in Verdict
        assert 0.0 <= result.confidence <= 1.0
        assert "verifier_type" in result.metadata
        assert result.metadata["verifier_type"] == "moa"

    def test_verify_batch_returns_all(self):
        """verify_batch processes all items."""
        verifier = MoAVerifier()
        items = [
            VerificationItem.create(claim_text="A"),
            VerificationItem.create(claim_text="B"),
            VerificationItem.create(claim_text="C"),
        ]
        results = verifier.verify_batch(items)
        assert len(results) == 3
        for r in results:
            assert r.verdict in Verdict

    def test_config_propagates_model(self):
        """VerifierConfig.model propagates through to result metadata."""
        config = VerifierConfig(model="custom-model")
        verifier = MoAVerifier(config=config)
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert result.metadata["model"] == "custom-model"


class TestMoAVerifierMetadata:
    """Metadata and evidence attachment."""

    def test_transcripts_attached(self):
        """Believer and Skeptic outputs are attached to metadata."""
        verifier = MoAVerifier()
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        assert "believer_output" in result.metadata
        assert "skeptic_output" in result.metadata
        # Both transcripts should be non-empty strings
        assert len(result.metadata["believer_output"]) > 0
        assert len(result.metadata["skeptic_output"]) > 0

    def test_evidence_has_confidence_lines(self):
        """Evidence includes extracted Believer and Skeptic confidence."""
        verifier = MoAVerifier()
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        assert any("Believer confidence" in e for e in result.evidence)
        assert any("Skeptic confidence" in e for e in result.evidence)

    def test_mock_confidence_extracted(self):
        """MockClient confidence (85) is extracted from both agents."""
        verifier = MoAVerifier()
        item = VerificationItem.create(claim_text="Test claim")
        result = verifier.verify(item)
        # Default MockClient has confidence=85
        assert any("85" in e for e in result.evidence)

    def test_rag_field_not_present(self):
        """MoA verifier should not include RAG-specific metadata."""
        verifier = MoAVerifier()
        item = VerificationItem.create(claim_text="Test")
        result = verifier.verify(item)
        assert "retrieved_context" not in result.metadata


class TestMoADegeneracyDiagnostics:
    """Degeneracy detection tests.

    In mock mode, MockClient returns the same verdict regardless of role
    prompt. This creates a degenerate scenario where Believer (pro-REAL)
    and Skeptic (pro-FAKE) output identical text — no actual debate occurs.

    These tests VERIFY that degeneracy IS detected, not that the verifier
    behaves non-degenerately in mock mode. Degeneracy is the EXPECTED
    outcome when the client doesn't vary by role.
    """

    def test_always_fake_detected(self):
        """DEGENERACY: default MockClient (FAKE) produces all-FAKE verdicts.

        Both Believer and Skeptic return "Verdict: FAKE" because MockClient
        ignores the role prompt. Risk Officer receives two FAKE analyses
        and outputs FAKE. Result: every item is classified FAKE regardless
        of content — a degenerate always-FAKE classifier.

        Detection: assert all predictions are FAKE and document that
        Believer/Skeptic show zero divergence.
        """
        verifier = MoAVerifier()
        items = [
            VerificationItem.create(claim_text="Claim A", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Claim B", ground_truth=Verdict.REAL),
        ]
        results = verifier.verify_batch(items)

        # All predictions are FAKE → degenerate always-FAKE
        verdicts = [r.verdict for r in results]
        assert all(v == Verdict.FAKE for v in verdicts), (
            f"Mock MoA predicted {verdicts} but expected all FAKE. "
            "This indicates MockClient did not behave as expected."
        )

        # Transcripts confirm zero divergence: Believer and Skeptic both say FAKE
        for r in results:
            assert "Verdict: FAKE" in r.metadata["believer_output"], (
                "Believer should output FAKE (MockClient ignores role prompt). "
                f"Got: {r.metadata['believer_output']}"
            )
            assert "Verdict: FAKE" in r.metadata["skeptic_output"], (
                "Skeptic should output FAKE (MockClient ignores role prompt). "
                f"Got: {r.metadata['skeptic_output']}"
            )

    def test_always_real_detected(self):
        """DEGENERACY: MockClient(fixed_verdict='REAL') produces all-REAL.

        With fixed_verdict=REAL, even the Skeptic (pro-FAKE) returns REAL.
        Recall on known-fake items is 0 — the verifier cannot detect
        misinformation.

        Detection: assert 0% recall on FAKE items and document that
        Skeptic contradicts its own role.
        """
        client = MockClient(fixed_verdict="REAL", fixed_confidence=85)
        verifier = MoAVerifier(client=client)

        items = [
            VerificationItem.create(claim_text="Definitely fake", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Actually real", ground_truth=Verdict.REAL),
        ]
        results = verifier.verify_batch(items)

        # All predictions are REAL
        verdicts = [r.verdict for r in results]
        assert all(v == Verdict.REAL for v in verdicts), (
            f"Expected all REAL with fixed_verdict=REAL, got {verdicts}"
        )

        # Skeptic (pro-FAKE role) also says REAL — contradiction of role is degenerate
        for r in results:
            assert "Verdict: REAL" in r.metadata["skeptic_output"], (
                "Skeptic should also output REAL (MockClient ignores role). "
                "This is a degeneracy signal: the skeptic role is not exercised."
            )

        # Recall on FAKE items is 0
        fake_results = [r for r, i in zip(results, items) if i.ground_truth == Verdict.FAKE]
        if fake_results:
            fake_recall = sum(1 for r in fake_results if r.verdict == Verdict.FAKE) / len(fake_results)
            assert fake_recall == 0.0, (
                f"Always-REAL degeneracy: recall on FAKE items is {fake_recall}. "
                "This confirms the verifier cannot detect misinformation when "
                "all agents say REAL."
            )

    def test_custom_mock_detected_as_degenerate(self):
        """DEGENERACY: custom mock verdicts are detected, not just defaults.

        Even with a non-standard verdict (ESCALATE), the debate architecture
        should produce identical outputs from all agents — still degenerate.
        """
        client = MockClient(fixed_verdict="ESCALATE", fixed_confidence=60)
        verifier = MoAVerifier(client=client)
        items = [
            VerificationItem.create(claim_text="Test claim", ground_truth=Verdict.FAKE),
        ]
        results = verifier.verify_batch(items)

        # All agents should return ESCALATE (MockClient ignores role)
        for r in results:
            assert "Verdict: ESCALATE" in r.metadata["believer_output"]
            assert "Verdict: ESCALATE" in r.metadata["skeptic_output"]

        # This is degenerate: Believer (pro-REAL) and Skeptic (pro-FAKE)
        # both say ESCALATE, which means neither role-specific reasoning
        # was exercised.


class TestMoAPrecisionVsBaseRate:
    """Compare MoA precision against the FAKE base rate P(FAKE).

    A non-degenerate verifier should have precision > P(FAKE) — it adds
    information beyond the unconditional prevalence of FAKE in the dataset.

    In mock mode with MockClient (always FAKE), the verifier predicts FAKE
    for every item. Therefore:
      - precision == P(FAKE) exactly (when dataset is balanced)
      - precision ≈ P(FAKE) (when dataset is imbalanced)
    This is the NULL INFORMATION case: the classifier adds zero information.

    These tests document the null-information baseline so that future runs
    with real API calls can be compared against it.
    """

    def test_precision_equals_base_rate_on_balanced_set(self):
        """DEGENERACY: precision == P(FAKE) on balanced dataset.

        With MockClient always-FAKE on a 50/50 REAL/FAKE set:
          TP = N_fake, FP = N_real
          precision = N_fake / (N_fake + N_real) = 0.5
          P(FAKE) = 0.5
          → precision == P(FAKE)
        """
        verifier = MoAVerifier()
        items = [
            VerificationItem.create(claim_text="Fake 1", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Fake 2", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Real 1", ground_truth=Verdict.REAL),
            VerificationItem.create(claim_text="Real 2", ground_truth=Verdict.REAL),
        ]
        results = verifier.verify_batch(items)
        ground_truths = [i.ground_truth for i in items]

        metrics = classification_metrics(results, ground_truths)
        fake_base_rate = sum(1 for gt in ground_truths if gt == Verdict.FAKE) / len(ground_truths)

        # Precision should equal P(FAKE) in mock always-FAKE mode
        assert abs(metrics.precision - fake_base_rate) < 0.01, (
            f"Mock MoA precision ({metrics.precision:.4f}) == P(FAKE) "
            f"({fake_base_rate:.4f}) within tolerance. This confirms the "
            f"verifier adds NO information beyond the base rate."
        )

        # The precision-base_rate gap documents information gain
        info_gap = metrics.precision - fake_base_rate
        assert info_gap <= 0.01, (
            f"Information gain (precision - P(FAKE)) = {info_gap:.4f}. "
            f"In mock mode, expected ≤ 0.01 (zero information gain). "
            f"A non-degenerate MoA should show info_gap >> 0."
        )

    def test_precision_approximates_base_rate_on_imbalanced_set(self):
        """DEGENERACY: precision ≈ P(FAKE) on imbalanced dataset.

        With 75% FAKE, precision should be ~0.75 (all predicted FAKE).
        """
        verifier = MoAVerifier()
        items = [
            VerificationItem.create(claim_text="Fake 1", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Fake 2", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Fake 3", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="Real 1", ground_truth=Verdict.REAL),
        ]
        results = verifier.verify_batch(items)
        ground_truths = [i.ground_truth for i in items]

        metrics = classification_metrics(results, ground_truths)
        fake_base_rate = sum(1 for gt in ground_truths if gt == Verdict.FAKE) / len(ground_truths)

        # All predicted FAKE → precision = 3/4 = 0.75 = P(FAKE)
        assert abs(metrics.precision - fake_base_rate) < 0.01, (
            f"Precision ({metrics.precision:.4f}) should equal P(FAKE) "
            f"({fake_base_rate:.4f}) in mock always-FAKE mode."
        )

    def test_info_gap_is_zero_in_mock_mode(self):
        """DEGENERACY: the information gain metric is explicitly zero.

        info_gap = precision - P(FAKE)

        In mock mode, this is always 0 (or near-0 with floating point).
        This is documented as the null-information baseline.
        """
        verifier = MoAVerifier()
        items = [
            VerificationItem.create(claim_text="A", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="B", ground_truth=Verdict.REAL),
            VerificationItem.create(claim_text="C", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="D", ground_truth=Verdict.REAL),
            VerificationItem.create(claim_text="E", ground_truth=Verdict.FAKE),
            VerificationItem.create(claim_text="F", ground_truth=Verdict.REAL),
        ]
        results = verifier.verify_batch(items)
        ground_truths = [i.ground_truth for i in items]

        metrics = classification_metrics(results, ground_truths)
        fake_base_rate = sum(1 for gt in ground_truths if gt == Verdict.FAKE) / len(ground_truths)
        info_gap = metrics.precision - fake_base_rate

        assert abs(info_gap) <= 0.01, (
            f"Null-information baseline: info_gap = {info_gap:.6f} "
            f"(precision={metrics.precision:.4f}, P(FAKE)={fake_base_rate:.4f}). "
            f"In mock mode, this should be ~0. Real MoA should show info_gap >> 0."
        )


class TestMoASmoke:
    """End-to-end smoke tests on actual (mock) data."""

    def test_smoke_on_finance_fallback(self):
        """End-to-end smoke test on synthetic finance data with mock client."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) > 10

        verifier = MoAVerifier()
        results = verifier.verify_batch(items[:10])

        assert len(results) == 10
        for r in results:
            assert r.verdict in Verdict
            assert 0.0 <= r.confidence <= 1.0
            assert r.metadata.get("verifier_type") == "moa"
            assert "believer_output" in r.metadata
            assert "skeptic_output" in r.metadata

    def test_smoke_with_conftest_fixtures(self, sample_items):
        """Run MoA verifier on the shared test fixtures."""
        verifier = MoAVerifier()
        results = verifier.verify_batch(sample_items)
        assert len(results) == 4
        for r in results:
            assert r.verdict in Verdict
            assert r.metadata.get("verifier_type") == "moa"
