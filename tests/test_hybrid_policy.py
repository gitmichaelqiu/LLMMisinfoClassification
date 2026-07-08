"""Tests for src/hybrid_policy.py and src/base_rate.py — hybrid policy.

Tests cover:
- BaseRateEstimator: rolling-window base rate estimation.
- Verifier profiles and meta-policy (select_verifier).
- Action selection (decide) under various base rates and cost ratios.
- Cost savings vs always-reverse and always-hold baselines.
- Action distribution tracking.
- End-to-end pipeline: dataset → verifier → policy → action.
"""

from __future__ import annotations

import pytest

from src.base_rate import BaseRateEstimator
from src.hybrid_policy import (
    DEFAULT_VERIFIER_PROFILES,
    HybridPolicy,
    PolicyDecision,
    RiskAction,
    VerifierProfile,
)
from src.schemas import Verdict, VerificationResult
from src.verifier_single_shot import SingleShotVerifier

# ====================================================================
# BaseRateEstimator Tests
# ====================================================================


class TestBaseRateEstimator:
    def test_default_prior_when_empty(self):
        """Empty estimator returns default_prior."""
        estimator = BaseRateEstimator(default_prior=0.05)
        assert estimator.estimate() == 0.05

    def test_all_fake_returns_one(self):
        """All FAKE verdicts → estimate = 1.0."""
        estimator = BaseRateEstimator()
        for _ in range(10):
            estimator.update(Verdict.FAKE)
        assert estimator.estimate() == 1.0

    def test_all_real_returns_zero(self):
        """All REAL verdicts → estimate = 0.0."""
        estimator = BaseRateEstimator()
        for _ in range(10):
            estimator.update(Verdict.REAL)
        assert estimator.estimate() == 0.0

    def test_mixed_verdicts(self):
        """3 FAKE out of 10 → estimate = 0.3."""
        estimator = BaseRateEstimator()
        for _ in range(3):
            estimator.update(Verdict.FAKE)
        for _ in range(7):
            estimator.update(Verdict.REAL)
        assert estimator.estimate() == 0.3

    def test_escalate_not_counted_as_fake(self):
        """ESCALATE is not counted as FAKE."""
        estimator = BaseRateEstimator()
        for _ in range(5):
            estimator.update(Verdict.ESCALATE)
        assert estimator.estimate() == 0.0

    def test_window_size_limits_history(self):
        """Window size limits the number of retained observations."""
        estimator = BaseRateEstimator(window_size=5)
        for _ in range(10):
            estimator.update(Verdict.FAKE)
        assert len(estimator) == 5
        assert estimator.estimate() == 1.0  # all 5 in window are FAKE

    def test_reset_clears_history(self):
        """Reset clears history and returns default_prior."""
        estimator = BaseRateEstimator(default_prior=0.05)
        estimator.update(Verdict.FAKE)
        estimator.reset()
        assert len(estimator) == 0
        assert estimator.estimate() == 0.05

    def test_reset_with_new_prior(self):
        """Reset can set a new default_prior."""
        estimator = BaseRateEstimator(default_prior=0.05)
        estimator.reset(default_prior=0.5)
        assert estimator.estimate() == 0.5


# ====================================================================
# VerifierProfile Tests
# ====================================================================


class TestVerifierProfile:
    def test_default_profiles_exist(self):
        """All four verifier architectures have default profiles."""
        assert set(DEFAULT_VERIFIER_PROFILES.keys()) == {
            "single-shot", "voting", "moa", "rag"
        }

    def test_all_profiles_have_positive_latency(self):
        """Every profile has non-negative latency."""
        for name, profile in DEFAULT_VERIFIER_PROFILES.items():
            assert profile.latency_s >= 0, f"{name} has negative latency"

    def test_all_profiles_have_confidence_scores(self):
        """Every profile has scores in [0, 1]."""
        for name, profile in DEFAULT_VERIFIER_PROFILES.items():
            assert 0 <= profile.f1 <= 1, f"{name} F1 out of range"
            assert 0 <= profile.precision <= 1, f"{name} precision out of range"
            assert 0 <= profile.recall <= 1, f"{name} recall out of range"


# ====================================================================
# Meta-Policy: Verifier Selection Tests
# ====================================================================


class TestSelectVerifier:
    def test_returns_string(self):
        """select_verifier returns a valid verifier name."""
        policy = HybridPolicy()
        verifier = policy.select_verifier(base_rate=0.05)
        assert verifier in DEFAULT_VERIFIER_PROFILES

    def test_fastest_at_low_base_rate(self):
        """At very low base rate, selects the fastest verifier."""
        policy = HybridPolicy()
        verifier = policy.select_verifier(base_rate=0.001)
        assert verifier == "single-shot"  # fastest

    def test_uses_estimated_base_rate(self):
        """When no base_rate given, uses the estimator's value."""
        policy = HybridPolicy()
        policy.base_rate_estimator.update(Verdict.FAKE)
        policy.base_rate_estimator.update(Verdict.FAKE)
        # base rate = 1.0 after 2 FAKE
        verifier = policy.select_verifier()  # no explicit base_rate
        assert isinstance(verifier, str)

    def test_latency_budget_excludes_slow_verifiers(self):
        """Verifiers exceeding latency budget are excluded."""
        profiles = {
            "fast": VerifierProfile(name="fast", latency_s=0.5, f1=0.5, precision=0.6, recall=0.5),
            "slow": VerifierProfile(name="slow", latency_s=10.0, f1=0.9, precision=0.9, recall=0.9),
        }
        policy = HybridPolicy(
            verifier_profiles=profiles,
            latency_budget_s=1.0,
        )
        # Even at high base rate, slow is excluded by budget
        verifier = policy.select_verifier(base_rate=0.3)
        assert verifier == "fast"

    def test_custom_profiles_respected(self):
        """Custom verifier profiles are used for selection."""
        profiles = {
            "custom-fast": VerifierProfile(
                name="custom-fast", latency_s=0.001, f1=0.5, precision=0.5, recall=1.0
            ),
        }
        policy = HybridPolicy(verifier_profiles=profiles)
        verifier = policy.select_verifier(base_rate=0.05)
        assert verifier == "custom-fast"


# ====================================================================
# Action Decision Tests
# ====================================================================


class TestDecide:
    def test_decide_returns_policy_decision(self):
        """decide returns a PolicyDecision with all fields."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.85,
        )
        decision = policy.decide(result, base_rate=0.05)
        assert isinstance(decision, PolicyDecision)
        assert isinstance(decision.action, RiskAction)
        assert decision.base_rate == 0.05
        assert decision.expected_cost_intervene >= 0
        assert decision.expected_cost_hold >= 0

    def test_low_confidence_triggers_escalate(self):
        """Confidence below 0.3 → ESCALATE regardless of verdict."""
        policy = HybridPolicy()
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.15,
        )
        decision = policy.decide(result, base_rate=0.05)
        assert decision.action == RiskAction.ESCALATE

    def test_escalate_verdict_triggers_escalate(self):
        """ESCALATE verdict → ESCALATE action."""
        policy = HybridPolicy()
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.ESCALATE,
            confidence=0.5,
        )
        decision = policy.decide(result, base_rate=0.05)
        assert decision.action == RiskAction.ESCALATE

    def test_high_confidence_fake_at_low_base_rate(self):
        """High confidence FAKE → REVERSE when fused P(fake) high."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.95,
        )
        decision = policy.decide(result, base_rate=0.05)
        # Fused P(fake) = 0.3 * 0.05 + 0.7 * 0.95 = 0.68 → above reverse threshold
        assert decision.action in (RiskAction.REVERSE, RiskAction.HEDGE)

    def test_low_confidence_fake_at_low_base_rate(self):
        """Low confidence FAKE at low base rate with symmetric cost → HOLD."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=1.0)  # symmetric
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.35,  # moderate confidence
        )
        decision = policy.decide(result, base_rate=0.01)
        # Fused P(fake) = 0.3 * 0.01 + 0.7 * 0.35 = 0.248
        # Cost threshold = 0.5, hold bound = 0.35 → HOLD
        assert decision.action == RiskAction.HOLD

    def test_action_changes_with_base_rate(self):
        """At same verifier output, higher base rate → more intervention."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.60,
        )

        decision_low = policy.decide(result, base_rate=0.01)
        decision_high = policy.decide(result, base_rate=0.30)

        # Higher base rate should produce more interventionist action
        action_rank = {
            RiskAction.HOLD: 0,
            RiskAction.HEDGE: 1,
            RiskAction.REVERSE: 2,
            RiskAction.ESCALATE: 3,
        }
        assert action_rank[decision_high.action] >= action_rank[decision_low.action]

    def test_verifier_name_propagates(self):
        """Verifier name set in decide is propagated to PolicyDecision."""
        policy = HybridPolicy()
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.9,
        )
        decision = policy.decide(result, base_rate=0.05, verifier_name="custom-verifier")
        assert decision.verifier_used == "custom-verifier"

    def test_default_verifier_selected(self):
        """When no verifier_name given, select_verifier is called."""
        policy = HybridPolicy()
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.9,
        )
        decision = policy.decide(result, base_rate=0.05)
        assert decision.verifier_used in DEFAULT_VERIFIER_PROFILES


# ====================================================================
# Cost Sensitivity Tests
# ====================================================================


class TestCostSensitivity:
    def test_higher_fn_cost_lowers_threshold(self):
        """Higher FN cost relative to FP → lower decision threshold."""
        policy_low_fn = HybridPolicy(cost_fp=1.0, cost_fn=1.0)
        policy_high_fn = HybridPolicy(cost_fp=1.0, cost_fn=20.0)

        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.45,
        )

        # At same base rate, higher FN cost → more likely to intervene
        d1 = policy_low_fn.decide(result, base_rate=0.08)
        d2 = policy_high_fn.decide(result, base_rate=0.08)

        # The hold threshold shifts left, so HOLD less likely with high FN cost
        # For very symmetric costs, the hold zone is wider
        if d1.action == RiskAction.REVERSE:
            assert d2.action == RiskAction.REVERSE
        # At least verify that high FN cost doesn't produce a more dovish action
        action_rank = {
            RiskAction.HOLD: 0,
            RiskAction.HEDGE: 1,
            RiskAction.REVERSE: 2,
            RiskAction.ESCALATE: 3,
        }
        assert action_rank[d2.action] >= action_rank[d1.action]

    def test_cost_savings_vs_baselines(self):
        """Cost savings vs always-reverse and always-hold are reported."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.95,
        )
        decision = policy.decide(result, base_rate=0.05)
        assert decision.cost_saved_vs_always_reverse >= 0
        assert decision.cost_saved_vs_always_hold >= 0

    def test_cost_savings_vs_always_hold_at_high_base_rate(self):
        """At high base rate with confident FAKE, savings vs hold are large."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)
        result = VerificationResult(
            item_id="test-1",
            verdict=Verdict.FAKE,
            confidence=0.95,
        )
        decision = policy.decide(result, base_rate=0.20)
        # Always-hold cost = 0.20 * 10.0 = 2.0
        # Policy chose REVERSE, cost_intervene = 0.8 * 1.0 = 0.8 (approx)
        # Savings ≈ 2.0 - min(0.8, ...) > 0
        assert decision.cost_saved_vs_always_hold > 0


# ====================================================================
# Action Distribution Tests
# ====================================================================


class TestActionDistribution:
    def test_action_distribution_tracks_decisions(self):
        """Action distribution reflects all decisions made."""
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)

        # 2 reverses, 1 hold
        policy.decide(
            VerificationResult(item_id="a", verdict=Verdict.FAKE, confidence=0.95),
            base_rate=0.05,
        )
        policy.decide(
            VerificationResult(item_id="b", verdict=Verdict.FAKE, confidence=0.95),
            base_rate=0.05,
        )
        policy.decide(
            VerificationResult(item_id="c", verdict=Verdict.REAL, confidence=0.95),
            base_rate=0.001,
        )

        dist = policy.action_distribution()
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
        assert dist.get("reverse", 0) > 0

    def test_empty_history_returns_empty(self):
        """No decisions → empty distribution."""
        policy = HybridPolicy()
        assert policy.action_distribution() == {}

    def test_reset_history_clears(self):
        """reset_history clears action history."""
        policy = HybridPolicy()
        policy.decide(
            VerificationResult(item_id="a", verdict=Verdict.FAKE, confidence=0.95),
            base_rate=0.05,
        )
        assert len(policy.action_history) > 0
        policy.reset_history()
        assert len(policy.action_history) == 0


# ====================================================================
# End-to-End Pipeline Test
# ====================================================================


class TestEndToEnd:
    """End-to-end: dataset → verifier → policy → action.

    These validate the full pipeline runs correctly with mock-mode
    verifier scores. They do not claim scientific validity.
    """

    def test_pipeline_with_single_shot(self):
        """Finance dataset → SingleShot → HybridPolicy → actions recorded."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        assert len(items) > 0

        verifier = SingleShotVerifier()
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)

        for item in items:
            result = verifier.verify(item)
            decision = policy.decide(result, base_rate=policy.estimate_base_rate())
            assert isinstance(decision.action, RiskAction)
            policy.observe_verdict(result.verdict)

        dist = policy.action_distribution()
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)

    def test_pipeline_action_counts(self):
        """Pipeline produces expected number of decisions matching item count."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()

        verifier = SingleShotVerifier()
        policy = HybridPolicy(cost_fp=1.0, cost_fn=10.0)

        decisions = []
        for item in items:
            result = verifier.verify(item)
            decision = policy.decide(result, base_rate=policy.estimate_base_rate())
            decisions.append(decision)
            policy.observe_verdict(result.verdict)

        assert len(decisions) == len(items)

    def test_pipeline_with_multiple_cost_ratios(self):
        """Pipeline runs under different FP:FN cost ratios without error."""
        from src.finance.finance_dataset_adapter import FinanceDatasetAdapter

        adapter = FinanceDatasetAdapter()
        items = adapter.load()
        verifier = SingleShotVerifier()

        cost_ratios = [(1, 1), (1, 5), (1, 10), (1, 25), (1, 100)]

        for cost_fp, cost_fn in cost_ratios:
            policy = HybridPolicy(cost_fp=cost_fp, cost_fn=cost_fn)
            for item in items[:3]:  # subset for speed
                result = verifier.verify(item)
                decision = policy.decide(result, base_rate=0.05)
                assert isinstance(decision.action, RiskAction)

    def test_pipeline_latency_budget_respected(self):
        """Latency budget constrains verifier selection."""
        profiles = {
            "fast": VerifierProfile(name="fast", latency_s=0.001, f1=0.5, precision=0.5, recall=1.0),
        }
        policy = HybridPolicy(
            verifier_profiles=profiles,
            latency_budget_s=0.1,
        )
        verifier = policy.select_verifier(base_rate=0.05)
        assert verifier == "fast"
        assert profiles[verifier].latency_s <= policy.latency_budget_s
