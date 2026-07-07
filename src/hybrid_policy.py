"""Hybrid policy for cost-sensitive decision making and verifier selection.

Provides:
- RiskAction enum (HOLD, HEDGE, REVERSE, ESCALATE)
- PolicyDecision with action, expected costs, and savings analysis
- VerifierProfile for meta-policy
- HybridPolicy: selects verifier → calls verifier → chooses action
  to minimise expected loss under asymmetric FP/FN costs

The meta-policy (select_verifier) chooses the cheapest verifier that
meets the required precision floor given the estimated base rate and
available latency budget.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from src.base_rate import BaseRateEstimator, cost_sensitive_threshold
from src.schemas import Verdict, VerificationResult


class RiskAction(Enum):
    """Risk management action selected by the hybrid policy."""

    HOLD = "hold"
    HEDGE = "hedge"
    REVERSE = "reverse"
    ESCALATE = "escalate"


@dataclass
class PolicyDecision:
    """Output of the hybrid policy for a single verification event.

    Attributes:
        action: Selected risk action.
        confidence_threshold: Fused P(fake) estimate used for the decision.
        expected_cost_intervene: Expected cost of the intervene action.
        expected_cost_hold: Expected cost of holding (not intervening).
        base_rate: Estimated base rate used.
        verifier_used: Name of the verifier selected by meta-policy.
        latency_budget_s: Available latency budget for this decision.
        cost_saved_vs_always_reverse: Per-item cost saved compared to
            always reversing.
        cost_saved_vs_always_hold: Per-item cost saved compared to
            always holding.
    """

    action: RiskAction
    confidence_threshold: float
    expected_cost_intervene: float
    expected_cost_hold: float
    base_rate: float
    verifier_used: str = "single-shot"
    latency_budget_s: float = 5.0
    cost_saved_vs_always_reverse: float = 0.0
    cost_saved_vs_always_hold: float = 0.0


@dataclass
class VerifierProfile:
    """Performance profile of a verifier architecture.

    Used by the meta-policy to select the best verifier given the
    operating context (base rate, latency budget, cost ratio).

    Attributes:
        name: Verifier identifier (e.g. "single-shot", "voting", "moa", "rag").
        latency_s: Expected wall-clock time per call.
        f1: Expected F1 score (implementation test values in mock mode).
        precision: Expected precision.
        recall: Expected recall.
        cost_per_call: Dollar cost per API call (0 in mock mode).
    """

    name: str
    latency_s: float
    f1: float
    precision: float
    recall: float
    cost_per_call: float = 0.0


# Default verifier profiles (mock-mode implementation values).
# In production these would be calibrated from real API runs.
DEFAULT_VERIFIER_PROFILES: Dict[str, VerifierProfile] = {
    "single-shot": VerifierProfile(
        name="single-shot",
        latency_s=0.001,
        f1=0.50,
        precision=0.50,
        recall=1.00,
    ),
    "voting": VerifierProfile(
        name="voting",
        latency_s=0.002,
        f1=0.50,
        precision=0.50,
        recall=1.00,
    ),
    "moa": VerifierProfile(
        name="moa",
        latency_s=0.003,
        f1=0.50,
        precision=0.50,
        recall=1.00,
    ),
    "rag": VerifierProfile(
        name="rag",
        latency_s=0.002,
        f1=0.50,
        precision=0.50,
        recall=1.00,
    ),
}


class HybridPolicy:
    """Cost-sensitive decision policy with verifier selection.

    Two-stage decision process:
    1. Meta-policy: select the cheapest verifier that meets precision
       and latency requirements for the current base rate.
    2. Action policy: given the verifier output, choose HOLD / HEDGE /
       REVERSE / ESCALATE to minimise expected cost under asymmetric loss.

    Attributes:
        cost_fp: Cost of a false positive (intervening on real news).
        cost_fn: Cost of a false negative (holding on fake news).
        verifier_profiles: Available verifiers with their performance profiles.
        latency_budget_s: Maximum acceptable latency per event.
        base_rate_estimator: Rolling-window estimator for P(fake).
        action_history: Counter of actions taken for distribution tracking.
    """

    def __init__(
        self,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        verifier_profiles: Optional[Dict[str, VerifierProfile]] = None,
        latency_budget_s: float = 5.0,
        base_rate_estimator: Optional[BaseRateEstimator] = None,
    ):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.verifier_profiles = verifier_profiles or DEFAULT_VERIFIER_PROFILES.copy()
        self.latency_budget_s = latency_budget_s
        self.base_rate_estimator = base_rate_estimator or BaseRateEstimator(
            default_prior=0.05, window_size=100
        )
        self.action_history: Counter[str] = Counter()

    # ── Meta-policy: verifier selection ─────────────────────────────

    def select_verifier(self, base_rate: Optional[float] = None) -> str:
        """Select the best verifier for the current context.

        Chooses the fastest verifier whose precision exceeds the
        base-rate-dependent minimum and whose latency fits the budget.

        Args:
            base_rate: Misinformation base rate. Estimated from stream
                       history if not provided.

        Returns:
            Verifier name string (key into self.verifier_profiles).
        """
        if base_rate is None:
            base_rate = self.base_rate_estimator.estimate()

        # Required precision: for a useful verifier, P(fake | alert) > base_rate
        # i.e. precision must exceed the base rate by a meaningful margin.
        min_precision = base_rate * 1.5  # 50% lift over random guessing
        min_precision = max(0.01, min(min_precision, 0.99))

        # Sort candidates by latency (fastest first) then filter
        candidates = sorted(
            self.verifier_profiles.values(),
            key=lambda p: p.latency_s,
        )

        selected = candidates[0].name  # fallback: fastest
        for profile in candidates:
            if profile.latency_s > self.latency_budget_s:
                continue  # too slow
            if profile.precision >= min_precision:
                selected = profile.name
                break  # first (fastest) acceptable candidate

        return selected

    # ── Base rate estimation ────────────────────────────────────────

    def observe_verdict(self, verdict: Verdict) -> None:
        """Record an observed verdict to update the base rate estimate.

        Args:
            verdict: The observed verdict.
        """
        self.base_rate_estimator.update(verdict)

    def estimate_base_rate(self) -> float:
        """Return the current estimated misinformation base rate.

        Returns:
            Base rate in [0, 1].
        """
        return self.base_rate_estimator.estimate()

    # ── Action decision ─────────────────────────────────────────────

    def decide(
        self,
        result: VerificationResult,
        base_rate: Optional[float] = None,
        verifier_name: Optional[str] = None,
    ) -> PolicyDecision:
        """Decide on an action given a verification result.

        Uses cost-sensitive thresholds from the asymmetric loss framework
        to set optimal decision boundaries. Records the decision for
        action distribution tracking.

        Args:
            result: VerificationResult from any verifier.
            base_rate: Estimated base rate (estimated if not provided).
            verifier_name: Verifier that produced this result (selected
                          via meta-policy if not provided).

        Returns:
            PolicyDecision with selected action and cost analysis.
        """
        if base_rate is None:
            base_rate = self.base_rate_estimator.estimate()

        if verifier_name is None:
            verifier_name = self.select_verifier(base_rate)

        confidence = result.confidence
        verdict = result.verdict

        # Compute optimal decision threshold from cost asymmetry
        optimal_threshold = cost_sensitive_threshold(
            cost_fp=self.cost_fp,
            cost_fn=self.cost_fn,
            prevalence=base_rate,
        )

        # Fuse verifier confidence with base rate
        p_fake = self._fuse_prior(confidence, base_rate)
        p_real = 1.0 - p_fake

        # ESCALATE on low confidence or explicit escalation
        if verdict == Verdict.ESCALATE or confidence < 0.3:
            action = RiskAction.ESCALATE
        elif p_fake <= self._hold_threshold(optimal_threshold):
            action = RiskAction.HOLD
        elif p_fake >= self._reverse_threshold(optimal_threshold):
            action = RiskAction.REVERSE
        else:
            action = RiskAction.HEDGE

        # Expected costs under the chosen action
        cost_intervene = p_real * self.cost_fp
        cost_hold = p_fake * self.cost_fn
        min_expected = min(cost_intervene, cost_hold)

        # Cost savings vs baselines (per-item)
        # "always reverse": incurs cost_fp on (1-base_rate) fraction of items
        cost_always_reverse = (1.0 - base_rate) * self.cost_fp
        # "always hold": incurs cost_fn on base_rate fraction of items
        cost_always_hold = base_rate * self.cost_fn
        savings_vs_reverse = cost_always_reverse - min_expected
        savings_vs_hold = cost_always_hold - min_expected

        decision = PolicyDecision(
            action=action,
            confidence_threshold=p_fake,
            expected_cost_intervene=cost_intervene,
            expected_cost_hold=cost_hold,
            base_rate=base_rate,
            verifier_used=verifier_name,
            latency_budget_s=self.latency_budget_s,
            cost_saved_vs_always_reverse=max(0.0, savings_vs_reverse),
            cost_saved_vs_always_hold=max(0.0, savings_vs_hold),
        )

        self.action_history[action.value] += 1
        return decision

    # ── Analysis helpers ────────────────────────────────────────────

    def action_distribution(self) -> Dict[str, float]:
        """Return the distribution of actions taken so far.

        Returns:
            Dict mapping action name to fraction of total decisions (0-1).
        """
        total = sum(self.action_history.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.action_history.items()}

    def reset_history(self) -> None:
        """Clear the action history counter."""
        self.action_history.clear()

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _hold_threshold(optimal_threshold: float, width: float = 0.15) -> float:
        """Compute the HOLD zone upper bound.

        Centred on the optimal threshold, with a HEDGE buffer zone.

        Args:
            optimal_threshold: The cost-sensitive decision threshold.
            width: Half-width of the HEDGE zone on each side.

        Returns:
            Upper bound of the HOLD zone.
        """
        return max(0.0, optimal_threshold - width)

    @staticmethod
    def _reverse_threshold(optimal_threshold: float, width: float = 0.15) -> float:
        """Compute the REVERSE zone lower bound.

        Args:
            optimal_threshold: The cost-sensitive decision threshold.
            width: Half-width of the HEDGE zone on each side.

        Returns:
            Lower bound of the REVERSE zone.
        """
        return min(1.0, optimal_threshold + width)

    @staticmethod
    def _fuse_prior(confidence: float, base_rate: float, alpha: float = 0.3) -> float:
        """Fuse verifier confidence with prior base rate.

        Uses a weighted average:
            P(fake) = alpha * base_rate + (1 - alpha) * confidence

        Args:
            confidence: Verifier's confidence score.
            base_rate: Prior base rate of misinformation.
            alpha: Weight given to the prior.

        Returns:
            Fused probability estimate.
        """
        return alpha * base_rate + (1 - alpha) * confidence
