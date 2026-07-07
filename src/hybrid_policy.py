"""Hybrid policy for cost-sensitive decision making.

Given a verifier's output, an estimated misinformation base rate, and a
cost matrix (FP cost, FN cost), the hybrid policy selects the optimal action:
HOLD, HEDGE, REVERSE, or ESCALATE — to minimize expected loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.schemas import Verdict, VerificationResult


class RiskAction(Enum):
    """Risk management action selected by the hybrid policy."""

    HOLD = "hold"
    HEDGE = "hedge"
    REVERSE = "reverse"
    ESCALATE = "escalate"


@dataclass
class PolicyDecision:
    """Output of the hybrid policy for a single verification."""

    action: RiskAction
    confidence_threshold: float
    expected_cost_intervene: float
    expected_cost_hold: float
    base_rate: float


class HybridPolicy:
    """Cost-sensitive decision policy for verification actions.

    Selects action to minimize expected cost given:
    - Verifier output (confidence, verdict)
    - Estimated base rate of misinformation
    - Cost asymmetry (FP cost vs FN cost)

    Decision zones (configurable thresholds):
        HOLD zone:      P(fake) <= hold_threshold
        HEDGE zone:     hold_threshold < P(fake) < reverse_threshold
        REVERSE zone:   P(fake) >= reverse_threshold
        ESCALATE:       confidence below escalation_confidence_min
    """

    def __init__(
        self,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        hold_threshold: float = 0.35,
        reverse_threshold: float = 0.65,
        escalation_confidence_min: float = 0.3,
    ):
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.hold_threshold = hold_threshold
        self.reverse_threshold = reverse_threshold
        self.escalation_confidence_min = escalation_confidence_min

    def decide(
        self,
        result: VerificationResult,
        base_rate: float = 0.05,
    ) -> PolicyDecision:
        """Decide on an action given a verification result.

        Args:
            result: VerificationResult from any verifier.
            base_rate: Estimated base rate of misinformation.

        Returns:
            PolicyDecision with selected action and expected costs.
        """
        confidence = result.confidence  # P(fake) as estimated by the verifier
        verdict = result.verdict

        # If verdict is ESCALATE or confidence too low, escalate
        if verdict == Verdict.ESCALATE:
            return self._decision(RiskAction.ESCALATE, confidence, base_rate)

        if confidence < self.escalation_confidence_min:
            return self._decision(RiskAction.ESCALATE, confidence, base_rate)

        # Weight verifier confidence with base rate via simple Bayesian fusion
        p_fake = self._fuse_prior(confidence, base_rate)

        # Expected cost calculations
        # E[intervene] = P(real) * cost_fp  (intervening on real news costs)
        # E[hold]      = P(fake) * cost_fn   (holding on fake news costs)
        p_real = 1.0 - p_fake

        if p_fake <= self.hold_threshold:
            action = RiskAction.HOLD
        elif p_fake >= self.reverse_threshold:
            action = RiskAction.REVERSE
        else:
            action = RiskAction.HEDGE

        return PolicyDecision(
            action=action,
            confidence_threshold=p_fake,
            expected_cost_intervene=p_real * self.cost_fp,
            expected_cost_hold=p_fake * self.cost_fn,
            base_rate=base_rate,
        )

    def _fuse_prior(self, confidence: float, base_rate: float, alpha: float = 0.3) -> float:
        """Fuse verifier confidence with prior base rate.

        Uses a simple weighted average:
            P(fake) = alpha * base_rate + (1 - alpha) * confidence

        Args:
            confidence: Verifier's confidence score.
            base_rate: Prior base rate of misinformation.
            alpha: Weight given to the prior.

        Returns:
            Fused probability estimate.
        """
        return alpha * base_rate + (1 - alpha) * confidence

    def _decision(
        self,
        action: RiskAction,
        confidence: float,
        base_rate: float,
    ) -> PolicyDecision:
        p_fake = self._fuse_prior(confidence, base_rate)
        p_real = 1.0 - p_fake
        return PolicyDecision(
            action=action,
            confidence_threshold=p_fake,
            expected_cost_intervene=p_real * self.cost_fp,
            expected_cost_hold=p_fake * self.cost_fn,
            base_rate=base_rate,
        )
