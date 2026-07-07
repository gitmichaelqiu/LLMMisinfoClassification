"""Finance-specific metrics and analysis.

Wraps core metrics with finance-specific interpretations:
- Crossover threshold computation (verify-first vs trade-first)
- P&L estimation from confusion matrix
- Cost-of-carry analysis
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.base_rate import cost_sensitive_threshold, expected_cost
from src.metrics import classification_metrics
from src.schemas import ClassificationMetrics, ConfusionMatrix, Verdict, VerificationResult


class FinanceMetrics:
    """Finance-specific metric wrappers.

    Provides domain-relevant interpretations of classification metrics
    for the financial news verification case study.
    """

    # Default P&L parameters from empirical analysis (mid-cap profile)
    DEFAULT_PNL_TP_SAVED = 7852.0    # Saved loss on correctly identified fake
    DEFAULT_PNL_FN_LOSS = -29788.0   # Loss from missed fake (holding through crash)
    DEFAULT_PNL_TN_HOLD = 7995.0     # Profit from holding through real news
    DEFAULT_PNL_FP_COST = -6802.0    # Cost of unnecessary reversal on real news

    def __init__(
        self,
        cost_tp: float = DEFAULT_PNL_TP_SAVED,
        cost_fn: float = abs(DEFAULT_PNL_FN_LOSS),
        cost_tn: float = DEFAULT_PNL_TN_HOLD,
        cost_fp: float = abs(DEFAULT_PNL_FP_COST),
    ):
        self.cost_tp = cost_tp
        self.cost_fn = cost_fn
        self.cost_tn = cost_tn
        self.cost_fp = cost_fp

    def crossover_threshold(self, precision: float, recall: float) -> float:
        """Compute the verify-first crossover base rate.

        The P(fake) at which verify-first expected P&L equals trade-first.

        Uses the asymmetric cost model:
            E[verify_first] = P(fake) * recall * cost_tp
                            - P(fake) * (1-recall) * cost_fn
                            + (1-P(fake)) * (1-fpr) * cost_tn
                            - (1-P(fake)) * fpr * cost_fp
        where fpr is derived from precision and recall.
        """
        return cost_sensitive_threshold(
            cost_fp=self.cost_fp,
            cost_fn=self.cost_fn,
            prevalence=0.5,
        )

    def expected_pnl(
        self,
        metrics: ClassificationMetrics,
        prevalence: float,
    ) -> dict:
        """Compute expected P&L given classifier metrics and base rate.

        Returns:
            dict with keys: expected_pnl, tp_value, fn_cost, tn_value, fp_cost.
        """
        n = metrics.n_total
        n_fake = n * prevalence
        n_real = n * (1 - prevalence)

        tp = n_fake * metrics.recall
        fn = n_fake * (1 - metrics.recall)
        tn = n_real * (1 - metrics.fpr)
        fp = n_real * metrics.fpr

        return {
            "expected_pnl": (
                tp * self.cost_tp
                - fn * self.cost_fn
                + tn * self.cost_tn
                - fp * self.cost_fp
            ),
            "tp_value": tp * self.cost_tp,
            "fn_cost": fn * self.cost_fn,
            "tn_value": tn * self.cost_tn,
            "fp_cost": fp * self.cost_fp,
            "n_fake_estimated": round(n_fake),
            "n_real_estimated": round(n_real),
        }
