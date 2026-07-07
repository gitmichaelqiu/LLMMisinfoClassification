"""Bayesian base-rate analysis and cost-sensitive decision thresholds.

Provides utilities for:
- Bayesian PPV / NPV computation across base rates
- Cost-sensitive decision thresholds (asymmetric loss)
- Expected cost calculations for verification strategies
- Stream-based base rate estimation from observed verdicts
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.schemas import Verdict


def bayesian_ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Compute Positive Predictive Value using Bayes' rule.

    PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    """
    fpr = 1.0 - specificity
    denom = sensitivity * prevalence + fpr * (1 - prevalence)
    return (sensitivity * prevalence) / denom if denom > 0 else 0.0


def bayesian_npv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Compute Negative Predictive Value using Bayes' rule."""
    fnr = 1.0 - sensitivity
    denom = specificity * (1 - prevalence) + fnr * prevalence
    return (specificity * (1 - prevalence)) / denom if denom > 0 else 0.0


def cost_sensitive_threshold(
    cost_fp: float,
    cost_fn: float,
    prevalence: float,
) -> float:
    """Compute the base-rate threshold where expected cost of acting equals
    expected cost of not acting, under asymmetric loss.

    The threshold P(fake) where:
        E[cost of intervene] = E[cost of hold]

    Assumes:
    - Intervening on fake news saves cost_fn (avoids FN loss)
    - Intervening on real news incurs cost_fp (FP cost)
    - Holding on real news costs 0
    - Holding on fake news incurs cost_fn

    Returns:
        Threshold prevalence where action flips.
    """
    if cost_fp <= 0 or cost_fn <= 0:
        return 0.5  # symmetric default
    # E[intervene] = P(fake) * 0 + P(real) * cost_fp
    # E[hold] = P(fake) * cost_fn + P(real) * 0
    # Threshold: P(fake) * cost_fn = P(real) * cost_fp
    # => prev * cost_fn = (1 - prev) * cost_fp
    # => prev = cost_fp / (cost_fp + cost_fn)
    return cost_fp / (cost_fp + cost_fn)


def ppv_curve(
    sensitivity: float,
    specificity: float,
    base_rates: Optional[List[float]] = None,
) -> List[Tuple[float, float, float]]:
    """Compute PPV/NPV across a sweep of base rates.

    Returns list of (base_rate, ppv, npv) tuples.
    """
    if base_rates is None:
        base_rates = list(np.logspace(-4, -0.301, 50))

    results = []
    for prev in base_rates:
        ppv = bayesian_ppv(sensitivity, specificity, prev)
        npv = bayesian_npv(sensitivity, specificity, prev)
        results.append((prev, ppv, npv))
    return results


def expected_cost(
    metrics: dict,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    prevalence: float = 0.05,
) -> float:
    """Compute expected per-item cost given classifier metrics and costs.

    Args:
        metrics: dict with 'fpr', 'fnr', 'n_total'.
        cost_fp: Cost of a false positive.
        cost_fn: Cost of a false negative.
        prevalence: Base rate of the positive class.

    Returns:
        Expected per-item cost.
    """
    n = metrics.get("n_total", 1)
    fpr = metrics.get("fpr", 0.0)
    fnr = metrics.get("fnr", 0.0)

    n_real = n * (1 - prevalence)
    n_fake = n * prevalence

    fp_cost = n_real * fpr * cost_fp
    fn_cost = n_fake * fnr * cost_fn

    return (fp_cost + fn_cost) / n


# ── Stream Base Rate Estimator ────────────────────────────────────


class BaseRateEstimator:
    """Rolling-window estimator for the misinformation base rate.

    Tracks observed verdicts over a sliding window and computes the
    empirical fraction of FAKE verdicts. Falls back to a configurable
    prior when insufficient data is available.

    Attributes:
        window_size: Maximum number of recent verdicts to track.
        default_prior: Prior base rate used when window is empty.
    """

    def __init__(self, default_prior: float = 0.05, window_size: int = 100):
        self.default_prior = default_prior
        self.window_size = window_size
        self._history: Deque[Verdict] = deque(maxlen=window_size)

    def update(self, verdict: Verdict) -> None:
        """Record an observed verdict in the rolling window.

        Args:
            verdict: The observed verdict (REAL, FAKE, ESCALATE, etc.).
        """
        self._history.append(verdict)

    def estimate(self) -> float:
        """Compute the current base rate estimate.

        Returns the empirical fraction of FAKE verdicts in the window.
        If the window is empty, returns the default_prior.

        Returns:
            Estimated misinformation base rate in [0, 1].
        """
        if not self._history:
            return self.default_prior

        fake_count = sum(1 for v in self._history if v == Verdict.FAKE)
        return fake_count / len(self._history)

    def reset(self, default_prior: Optional[float] = None) -> None:
        """Clear the history and optionally reset the prior.

        Args:
            default_prior: New prior (uses current prior if None).
        """
        self._history.clear()
        if default_prior is not None:
            self.default_prior = default_prior

    def __len__(self) -> int:
        return len(self._history)
