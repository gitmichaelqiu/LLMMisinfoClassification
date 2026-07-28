"""Metrics aggregation and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import SEED
from src.metrics import classification_metrics, compute_ece
from src.schemas import Verdict, VerificationItem, VerificationResult


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "min": float(min(latencies)),
        "max": float(max(latencies)),
    }


def _f1_metric(
    results: list[VerificationResult], truths: list[Verdict]
) -> float:
    cm = classification_metrics(results, truths).confusion
    p = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    r = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _prec_metric(
    results: list[VerificationResult], truths: list[Verdict]
) -> float:
    cm = classification_metrics(results, truths).confusion
    return cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0


def _rec_metric(
    results: list[VerificationResult], truths: list[Verdict]
) -> float:
    cm = classification_metrics(results, truths).confusion
    return cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0


def _bootstrap_ci(
    items: list[VerificationItem],
    results: list[VerificationResult],
    metric_fn: Any,
    n_iter: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Paired percentile bootstrap CI for a metric."""
    truths = [it.ground_truth or Verdict.REAL for it in items]
    paired = [(r, t) for r, t in zip(results, truths)]
    if len(paired) < 2:
        return (0.0, 0.0)
    rng = np.random.RandomState(SEED)
    vals: list[float] = []
    for _ in range(n_iter):
        idx = rng.choice(len(paired), len(paired), replace=True)
        try:
            vals.append(
                metric_fn([paired[i][0] for i in idx], [paired[i][1] for i in idx])
            )
        except Exception:
            continue
    if len(vals) < 2:
        return (0.0, 0.0)
    return (
        float(np.percentile(vals, alpha / 2 * 100)),
        float(np.percentile(vals, (1 - alpha / 2) * 100)),
    )


def evaluate_architecture(
    items: list[VerificationItem],
    results: list[VerificationResult],
    arch_name: str,
    total_api_calls: int,
) -> dict[str, Any]:
    """Compute metrics, CIs, and latency for one architecture run."""
    truths = [it.ground_truth or Verdict.REAL for it in items]
    latencies = [r.latency_s for r in results if r.latency_s > 0]

    try:
        m = classification_metrics(results, truths)
        section: dict[str, Any] = {
            "metrics": {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "fpr": m.fpr,
                "fnr": m.fnr,
                "accuracy": m.accuracy,
                "n_total": m.n_total,
            },
        }
        n_esc = sum(1 for r in results if r.verdict == Verdict.ESCALATE)
        section["escalate_rate"] = n_esc / len(results) if results else 0.0
        correct = [r.verdict == t for r, t in zip(results, truths)]
        ece_val, _, _, _ = compute_ece(
            [r.confidence for r in results], correct, n_bins=5
        )
        section["ece"] = ece_val
        section["confidence_intervals"] = {
            "f1_95": list(_bootstrap_ci(items, results, _f1_metric)),
            "precision_95": list(_bootstrap_ci(items, results, _prec_metric)),
            "recall_95": list(_bootstrap_ci(items, results, _rec_metric)),
        }
    except Exception as e:
        section = {"metrics_error": str(e)}

    section["latency"] = _latency_stats(latencies)
    section["total_api_calls"] = total_api_calls
    return section
