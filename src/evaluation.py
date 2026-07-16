"""Metrics aggregation, bootstrap confidence intervals, PPV, and voter analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.config import COST_RATIOS, PPV_BASE_RATES, SEED
from src.metrics import classification_metrics, compute_ece
from src.schemas import Verdict, VerificationItem, VerificationResult


# ── Latency statistics ──────────────────────────────────────────
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


# ── Bootstrap CI helpers ────────────────────────────────────────
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


# ── PPV and expected cost curves ────────────────────────────────
def compute_ppv_curve(
    sens: float, fpr: float, base_rates: list[float]
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for br in base_rates:
        denom = sens * br + fpr * (1 - br)
        ppv = round((sens * br) / denom, 6) if denom > 0 else 0.0
        results.append({"base_rate": br, "ppv": ppv})
    return results


def compute_expected_cost(
    fpr: float,
    fnr: float,
    base_rates: list[float],
    cfp: float,
    cfn: float,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    for br in base_rates:
        cost = round((1 - br) * fpr * cfp + br * fnr * cfn, 6)
        results.append({"base_rate": br, "expected_cost": cost})
    return results


# ── Full architecture evaluation ────────────────────────────────
def evaluate_architecture(
    items: list[VerificationItem],
    results: list[VerificationResult],
    arch_name: str,
    total_api_calls: int,
) -> dict[str, Any]:
    """Compute all metrics, CIs, PPV, and expected cost for one architecture run."""
    truths = [it.ground_truth or Verdict.REAL for it in items]
    latencies = [r.latency_s for r in results if r.latency_s > 0]

    section: dict[str, Any] = {
        "n": len(items),
        "n_fake": sum(1 for it in items if it.ground_truth == Verdict.FAKE),
        "n_real": sum(1 for it in items if it.ground_truth == Verdict.REAL),
    }

    try:
        m = classification_metrics(results, truths)
        section["metrics"] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "fpr": m.fpr,
            "fnr": m.fnr,
            "accuracy": m.accuracy,
            "n_total": m.n_total,
        }
        section["verdict_distribution"] = dict(
            Counter(r.verdict.name for r in results)
        )
        n_esc = sum(1 for r in results if r.verdict == Verdict.ESCALATE)
        section["escalate_rate"] = n_esc / len(results) if results else 0.0
        correct = [r.verdict == t for r, t in zip(results, truths)]
        ece_val, _, _, _ = compute_ece(
            [r.confidence for r in results], correct, n_bins=5
        )
        section["ece"] = ece_val
        section["ppv"] = compute_ppv_curve(m.recall, m.fpr, PPV_BASE_RATES)
        section["expected_cost"] = {
            f"FP{cfp}_FN{cfn}": compute_expected_cost(
                m.fpr, m.fnr, PPV_BASE_RATES, cfp, cfn
            )
            for cfp, cfn in COST_RATIOS
        }
        section["confidence_intervals"] = {
            "f1_95": list(_bootstrap_ci(items, results, _f1_metric)),
            "precision_95": list(_bootstrap_ci(items, results, _prec_metric)),
            "recall_95": list(_bootstrap_ci(items, results, _rec_metric)),
        }
    except Exception as e:
        section["metrics_error"] = str(e)

    section["latency"] = _latency_stats(latencies)
    section["total_api_calls"] = total_api_calls
    return section


# ── Voter agreement analysis ────────────────────────────────────
def analyze_voter_agreement(
    items: list[VerificationItem],
    per_item_voters: list[list[VerificationResult]],
    n_voters: int,
) -> dict[str, Any]:
    """Analyze agreement patterns across *n_voters* per item."""
    truths = [it.ground_truth or Verdict.REAL for it in items]
    stats: dict[str, Any] = {
        "total_items": len(items),
        "n_voters": n_voters,
        "unanimous": 0,
        "majority_flip_v1": 0,
        "corrected_by_voting": 0,
        "damaged_by_voting": 0,
        "pairwise_agreement": 0.0,
        "majority_verdicts": {},
        "voter1_verdicts": {},
        "voter_fake_counts": [],
    }
    pairwise_agree = 0
    pairwise_total = 0
    threshold = n_voters // 2 + 1

    for i, voters in enumerate(per_item_voters):
        if not voters or len(voters) < n_voters:
            continue
        gt = truths[i] if i < len(truths) else None
        verdicts = [v.verdict for v in voters]
        v1 = verdicts[0]
        if len(set(verdicts)) == 1:
            stats["unanimous"] += 1
        counter = Counter(verdicts)
        majority_v, majority_c = counter.most_common(1)[0]
        if majority_c >= threshold:
            stats["majority_verdicts"][majority_v.name] = (
                stats["majority_verdicts"].get(majority_v.name, 0) + 1
            )
        else:
            stats["majority_verdicts"]["ESCALATE"] = (
                stats["majority_verdicts"].get("ESCALATE", 0) + 1
            )
        stats["voter1_verdicts"][v1.name] = (
            stats["voter1_verdicts"].get(v1.name, 0) + 1
        )
        maj_v = majority_v if majority_c >= threshold else Verdict.ESCALATE
        if (
            v1 != Verdict.ESCALATE
            and maj_v != Verdict.ESCALATE
            and maj_v != v1
            and n_voters > 1
        ):
            stats["majority_flip_v1"] += 1
        for a in range(len(verdicts)):
            for b in range(a + 1, len(verdicts)):
                pairwise_total += 1
                if verdicts[a] == verdicts[b]:
                    pairwise_agree += 1
        if gt is not None:
            if v1 != gt and maj_v == gt:
                stats["corrected_by_voting"] += 1
            if v1 == gt and maj_v != gt and n_voters > 1:
                stats["damaged_by_voting"] += 1

    stats["pairwise_agreement"] = (
        pairwise_agree / pairwise_total if pairwise_total > 0 else 0
    )
    for vi in range(n_voters):
        stats["voter_fake_counts"].append(
            sum(
                1
                for voters in per_item_voters
                if vi < len(voters) and voters[vi].verdict == Verdict.FAKE
            )
        )
    return stats
