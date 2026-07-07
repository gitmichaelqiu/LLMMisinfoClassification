"""Metrics computation for information verification.

Provides:
- Confusion matrix (integer counts)
- Classification metrics (precision, recall, F1, FPR, FNR)
- Calibration metrics (ECE)
- Bayesian PPV / NPV curves
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.schemas import (
    ClassificationMetrics,
    ConfusionMatrix,
    Verdict,
    VerificationResult,
)


def compute_confusion_matrix(
    results: List[VerificationResult],
    ground_truths: List[Verdict],
    positive_class: Verdict = Verdict.FAKE,
) -> ConfusionMatrix:
    """Compute confusion matrix from verification results.

    Args:
        results: List of verifier outputs.
        ground_truths: Corresponding ground-truth labels.
        positive_class: Which Verdict is treated as the positive class.

    Returns:
        ConfusionMatrix with integer counts.
    """
    cm = ConfusionMatrix()
    for result, truth in zip(results, ground_truths):
        pred = result.verdict
        if pred == positive_class and truth == positive_class:
            cm.tp += 1
        elif pred == positive_class and truth != positive_class:
            cm.fp += 1
        elif pred != positive_class and truth != positive_class:
            cm.tn += 1
        elif pred != positive_class and truth == positive_class:
            cm.fn += 1
    return cm


def classification_metrics(
    results: List[VerificationResult],
    ground_truths: List[Verdict],
    positive_class: Verdict = Verdict.FAKE,
) -> ClassificationMetrics:
    """Compute all standard classification metrics.

    Returns:
        ClassificationMetrics with precision, recall, F1, FPR, FNR, accuracy.
    """
    cm = compute_confusion_matrix(results, ground_truths, positive_class)
    n = cm.tp + cm.fp + cm.tn + cm.fn

    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0
    fnr = cm.fn / (cm.fn + cm.tp) if (cm.fn + cm.tp) > 0 else 0.0
    accuracy = (cm.tp + cm.tn) / n if n > 0 else 0.0

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        accuracy=accuracy,
        n_total=n,
        confusion=cm,
    )


def compute_ece(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> Tuple[float, List[float], List[float], List[int]]:
    """Compute Expected Calibration Error.

    Args:
        confidences: Predicted confidence scores in [0, 1].
        correct: Whether each prediction was correct.
        n_bins: Number of equal-width bins.

    Returns:
        Tuple of (ece, bin_confidences, bin_accuracies, bin_counts).
    """
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            bin_conf = float(confidences[mask].mean())
            bin_acc = float(correct[mask].mean())
            ece += (count / len(confidences)) * abs(bin_acc - bin_conf)
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    return ece, bin_confidences, bin_accuracies, bin_counts


def compute_ppv(
    sensitivity: float,
    specificity: float,
    base_rates: Optional[List[float]] = None,
) -> List[Tuple[float, float, float]]:
    """Compute Bayesian Positive Predictive Value across base rates.

    PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

    Args:
        sensitivity: True positive rate (recall).
        specificity: 1 - false positive rate.
        base_rates: List of base rates to evaluate. Defaults to
            log-spaced values from 0.0001 to 0.5.

    Returns:
        List of (base_rate, ppv, npv) tuples.
    """
    if base_rates is None:
        base_rates = list(np.logspace(-4, -0.301, 50))  # 0.01% to 50%

    fpr = 1.0 - specificity
    results = []
    for prev in base_rates:
        denominator_ppv = sensitivity * prev + fpr * (1 - prev)
        ppv = (sensitivity * prev) / denominator_ppv if denominator_ppv > 0 else 0.0

        denominator_npv = specificity * (1 - prev) + (1 - sensitivity) * prev
        npv = (specificity * (1 - prev)) / denominator_npv if denominator_npv > 0 else 0.0

        results.append((prev, ppv, npv))

    return results
