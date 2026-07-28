"""Metrics computation for information verification."""

from __future__ import annotations

from typing import List, Tuple

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
    """Compute confusion matrix from verification results."""
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
    """Compute all standard classification metrics."""
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
    """Compute Expected Calibration Error."""
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
