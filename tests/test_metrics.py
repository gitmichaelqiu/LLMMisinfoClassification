"""Tests for src/metrics.py — metrics computation."""

from __future__ import annotations

import pytest

from src.metrics import (
    classification_metrics,
    compute_confusion_matrix,
    compute_ece,
    compute_ppv,
)
from src.schemas import (
    ConfusionMatrix,
    Verdict,
    VerificationResult,
)


def _result(verdict: Verdict, confidence: float = 0.9) -> VerificationResult:
    return VerificationResult(
        item_id="t",
        verdict=verdict,
        confidence=confidence,
        latency_s=0.1,
    )


class TestComputeConfusionMatrix:
    def test_all_correct(self):
        results = [_result(Verdict.FAKE), _result(Verdict.REAL)]
        truths = [Verdict.FAKE, Verdict.REAL]
        cm = compute_confusion_matrix(results, truths)
        assert cm == ConfusionMatrix(tp=1, fp=0, tn=1, fn=0)

    def test_all_wrong(self):
        results = [_result(Verdict.FAKE), _result(Verdict.REAL)]
        truths = [Verdict.REAL, Verdict.FAKE]
        cm = compute_confusion_matrix(results, truths)
        assert cm == ConfusionMatrix(tp=0, fp=1, tn=0, fn=1)

    def test_mixed(self):
        results = [
            _result(Verdict.FAKE),
            _result(Verdict.FAKE),
            _result(Verdict.REAL),
            _result(Verdict.REAL),
            _result(Verdict.FAKE),
        ]
        truths = [Verdict.FAKE, Verdict.REAL, Verdict.REAL, Verdict.FAKE, Verdict.FAKE]
        cm = compute_confusion_matrix(results, truths)
        assert cm == ConfusionMatrix(tp=2, fp=1, tn=1, fn=1)

    def test_empty_lists(self):
        cm = compute_confusion_matrix([], [])
        assert cm == ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)

    def test_positive_class_real(self):
        results = [_result(Verdict.REAL), _result(Verdict.REAL), _result(Verdict.FAKE)]
        truths = [Verdict.REAL, Verdict.FAKE, Verdict.FAKE]
        cm = compute_confusion_matrix(results, truths, positive_class=Verdict.REAL)
        assert cm == ConfusionMatrix(tp=1, fp=1, tn=1, fn=0)

    def test_escalate_verdict_not_counted(self):
        results = [
            _result(Verdict.ESCALATE),
            _result(Verdict.REAL),
            _result(Verdict.FAKE),
        ]
        truths = [Verdict.FAKE, Verdict.REAL, Verdict.FAKE]
        cm = compute_confusion_matrix(results, truths)
        assert cm == ConfusionMatrix(tp=1, fp=0, tn=1, fn=1)


class TestClassificationMetrics:
    def test_perfect_classifier(self):
        results = [_result(Verdict.FAKE), _result(Verdict.REAL)]
        truths = [Verdict.FAKE, Verdict.REAL]
        m = classification_metrics(results, truths)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.fpr == 0.0
        assert m.fnr == 0.0
        assert m.accuracy == 1.0
        assert m.n_total == 2

    def test_known_values(self):
        results = [
            _result(Verdict.FAKE),
            _result(Verdict.FAKE),
            _result(Verdict.FAKE),
            _result(Verdict.REAL),
            _result(Verdict.REAL),
        ]
        truths = [Verdict.FAKE, Verdict.FAKE, Verdict.REAL, Verdict.REAL, Verdict.FAKE]
        m = classification_metrics(results, truths)
        assert m.confusion == ConfusionMatrix(tp=2, fp=1, tn=1, fn=1)
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(2 / 3)
        assert m.f1 == pytest.approx(2 / 3)
        assert m.fpr == pytest.approx(0.5)
        assert m.fnr == pytest.approx(1 / 3)
        assert m.accuracy == pytest.approx(3 / 5)
        assert m.n_total == 5

    def test_zero_precision(self):
        results = [_result(Verdict.FAKE), _result(Verdict.FAKE)]
        truths = [Verdict.REAL, Verdict.REAL]
        m = classification_metrics(results, truths)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.fpr == 1.0

    def test_zero_false_positives(self):
        results = [_result(Verdict.REAL), _result(Verdict.REAL), _result(Verdict.FAKE)]
        truths = [Verdict.REAL, Verdict.REAL, Verdict.FAKE]
        m = classification_metrics(results, truths)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.fpr == 0.0

    def test_empty(self):
        m = classification_metrics([], [])
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.fpr == 0.0
        assert m.fnr == 0.0
        assert m.accuracy == 0.0
        assert m.n_total == 0


class TestComputeECE:
    def test_perfect_calibration(self):
        confidences = [0.5, 0.5, 0.5, 0.5]
        correct = [True, False, True, False]
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_systematic_overconfidence(self):
        confidences = [0.9, 0.9, 0.9, 0.9]
        correct = [True, True, False, False]
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.4, abs=1e-6)

    def test_systematic_underconfidence(self):
        confidences = [0.3, 0.3, 0.3, 0.3]
        correct = [True, True, True, True]
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.7, abs=1e-6)

    def test_empty(self):
        ece, bin_conf, bin_acc, bin_counts = compute_ece([], [], n_bins=5)
        assert ece == 0.0
        assert len(bin_counts) == 5

    def test_bin_boundaries(self):
        confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
        correct = [True, True, False, True, True]
        ece, bin_conf, bin_acc, bin_counts = compute_ece(confidences, correct, n_bins=4)
        assert sum(bin_counts) == 4


class TestComputePPV:
    def test_perfect_sensitivity_specificity(self):
        points = compute_ppv(sensitivity=1.0, specificity=1.0)
        for prev, ppv, npv in points:
            assert ppv == pytest.approx(1.0)
            assert npv == pytest.approx(1.0)

    def test_bayes_formula(self):
        points = compute_ppv(sensitivity=0.8, specificity=0.9, base_rates=[0.1])
        prev, ppv, npv = points[0]
        assert ppv == pytest.approx(0.08 / 0.17)
        assert npv == pytest.approx(0.81 / 0.83)

    def test_low_base_rate(self):
        points = compute_ppv(sensitivity=0.9, specificity=0.95, base_rates=[0.001])
        _, ppv, npv = points[0]
        assert ppv < 0.05
        assert npv > 0.99

    def test_default_base_rates(self):
        points = compute_ppv(sensitivity=0.5, specificity=0.5)
        assert len(points) == 50
        first_prev, _, _ = points[0]
        last_prev, _, _ = points[-1]
        assert first_prev == pytest.approx(0.0001, abs=1e-5)
        assert last_prev == pytest.approx(0.5, abs=5e-4)

    def test_random_guessing(self):
        points = compute_ppv(sensitivity=0.5, specificity=0.5, base_rates=[0.1, 0.5])
        for prev, ppv, npv in points:
            assert ppv == pytest.approx(prev)
            assert npv == pytest.approx(1 - prev)

    def test_zero_sensitivity(self):
        points = compute_ppv(sensitivity=0.0, specificity=1.0, base_rates=[0.1, 0.5])
        for prev, ppv, npv in points:
            assert ppv == 0.0
            assert npv == pytest.approx(1.0 - prev)
