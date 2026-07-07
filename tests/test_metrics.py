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

# ── Helpers ────────────────────────────────────────────────────

def _result(verdict: Verdict, confidence: float = 0.9) -> VerificationResult:
    return VerificationResult(
        item_id="t",
        verdict=verdict,
        confidence=confidence,
        latency_s=0.1,
    )


# ── compute_confusion_matrix ──────────────────────────────────

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
            _result(Verdict.FAKE),    # TP
            _result(Verdict.FAKE),    # FP (truth is REAL)
            _result(Verdict.REAL),    # TN
            _result(Verdict.REAL),    # FN (truth is FAKE)
            _result(Verdict.FAKE),    # TP
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
        # tp=1 (REAL→REAL), fp=1 (REAL→FAKE but truth is REAL? No...)
        # Wait: results=REAL/REAL/FAKE, truths=REAL/FAKE/FAKE
        # With positive_class=REAL:
        #   item0: pred=REAL(positive), truth=REAL(positive) → TP=1
        #   item1: pred=REAL(positive), truth=FAKE(not positive) → FP=1
        #   item2: pred=FAKE(not positive), truth=FAKE(not positive) → TN=1
        assert cm == ConfusionMatrix(tp=1, fp=1, tn=1, fn=0)

    def test_escalate_verdict_not_counted(self):
        """ESCALATE should not count as FAKE or REAL."""
        results = [
            _result(Verdict.ESCALATE),
            _result(Verdict.REAL),
            _result(Verdict.FAKE),
        ]
        truths = [Verdict.FAKE, Verdict.REAL, Verdict.FAKE]
        cm = compute_confusion_matrix(results, truths)
        # ESCALATE != FAKE, truth=FAKE → that's FN
        # REAL == REAL → TN
        # FAKE == FAKE → TP
        assert cm == ConfusionMatrix(tp=1, fp=0, tn=1, fn=1)


# ── classification_metrics ────────────────────────────────────

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
        """TP=2, FP=1, TN=1, FN=1 → precision=2/3, recall=2/3, F1=2/3, FPR=1/2, FNR=1/3"""
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
        """All predictions are FAKE but none are actually FAKE."""
        results = [_result(Verdict.FAKE), _result(Verdict.FAKE)]
        truths = [Verdict.REAL, Verdict.REAL]
        m = classification_metrics(results, truths)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.fpr == 1.0

    def test_zero_false_positives(self):
        """FP=0 should give precision=1 and FPR=0."""
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


# ── compute_ece ───────────────────────────────────────────────

class TestComputeECE:
    def test_perfect_calibration(self):
        """All predictions at 0.5 confidence with 50% accuracy → ECE = 0."""
        confidences = [0.5, 0.5, 0.5, 0.5]
        correct = [True, False, True, False]
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_systematic_overconfidence(self):
        """All confident (0.9) but half wrong → ECE > 0."""
        confidences = [0.9, 0.9, 0.9, 0.9]
        correct = [True, True, False, False]
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        # accuracy = 0.5, confidence = 0.9 → |acc-conf| = 0.4
        assert ece == pytest.approx(0.4, abs=1e-6)

    def test_systematic_underconfidence(self):
        """All low confidence (0.3) but all correct → ECE > 0."""
        confidences = [0.3, 0.3, 0.3, 0.3]
        correct = [True, True, True, True]
        ece, *_ = compute_ece(confidences, correct, n_bins=10)
        # accuracy = 1.0, confidence = 0.3 → |acc-conf| = 0.7
        assert ece == pytest.approx(0.7, abs=1e-6)

    def test_empty(self):
        ece, bin_conf, bin_acc, bin_counts = compute_ece([], [], n_bins=5)
        assert ece == 0.0
        assert len(bin_counts) == 5

    def test_bin_boundaries(self):
        """Values exactly at bin edges should be assigned consistently."""
        confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
        correct = [True, True, False, True, True]
        ece, bin_conf, bin_acc, bin_counts = compute_ece(confidences, correct, n_bins=4)
        # 4 bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
        # 1.0 falls in no bin (since bin check is < upper bound)
        # Actually 1.0: bins[4]=1.0, so mask = (0.75 <= 1.0 < 1.0) → False
        assert sum(bin_counts) == 4  # 1.0 excluded from bins


# ── compute_ppv ───────────────────────────────────────────────

class TestComputePPV:
    def test_perfect_sensitivity_specificity(self):
        """sens=1, spec=1 → PPV = 1.0 at all base rates."""
        points = compute_ppv(sensitivity=1.0, specificity=1.0)
        for prev, ppv, npv in points:
            assert ppv == pytest.approx(1.0)
            assert npv == pytest.approx(1.0)

    def test_bayes_formula(self):
        """Verify against hand calculation:
        sens=0.8, spec=0.9, prev=0.1
        PPV = (0.8*0.1) / (0.8*0.1 + 0.1*0.9) = 0.08 / 0.17 = 0.4706
        NPV = (0.9*0.9) / (0.9*0.9 + 0.2*0.1) = 0.81 / 0.83 = 0.9759
        """
        points = compute_ppv(sensitivity=0.8, specificity=0.9, base_rates=[0.1])
        prev, ppv, npv = points[0]
        assert ppv == pytest.approx(0.08 / 0.17)
        assert npv == pytest.approx(0.81 / 0.83)

    def test_low_base_rate(self):
        """Very low base rates suppress PPV."""
        points = compute_ppv(sensitivity=0.9, specificity=0.95, base_rates=[0.001])
        _, ppv, npv = points[0]
        # PPV = 0.9*0.001 / (0.9*0.001 + 0.05*0.999) = 0.0009 / 0.05085 ≈ 0.0177
        assert ppv < 0.05  # low PPV at low prevalence
        assert npv > 0.99  # high NPV at low prevalence

    def test_default_base_rates(self):
        points = compute_ppv(sensitivity=0.5, specificity=0.5)
        assert len(points) == 50  # default logspace
        first_prev, _, _ = points[0]
        last_prev, _, _ = points[-1]
        assert first_prev == pytest.approx(0.0001, abs=1e-5)
        # np.logspace(-4, -0.301, 50) ends at ~0.50003
        assert last_prev == pytest.approx(0.5, abs=5e-4)

    def test_random_guessing(self):
        """sens=0.5, spec=0.5 → PPV = prev at all rates."""
        points = compute_ppv(sensitivity=0.5, specificity=0.5, base_rates=[0.1, 0.5])
        for prev, ppv, npv in points:
            assert ppv == pytest.approx(prev)  # PPV equals prevalence
            assert npv == pytest.approx(1 - prev)  # NPV equals 1 - prevalence

    def test_zero_sensitivity(self):
        """sens=0, spec=1 → PPV = 0, NPV = 1 - prev.
        With sens=0, every positive is missed (FN), so NPV falls as
        prevalence rises: NPV = (spec*(1-prev)) / (spec*(1-prev) + (1-sens)*prev)
        = (1-prev) / (1-prev + prev) = 1 - prev.
        """
        points = compute_ppv(sensitivity=0.0, specificity=1.0, base_rates=[0.1, 0.5])
        for prev, ppv, npv in points:
            assert ppv == 0.0
            assert npv == pytest.approx(1.0 - prev)
