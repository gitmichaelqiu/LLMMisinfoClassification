"""Tests for src/schemas.py — unified verification data structures."""

from __future__ import annotations

from src.schemas import (
    ClassificationMetrics,
    ConfusionMatrix,
    Verdict,
    VerificationItem,
    VerificationResult,
)


class TestVerdict:
    def test_values(self):
        assert Verdict.REAL == 0
        assert Verdict.FAKE == 1
        assert Verdict.ESCALATE == 2

    def test_membership(self):
        assert Verdict.REAL in Verdict
        # IntEnum: int values match by value (REAL.value == 0)
        assert 0 in Verdict


class TestVerificationItem:
    def test_create_with_ground_truth(self):
        item = VerificationItem.create(
            claim_text="Test claim",
            ground_truth=Verdict.FAKE,
            metadata={"domain": "test"},
        )
        assert item.claim_text == "Test claim"
        assert item.ground_truth == Verdict.FAKE
        assert item.metadata["domain"] == "test"
        assert len(item.id) == 8  # uuid4 truncated

    def test_create_without_ground_truth(self):
        item = VerificationItem.create(claim_text="Unlabeled claim")
        assert item.ground_truth is None

    def test_create_with_context(self):
        item = VerificationItem.create(
            claim_text="Claim with context",
            context="Supporting context paragraph.",
        )
        assert item.context == "Supporting context paragraph."


class TestVerificationResult:
    def test_create_defaults(self):
        result = VerificationResult(
            item_id="abc123",
            verdict=Verdict.FAKE,
            confidence=0.85,
        )
        assert result.item_id == "abc123"
        assert result.verdict == Verdict.FAKE
        assert result.confidence == 0.85
        assert result.latency_s == 0.0
        assert result.evidence == []

    def test_with_evidence(self):
        result = VerificationResult(
            item_id="abc123",
            verdict=Verdict.REAL,
            confidence=0.92,
            latency_s=1.5,
            evidence=["Flags: none", "Looks authentic"],
        )
        assert len(result.evidence) == 2


class TestConfusionMatrix:
    def test_defaults(self):
        cm = ConfusionMatrix()
        assert cm.tp == 0
        assert cm.fp == 0
        assert cm.tn == 0
        assert cm.fn == 0

    def test_counts(self):
        cm = ConfusionMatrix(tp=10, fp=2, tn=80, fn=8)
        assert cm.tp == 10


class TestClassificationMetrics:
    def test_from_confusion(self):
        metrics = ClassificationMetrics(
            precision=0.8,
            recall=0.9,
            f1=0.85,
            fpr=0.1,
            fnr=0.1,
            accuracy=0.9,
            n_total=100,
            confusion=ConfusionMatrix(tp=10, fp=2, tn=80, fn=8),
        )
        assert metrics.precision == 0.8
        assert metrics.n_total == 100
