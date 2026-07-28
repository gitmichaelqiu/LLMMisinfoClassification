"""Data structures shared across all verifier architectures."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


class Verdict(IntEnum):
    """Canonical verdict labels for information verification."""

    REAL = 0
    FAKE = 1
    ESCALATE = 2


@dataclass
class VerificationItem:
    """A single claim to be verified."""

    id: str
    claim_text: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ground_truth: Optional[Verdict] = None

    @classmethod
    def create(
        cls,
        claim_text: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Verdict] = None,
    ) -> "VerificationItem":
        return cls(
            id=str(uuid.uuid4())[:8],
            claim_text=claim_text,
            context=context,
            metadata=metadata or {},
            ground_truth=ground_truth,
        )


@dataclass
class VerificationResult:
    """Output from a single verifier invocation."""

    item_id: str
    verdict: Verdict
    confidence: float
    latency_s: float = 0.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    """Configuration for a verifier run."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_template: str = "default"
    n_voters: int = 5
    retriever_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfusionMatrix:
    """Raw confusion matrix counts."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


@dataclass
class ClassificationMetrics:
    """Standard classification metrics computed from a confusion matrix."""

    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    accuracy: float
    n_total: int
    confusion: ConfusionMatrix
