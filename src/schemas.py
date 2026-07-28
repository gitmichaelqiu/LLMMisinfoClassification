"""Unified verification schemas and data structures.

Defines the canonical types shared across all verifier architectures:
- VerificationItem: input claim with context and ground truth
- Verdict: classification outcome (REAL, FAKE, ESCALATE, EXAGGERATED)
- VerificationResult: structured output from a verifier
- VerifierConfig: configuration for a verifier run
"""

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
    EXAGGERATED = 3


@dataclass
class VerificationItem:
    """A single claim to be verified.

    Attributes:
        id: Unique identifier.
        claim_text: The claim or headline to verify.
        context: Optional supporting context (e.g., article body, related claims).
        metadata: Arbitrary metadata (domain, source, timestamp, etc.).
        ground_truth: Known ground-truth verdict (None if unlabeled).
    """

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
    """Output from a single verifier invocation.

    Attributes:
        item_id: Matches VerificationItem.id.
        verdict: The predicted verdict.
        confidence: Confidence score in [0, 1].
        latency_s: Wall-clock time in seconds.
        evidence: Optional list of evidence snippets or reasoning traces.
        metadata: Verifier-specific metadata (model, prompt version, etc.).
    """

    item_id: str
    verdict: Verdict
    confidence: float
    latency_s: float = 0.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierConfig:
    """Configuration for a verifier run.

    Attributes:
        model: Model identifier (e.g., "gpt-4o-mini", "deepseek-chat").
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        prompt_template: Name or key of the prompt template to use.
        n_voters: Number of voters (for voting verifier).
        retriever_type: Retriever type (for RAG verifier, e.g., "dense", "sparse").
        extra: Additional verifier-specific parameters.
    """

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
