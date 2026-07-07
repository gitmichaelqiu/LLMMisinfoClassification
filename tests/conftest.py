"""Shared pytest fixtures for the verification framework."""

from __future__ import annotations

import pytest

from src.schemas import Verdict, VerificationItem, VerifierConfig


@pytest.fixture
def sample_items() -> list[VerificationItem]:
    """Return a small set of labeled verification items for testing."""
    return [
        VerificationItem.create(
            claim_text="Federal Reserve raises interest rates by 25 basis points",
            ground_truth=Verdict.REAL,
            metadata={"domain": "finance", "source": "test"},
        ),
        VerificationItem.create(
            claim_text="BREAKING: Federal Reserve declares all USD bank deposits void",
            ground_truth=Verdict.FAKE,
            metadata={"domain": "finance", "source": "test"},
        ),
        VerificationItem.create(
            claim_text="WHO approves new vaccine for malaria treatment",
            ground_truth=Verdict.REAL,
            metadata={"domain": "health", "source": "test"},
        ),
        VerificationItem.create(
            claim_text="Scientists discover cure for all cancers in clinical trial",
            ground_truth=Verdict.FAKE,
            metadata={"domain": "health", "source": "test"},
        ),
    ]


@pytest.fixture
def default_config() -> VerifierConfig:
    """Return a default verifier config for testing."""
    return VerifierConfig(model="gpt-4o-mini", temperature=0.0, max_tokens=512)
