"""Tests for src/llm_clients.py — LLM client abstractions."""

from __future__ import annotations

import pytest

from src.llm_clients import (
    LLMClient,
    MockClient,
    OpenAIClient,
    create_client,
)
from src.schemas import VerifierConfig


class TestMockClient:
    def test_generate_returns_expected_format(self):
        client = MockClient(fixed_verdict="REAL", fixed_confidence=92)
        output = client.generate("sys", "user", VerifierConfig())
        assert "Verdict: REAL" in output
        assert "Confidence: 92" in output
        assert "Reasoning:" in output

    def test_generate_batch(self):
        client = MockClient()
        prompts = [("sys1", "user1"), ("sys2", "user2")]
        outputs = client.generate_batch(prompts, VerifierConfig())
        assert len(outputs) == 2
        for o in outputs:
            assert o.startswith("Verdict:")

    def test_default_verdict(self):
        client = MockClient()
        output = client.generate("sys", "user", VerifierConfig())
        assert "Verdict: FAKE" in output

    def test_is_llm_client(self):
        assert isinstance(MockClient(), LLMClient)


class TestCreateClient:
    def test_mock_factory(self):
        client = create_client(mock=True)
        assert isinstance(client, MockClient)

    def test_openai_factory_creates_instance(self):
        client = create_client(provider="openai", mock=False)
        assert isinstance(client, OpenAIClient)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client(provider="nonexistent", mock=False)
