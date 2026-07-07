"""LLM client abstractions for model-agnostic verification.

Provides:
- LLMClient ABC with generate / generate_batch interface
- OpenAIClient wrapper for OpenAI-compatible APIs
- DeepSeekClient wrapper
- MockClient for testing without API keys
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional

from src.schemas import VerifierConfig


class LLMClient(ABC):
    """Abstract base for LLM API clients."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: VerifierConfig,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User message / query.
            config: Verifier configuration (model, temperature, etc.).

        Returns:
            Raw text response from the LLM.
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[tuple[str, str]],
        config: VerifierConfig,
    ) -> List[str]:
        """Generate responses for multiple prompt pairs.

        Args:
            prompts: List of (system_prompt, user_prompt) tuples.
            config: Verifier configuration.

        Returns:
            List of raw text responses.
        """
        ...


class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible APIs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None  # lazy import to avoid hard dependency

    def _lazy_init(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, config: VerifierConfig) -> str:
        self._lazy_init()
        response = self._client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def generate_batch(self, prompts: List[tuple[str, str]], config: VerifierConfig) -> List[str]:
        self._lazy_init()
        results = []
        for system_prompt, user_prompt in prompts:
            resp = self._client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            results.append(resp.choices[0].message.content or "")
        return results


class DeepSeekClient(LLMClient):
    """Client for DeepSeek API (OpenAI-compatible)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._client = None

    def _lazy_init(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )

    def generate(self, system_prompt: str, user_prompt: str, config: VerifierConfig) -> str:
        self._lazy_init()
        response = self._client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def generate_batch(self, prompts: List[tuple[str, str]], config: VerifierConfig) -> List[str]:
        self._lazy_init()
        results = []
        for system_prompt, user_prompt in prompts:
            resp = self._client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            results.append(resp.choices[0].message.content or "")
        return results


class MockClient(LLMClient):
    """Deterministic mock client for testing without API keys.

    Returns a fixed response format for reproducibility.
    """

    def __init__(self, fixed_verdict: str = "FAKE", fixed_confidence: int = 85):
        self.fixed_verdict = fixed_verdict
        self.fixed_confidence = fixed_confidence

    def generate(self, system_prompt: str, user_prompt: str, config: VerifierConfig) -> str:
        return (
            f"Verdict: {self.fixed_verdict}\n"
            f"Confidence: {self.fixed_confidence}\n"
            f"Flags: none\n"
            f"Reasoning: Mock response for testing."
        )

    def generate_batch(self, prompts: List[tuple[str, str]], config: VerifierConfig) -> List[str]:
        return [self.generate(sp, up, config) for sp, up in prompts]


def create_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    mock: bool = False,
) -> LLMClient:
    """Factory function for LLM clients.

    Args:
        provider: "openai", "deepseek", or other registered providers.
        api_key: API key (defaults to environment variable).
        mock: If True, return MockClient regardless of provider.

    Returns:
        An LLMClient instance.
    """
    if mock:
        return MockClient()

    provider_map = {
        "openai": OpenAIClient,
        "deepseek": DeepSeekClient,
    }
    client_cls = provider_map.get(provider)
    if client_cls is None:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(provider_map.keys())}")
    return client_cls(api_key=api_key)
