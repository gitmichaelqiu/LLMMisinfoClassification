"""Single-shot (direct) LLM verifier.

Calls the LLM once with a system prompt and parses the response into
a structured VerificationResult. This is the simplest verifier architecture
and serves as the baseline for all comparisons.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from src.llm_clients import LLMClient, create_client
from src.prompts import (
    SINGLE_SHOT_SYSTEM,
    SINGLE_SHOT_USER,
    format_user_prompt,
)
from src.schemas import (
    Verdict,
    VerificationItem,
    VerificationResult,
    VerifierConfig,
)


class SingleShotVerifier:
    """Single-shot verifier: one LLM call, parsed response.

    Attributes:
        client: LLM client instance.
        config: Verifier configuration.
        system_prompt: System prompt override (uses default if None).
    """

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        client: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
    ):
        self.config = config or VerifierConfig()
        self.client = client or create_client(mock=True)
        self.system_prompt = system_prompt or SINGLE_SHOT_SYSTEM

    def verify(self, item: VerificationItem) -> VerificationResult:
        """Verify a single claim.

        Args:
            item: The claim to verify.

        Returns:
            VerificationResult with parsed verdict, confidence, and evidence.
        """
        user_prompt = format_user_prompt(
            SINGLE_SHOT_USER,
            claim_text=item.claim_text,
            context=item.context or "",
        )

        start = time.time()
        raw_output = self.client.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            config=self.config,
        )
        elapsed = time.time() - start

        return self._parse_response(raw_output, item.id, elapsed)

    def verify_batch(
        self,
        items: list[VerificationItem],
    ) -> list[VerificationResult]:
        """Verify multiple claims sequentially.

        Args:
            items: List of claims to verify.

        Returns:
            List of VerificationResult objects.
        """
        return [self.verify(item) for item in items]

    # ── Response Parsing ──────────────────────────────────────────

    _VERDICT_PATTERN = re.compile(
        r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)",
        re.IGNORECASE,
    )
    _CONFIDENCE_PATTERN = re.compile(
        r"Confidence:\s*(\d+)",
        re.IGNORECASE,
    )
    _FLAGS_PATTERN = re.compile(
        r"Flags:\s*\[([^\]]*)\]",
        re.IGNORECASE,
    )
    _REASONING_PATTERN = re.compile(
        r"Reasoning:\s*(.+)",
        re.IGNORECASE,
    )

    _VERDICT_MAP = {
        "REAL": Verdict.REAL,
        "FAKE": Verdict.FAKE,
        "ESCALATE": Verdict.ESCALATE,
        "EXAGGERATED": Verdict.EXAGGERATED,
    }

    def _parse_response(
        self,
        raw: str,
        item_id: str,
        latency_s: float,
    ) -> VerificationResult:
        """Parse raw LLM output into a structured result.

        Args:
            raw: Raw LLM response text.
            item_id: Corresponding VerificationItem id.
            latency_s: Wall-clock time for the LLM call.

        Returns:
            Parsed VerificationResult.
        """
        verdict_match = self._VERDICT_PATTERN.search(raw)
        verdict = Verdict.REAL  # default
        if verdict_match:
            key = verdict_match.group(1).upper()
            verdict = self._VERDICT_MAP.get(key, Verdict.REAL)

        confidence_match = self._CONFIDENCE_PATTERN.search(raw)
        confidence = 0.5
        if confidence_match:
            confidence = int(confidence_match.group(1)) / 100.0

        flags_match = self._FLAGS_PATTERN.search(raw)
        flags = flags_match.group(1) if flags_match else ""

        reasoning_match = self._REASONING_PATTERN.search(raw)
        reasoning = reasoning_match.group(1) if reasoning_match else ""

        evidence = []
        if flags:
            evidence.append(f"Flags: {flags}")
        if reasoning:
            evidence.append(reasoning)

        return VerificationResult(
            item_id=item_id,
            verdict=verdict,
            confidence=min(max(confidence, 0.0), 1.0),
            latency_s=latency_s,
            evidence=evidence,
            metadata={
                "model": self.config.model,
                "prompt_template": self.config.prompt_template,
                "raw_output_first_100": raw[:100],
            },
        )
