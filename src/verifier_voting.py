"""Voting ensemble verifier.

Calls the LLM N times (with varied prompts or temperatures) and aggregates
verdicts by majority vote. Provides precision-recall trade-off analysis
against the single-shot baseline.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.llm_clients import LLMClient, create_client
from src.prompts import SINGLE_SHOT_USER, VOTING_VARIATIONS, format_user_prompt
from src.schemas import Verdict, VerificationItem, VerificationResult, VerifierConfig
from src.verifier_single_shot import SingleShotVerifier


class VotingVerifier:
    """Voting ensemble verifier with N parallel LLM calls.

    Attributes:
        n_voters: Number of parallel voters.
        aggregation: Strategy — "majority", "unanimous", or "confidence_weighted".
        client: LLM client instance.
        config: Verifier configuration.
    """

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        client: Optional[LLMClient] = None,
        n_voters: int = 5,
        aggregation: str = "majority",
    ):
        self.config = config or VerifierConfig(n_voters=n_voters)
        self.client = client or create_client(mock=True)
        self.n_voters = n_voters
        self.aggregation = aggregation
        self._base_verifier = SingleShotVerifier(config=self.config, client=self.client)

    def verify(self, item: VerificationItem) -> VerificationResult:
        """Verify a claim using N parallel voters, then aggregate.

        Args:
            item: The claim to verify.

        Returns:
            Aggregated VerificationResult.
        """
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.n_voters) as executor:
            futures = []
            for i in range(self.n_voters):
                variation = VOTING_VARIATIONS[i % len(VOTING_VARIATIONS)]
                user_prompt = format_user_prompt(
                    SINGLE_SHOT_USER,
                    claim_text=item.claim_text,
                    context=item.context or "",
                )
                futures.append(
                    executor.submit(
                        self.client.generate,
                        variation,
                        user_prompt,
                        self.config,
                    )
                )

            raw_outputs = [f.result() for f in as_completed(futures)]

        elapsed = time.time() - start

        # Parse each response
        results = []
        for raw in raw_outputs:
            parsed = self._base_verifier._parse_response(raw, item.id, elapsed)
            results.append(parsed)

        # Aggregate
        return self._aggregate(results, item.id, elapsed)

    def verify_batch(self, items: list[VerificationItem]) -> list[VerificationResult]:
        return [self.verify(item) for item in items]

    def _aggregate(
        self,
        results: list[VerificationResult],
        item_id: str,
        latency_s: float,
    ) -> VerificationResult:
        """Aggregate multiple voter results into a single verdict.

        Args:
            results: List of individual voter results.
            item_id: The item being verified.
            latency_s: Total wall-clock time.

        Returns:
            Aggregated VerificationResult.
        """
        verdicts = [r.verdict for r in results]
        confidences = [r.confidence for r in results]

        if self.aggregation == "unanimous":
            if len(set(verdicts)) == 1:
                final_verdict = verdicts[0]
            else:
                final_verdict = Verdict.ESCALATE
        elif self.aggregation == "confidence_weighted":
            score = sum(
                c if v == Verdict.FAKE else (1 - c)
                for v, c in zip(verdicts, confidences)
            )
            final_verdict = Verdict.FAKE if score / len(verdicts) > 0.5 else Verdict.REAL
        else:  # majority
            counter = Counter(verdicts)
            most_common = counter.most_common(1)
            final_verdict = most_common[0][0] if most_common else Verdict.REAL

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        agreement_rate = max(counter.values()) / len(verdicts) if verdicts else 0.0

        return VerificationResult(
            item_id=item_id,
            verdict=final_verdict,
            confidence=avg_confidence,
            latency_s=latency_s,
            evidence=[
                f"Voters: {self.n_voters}",
                f"Aggregation: {self.aggregation}",
                f"Agreement rate: {agreement_rate:.2f}",
                f"Verdict distribution: {dict(counter)}",
            ],
            metadata={
                "model": self.config.model,
                "n_voters": self.n_voters,
                "aggregation": self.aggregation,
                "agreement_rate": agreement_rate,
            },
        )
