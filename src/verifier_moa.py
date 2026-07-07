"""Mixture-of-Agents (MoA) debate verifier.

Three agents with distinct roles debate the authenticity of a claim:
1. Believer — argues the claim is REAL (pro-authenticity)
2. Skeptic — argues the claim is FAKE (pro-hoax)
3. Risk Officer — synthesizes both arguments and delivers final verdict

Believer and Skeptic run CONCURRENTLY; Risk Officer runs after both complete.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.llm_clients import LLMClient, create_client
from src.schemas import Verdict, VerificationItem, VerificationResult, VerifierConfig
from src.verifier_single_shot import SingleShotVerifier

# ── Role Prompts ──────────────────────────────────────────────

BELIEVER_SYSTEM_PROMPT = """You are Agent 1 (The Believer) in an information verification debate.

Your role is to find evidence SUPPORTING the authenticity of the claim. You argue the claim is REAL and should NOT be flagged.

Analyze the claim and build the strongest possible case for authenticity:
1. Source credibility: Could this originate from a legitimate source?
2. Factual alignment: Does the claim match known facts?
3. Internal consistency: Are the claims internally logical?
4. Plausibility: Is the magnitude within the realm of possibility?

Be thorough — list specific supporting evidence.

Output:
Verdict: REAL or UNCERTAIN
Evidence: <bulleted list of supporting points>
Confidence: <0-100>"""

SKEPTIC_SYSTEM_PROMPT = """You are Agent 2 (The Skeptic) in an information verification debate.

Your role is to find evidence that the claim is FALSE or misleading. You argue the claim is FAKE and should be flagged.

Analyze the claim and build the strongest possible case against authenticity:
1. Contradictions: Does the claim contradict known facts?
2. Implausibility: Are specific numbers or claims unrealistic?
3. Inconsistencies: Are there temporal or logical inconsistencies?
4. Red flags: Are there hallmarks of misinformation?

Be thorough — list specific problematic evidence.

Output:
Verdict: FAKE or UNCERTAIN
Evidence: <bulleted list of problematic points>
Confidence: <0-100>"""

RISK_OFFICER_SYSTEM_PROMPT = """You are Agent 3 (The Risk Officer) in an information verification debate.

You have received analyses from two agents:
- The Believer argues the claim is REAL.
- The Skeptic argues the claim is FAKE.

Your job is to evaluate both arguments, weigh the evidence, and deliver a final verdict.

Consider:
1. Which side provided more specific, verifiable evidence?
2. Which side's argument has more internal consistency?
3. Is there enough evidence for a clear decision, or is this ambiguous?

Output your final verdict in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale weighing both sides>"""


class MoAVerifier:
    """Mixture-of-Agents debate verifier.

    Attributes:
        client: LLM client instance.
        config: Verifier configuration.
        run_degeneracy_tests: If True, run base-rate checks after verification.
    """

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        client: Optional[LLMClient] = None,
    ):
        self.config = config or VerifierConfig()
        self.client = client or create_client(mock=True)
        self._parser = SingleShotVerifier(config=self.config, client=self.client)

    def verify(self, item: VerificationItem) -> VerificationResult:
        """Verify a claim via MoA debate.

        Believer and Skeptic run concurrently. Risk Officer runs after both complete.

        Args:
            item: The claim to verify.

        Returns:
            VerificationResult with final verdict from Risk Officer.
        """
        start = time.time()
        user_prompt = f"Claim to verify:\n{item.claim_text}\n\nContext:\n{item.context or 'None provided.'}"

        with ThreadPoolExecutor(max_workers=2) as executor:
            believer_future = executor.submit(
                self.client.generate,
                BELIEVER_SYSTEM_PROMPT,
                user_prompt,
                self.config,
            )
            skeptic_future = executor.submit(
                self.client.generate,
                SKEPTIC_SYSTEM_PROMPT,
                user_prompt,
                self.config,
            )
            believer_output = believer_future.result()
            skeptic_output = skeptic_future.result()

        # Risk Officer synthesizes
        synthesis_prompt = (
            f"Claim to verify:\n{item.claim_text}\n\n"
            f"=== Believer's Analysis (pro-REAL) ===\n{believer_output}\n\n"
            f"=== Skeptic's Analysis (pro-FAKE) ===\n{skeptic_output}\n\n"
            f"Based on both analyses above, deliver your final verdict."
        )

        risk_officer_output = self.client.generate(
            RISK_OFFICER_SYSTEM_PROMPT,
            synthesis_prompt,
            self.config,
        )

        elapsed = time.time() - start
        result = self._parser._parse_response(risk_officer_output, item.id, elapsed)

        # Attach debate transcripts to metadata
        result.metadata["believer_output"] = believer_output[:200]
        result.metadata["skeptic_output"] = skeptic_output[:200]
        result.metadata["verifier_type"] = "moa"
        result.evidence = [
            f"Believer confidence: {self._extract_confidence(believer_output)}",
            f"Skeptic confidence: {self._extract_confidence(skeptic_output)}",
        ]

        return result

    def verify_batch(self, items: list[VerificationItem]) -> list[VerificationResult]:
        return [self.verify(item) for item in items]

    @staticmethod
    def _extract_confidence(text: str) -> str:
        """Extract confidence value from an agent's output."""
        import re
        match = re.search(r"Confidence:\s*(\d+)", text, re.IGNORECASE)
        return match.group(1) if match else "unknown"
