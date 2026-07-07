"""Mixture of Agents (MoA) debate architecture for MFT verification arbitrage.

Three agents with distinct system prompts:
1. The Believer — finds RAG evidence supporting the news as REAL
2. The Skeptic — finds RAG evidence suggesting a hoax (FAKE)
3. The Risk Officer — synthesizes both and makes the final FAKE/REAL/ESCALATE decision

Concurrency design (Phase 9 audit):
- Agent 1 and Agent 2 run CONCURRENTLY via ThreadPoolExecutor
- Agent 3 runs SEQUENTIALLY after both complete
- Total T1 latency = max(T_believer, T_skeptic) + T_risk_officer
  NOT T_believer + T_skeptic + T_risk_officer (sequential)

Reference: Phase 9 of CLAUDE.md
"""

import time
from concurrent.futures import ThreadPoolExecutor

BELIEVER_SYSTEM_PROMPT = """You are Agent 1 (The Believer) in a financial news verification debate.

Your role is to aggressively find evidence SUPPORTING the authenticity of the headline. You argue that the news is REAL and should NOT be intervened on.

Given the news corpus context and social media data, build the strongest possible case that this headline is AUTHENTIC:

1. Source credibility: Could the headline originate from a legitimate source?
2. Factual alignment: Does the headline align with the entity's known operations, strategy, or market position?
3. Internal consistency: Are the claims internally logical and self-consistent?
4. Social corroboration: Could the early social reaction be genuine organic interest?
5. Plausibility: Is the magnitude of the claim within the realm of possibility for this entity?

Be thorough — list specific evidence points. When you find genuine supporting evidence, report it clearly.

Output your verdict in exactly this format:
Verdict: REAL or UNCERTAIN
Evidence: <bulleted list of supporting evidence>
Confidence: <0-100>"""

BELIEVER_USER_PROMPT = """Headline: {headline}

Entity: {entity}

News Context:
{news_context}

Social Media Context:
{social_context}

Review the evidence above. Build the strongest case that this headline is REAL. List specific supporting evidence."""

SKEPTIC_SYSTEM_PROMPT = """You are Agent 2 (The Skeptic) in a financial news verification debate.

Your role is to aggressively find evidence that the headline is a HOAX, misinformation, or fabricated news. You argue that the trade should be REVERSED.

Given the news corpus context and social media data, build the strongest possible case that this headline is FAKE:

1. Factual contradictions: Does the headline contradict known facts from the verified corpus?
2. Metric implausibility: Are the numbers, valuations, or claims wildly unrealistic?
3. Temporal/logical inconsistencies: Do timelines or causal relationships not make sense?
4. Entity mismatch: Does the headline attribute actions or statements that don't fit the entity?
5. Social anomalies: Is there evidence of coordinated or inorganic social media behavior?
6. Red flags: What specific details suggest this is manufactured disinformation?

Be thorough — list specific contradictions and red flags. When you find genuine problems, report them clearly.

Output your verdict in exactly this format:
Verdict: FAKE or UNCERTAIN
Evidence: <bulleted list of contradictory evidence>
Confidence: <0-100>"""

SKEPTIC_USER_PROMPT = """Headline: {headline}

Entity: {entity}

News Context:
{news_context}

Social Media Context:
{social_context}

Review the evidence above. Build the strongest case that this headline is FAKE. List specific contradictions and red flags."""

RISK_OFFICER_SYSTEM_PROMPT = """You are Agent 3 (The Risk Officer) in a financial news verification debate.

You have received analyses from two specialist agents:
- Agent 1 (The Believer) argues the news is REAL and the trade should be held.
- Agent 2 (The Skeptic) argues the news is FAKE and the trade should be reversed.

Your job is to make the FINAL DECISION by weighing both arguments against each other and considering the market context.

Consider:
1. Evidence quality: Which agent provided more specific, factual evidence?
2. Evidence quantity: Which agent identified more concrete contradictions or corroborations?
3. Asymmetric costs: Intervening on a REAL event costs missed profit (FP cost). NOT intervening on a FAKE event causes a crash loss (TP saving). These are ASYMMETRIC — reversing a FAKE saves ~$10k but reversing a REAL costs ~$6.8k in missed profit.
4. Market context: In volatile conditions or thin liquidity, the cost of wrong intervention may be higher.

Step 1 — Evaluate each agent's evidence quality and specificity.
Step 2 — Weigh the asymmetric economics (FAKE losses > REAL missed profits).
Step 3 — Make your final decision.

Output in exactly this format:
Verdict: FAKE or REAL or ESCALATE
Confidence: <0-100>
Flags: [contradiction|entity_mismatch|temporal_inconsistency|metric_implausibility|social_velocity_anomaly|debate_disagreement|none]
Reasoning: <one-sentence rationale>"""

RISK_OFFICER_USER_PROMPT = """Headline to verify: {headline}

Entity: {entity}

=== Agent 1 (Believer) — Evidence for REAL ===
{agent1_output}

=== Agent 2 (Skeptic) — Evidence for FAKE ===
{agent2_output}

=== Market Liquidity Context ===
{liquidity_context}

Review both agent analyses. Weigh the evidence quality. Consider the asymmetric economics. Make your final decision: FAKE (reverse trade), REAL (hold), or ESCALATE (human review)."""

# Simple liquidity profile descriptions
LIQUIDITY_DESCRIPTIONS = {
    "high_cap": "High-cap equity with deep books (~20,000 bid depth), tight spreads (~0.3 bps). Low reflexivity penalty for intervention.",
    "mid_cap": "Mid-cap equity with moderate liquidity (~5,000 bid depth), normal spreads (~0.5 bps). Standard reflexivity penalty.",
    "low_cap": "Low-cap / micro-cap equity with thin books (~500 bid depth), wide spreads (~2.0 bps). High reflexivity penalty for intervention.",
}


class MoADebate:
    """Mixture of Agents debate for financial news verification.

    Flow:
    1. Agent 1 (Believer) + Agent 2 (Skeptic) — CONCURRENT
    2. Agent 3 (Risk Officer) — SEQUENTIAL, after both complete

    This minimizes T1 latency: max(T_believer, T_skeptic) + T_risk_officer
    vs sequential: T_believer + T_skeptic + T_risk_officer.
    """

    def __init__(self, client, model="deepseek-v4-flash", thinking="enabled"):
        """
        Args:
            client: OpenAI-compatible LLM client (None for mock mode)
            model: Model name for API calls
            thinking: DeepSeek thinking mode
        """
        self.client = client
        self.model = model
        self.thinking = thinking
        self.last_latencies = {}

    def _call_llm(self, system_prompt, user_prompt, max_tokens=600):
        """Make a single LLM call and return (response_text, latency_ms)."""
        if self.client is None:
            return "Mock mode — no LLM available.", 50.0

        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": self.thinking}},
            )
            latency = (time.time() - t0) * 1000
            return response.choices[0].message.content.strip(), latency
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return f"Agent error: {e}", latency

    def _run_agent(self, prompt_type, headline, entity, news_context, social_context):
        """Run a single agent (believer or skeptic). Returns (output_text, latency_ms)."""
        if prompt_type == "believer":
            system = BELIEVER_SYSTEM_PROMPT
            user = BELIEVER_USER_PROMPT.format(
                headline=headline,
                entity=entity,
                news_context=news_context[:2000],
                social_context=social_context[:1500],
            )
        else:
            system = SKEPTIC_SYSTEM_PROMPT
            user = SKEPTIC_USER_PROMPT.format(
                headline=headline,
                entity=entity,
                news_context=news_context[:2000],
                social_context=social_context[:1500],
            )

        return self._call_llm(system, user)

    def evaluate(self, headline, entity, news_context, social_context,
                 liquidity_profile="mid_cap"):
        """Run full MoA evaluation with concurrent agents.

        Flow:
        1. Believer + Skeptic — CONCURRENT (ThreadPoolExecutor)
        2. Risk Officer — SEQUENTIAL (depends on both outputs)

        Args:
            headline: The headline text to verify
            entity: Entity name
            news_context: Formatted news context string
            social_context: Formatted social context string
            liquidity_profile: Key into LIQUIDITY_DESCRIPTIONS

        Returns:
            dict with keys:
                verdict (int): 0=REAL, 1=FAKE, 2=ESCALATE
                confidence (float): 0.0-1.0
                cot_flags (dict): Parsed verdict fields
                raw_output (str): Risk Officer's raw output
                believer_output (str): Agent 1 raw output
                skeptic_output (str): Agent 2 raw output
                latencies (dict): Per-agent latency breakdown
        """
        t_start = time.time()

        # Phase 1: Believer + Skeptic CONCURRENTLY
        with ThreadPoolExecutor(max_workers=2) as pool:
            believer_future = pool.submit(
                self._run_agent, "believer", headline, entity, news_context, social_context
            )
            skeptic_future = pool.submit(
                self._run_agent, "skeptic", headline, entity, news_context, social_context
            )
            believer_output, believer_latency = believer_future.result()
            skeptic_output, skeptic_latency = skeptic_future.result()

        concurrent_phase_ms = (time.time() - t_start) * 1000

        # Phase 2: Risk Officer SEQUENTIALLY (depends on Agent 1 + Agent 2 outputs)
        liquidity_desc = LIQUIDITY_DESCRIPTIONS.get(
            liquidity_profile,
            "Mid-cap equity with standard liquidity (~5,000 bid depth)."
        )
        risk_user = RISK_OFFICER_USER_PROMPT.format(
            headline=headline,
            entity=entity,
            agent1_output=believer_output,
            agent2_output=skeptic_output,
            liquidity_context=liquidity_desc,
        )
        risk_output, risk_latency = self._call_llm(RISK_OFFICER_SYSTEM_PROMPT, risk_user)

        total_ms = (time.time() - t_start) * 1000

        self.last_latencies = {
            "believer_ms": round(believer_latency, 2),
            "skeptic_ms": round(skeptic_latency, 2),
            "concurrent_phase_ms": round(concurrent_phase_ms, 2),
            "risk_officer_ms": round(risk_latency, 2),
            "total_moa_ms": round(total_ms, 2),
        }

        # Parse the Risk Officer's verdict
        parsed = self._parse_verdict(risk_output)

        return {
            "verdict": parsed.get("verdict", 0),
            "confidence": parsed.get("confidence", 0.5),
            "cot_flags": parsed,
            "raw_output": risk_output,
            "believer_output": believer_output,
            "skeptic_output": skeptic_output,
            "latencies": self.last_latencies,
        }

    @staticmethod
    def _parse_verdict(text):
        """Parse the Risk Officer's structured output."""
        from src.cot_parser import parse_cot_output
        return parse_cot_output(text)

    def shutdown(self):
        """No-op for API-based agents (threads are per-call)."""
        pass
