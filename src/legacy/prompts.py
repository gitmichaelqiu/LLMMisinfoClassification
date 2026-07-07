"""RAG prompt templates for MFT verification arbitrage.

Supports dual-source evaluation: historical contradictions from
Official_Corpus (static news) and social media consensus from
Social_Velocity_Stream (real-time posts).

Verdicts: FAKE (intervene), REAL (no action), ESCALATE (grey swan).
"""

from src.domain_adapter import DomainAdapter

RAG_SYSTEM_PROMPT = """You are a Financial News Authenticity Verifier. Your task is to determine whether a given financial news headline is authentic or contains logical contradictions by comparing it against verified news articles about the same entity.

Given recent verified news articles about {entity}, assess whether the headline:
1. Contradicts known facts from the verified articles
2. Makes implausible or impossible claims about financial metrics
3. Contains temporal inconsistencies (events happening out of order)
4. Describes scenarios that conflict with the entity's known operations

Output only: FAKE if the headline is suspicious or contradictory, REAL if it appears authentic. Follow with one-sentence reasoning."""

RAG_USER_PROMPT = """Recent verified articles about {entity}:
{context}

Headline to verify:
{headline}

Is this headline FAKE or REAL?"""

NON_RAG_SYSTEM_PROMPT = """You are a Financial News Logic Validator. Detect macroeconomic impossibilities or fraudulent narratives. Output 'FAKE' for anomalies/misinformation and 'REAL' for authentic news."""

# ── Dual-Source CoT Prompt (Phase 3.2) ─────────────────────────

COT_RAG_SYSTEM_PROMPT = """You are a Financial News Authenticity Verifier with access to two context sources:

1. A static news corpus with verified facts about companies.
2. A real-time social media stream showing early public reaction.

Analyze the headline and determine whether the news is FAKE (a hoax/misinformation), REAL (authentic news), or if you are UNCERTAIN and need to ESCALATE to a human reviewer.

Step 1 — Known facts: What do the verified news articles tell us about {entity}? Summarize the key facts.

Step 2 — Claim analysis: What does the headline claim about {entity}? Identify specific metrics, numbers, and events.

Step 3 — Contradiction detection: Identify contradictions between the headline and verified articles.
- Metric plausibility: Are the numbers (revenue, valuation, headcount) consistent with known facts?
- Temporal consistency: Do timelines and schedules make sense?
- Operational feasibility: Could the entity realistically achieve what's claimed?
- Entity-specific facts: Does the claim conflict with entity-specific knowledge?

Step 4 — Social consensus evaluation: Analyze the social media posts and their velocity.
- Are early posts showing panic (typical of fake news impact)?
- Is there rising skepticism or debunking sentiment?
- Or is the social stream amplifying and confirming the news?
- What is the dominant consensus direction?

Step 5 — Verdict: Choose EXACTLY one of three verdicts:

- FAKE: The headline clearly contradicts verified facts AND/OR social consensus shows strong debunking velocity. The trade should be reversed.
- REAL: The headline is consistent with verified facts AND social consensus supports it. No intervention needed.
- ESCALATE: The headline is highly anomalous or the event is a potential Grey Swan — the facts don't add up but social consensus has NOT yet confirmed debunking. Escalate to human reviewer.

Output your verdict in EXACTLY this format:

Verdict: FAKE or REAL or ESCALATE
Confidence: <0-100>
Flags: [contradiction|entity_mismatch|temporal_inconsistency|metric_implausibility|social_velocity_anomaly|none]
Reasoning: <one-sentence rationale>"""

COT_RAG_USER_PROMPT = """Recent verified articles about {entity}:
{context}

Social Media Posts (before T1 verification window):
{social_context}

Headline to verify:
{headline}

Follow the five-step reasoning process above. End with your verdict in the exact format specified. Choose FAKE, REAL, or ESCALATE."""


# ── Domain Adapter ──────────────────────────────────────────────

def get_domain_prompts(domain="finance"):
    """Return domain-specific prompt templates for cross-domain use.

    Args:
        domain: "finance" or "health"

    Returns:
        dict with keys: rag_system, rag_user, non_rag_system,
                        cot_rag_system, cot_rag_user
    """
    adapter = DomainAdapter(domain)
    role = adapter.verifier_role
    ctx = adapter.domain_context
    question = adapter.verdict_question

    prompts = {
        "rag_system": (
            f"You are a {role}. Your task is to determine whether a given "
            f"headline is authentic or contains logical contradictions by "
            f"comparing it against verified articles about the same entity.\n\n"
            f"Given recent verified articles about {{entity}}, assess whether the headline:\n"
            f"1. Contradicts known facts from the verified articles\n"
            f"2. Makes implausible or impossible claims\n"
            f"3. Contains temporal inconsistencies\n"
            f"4. Describes scenarios that conflict with known operations\n\n"
            f"Output only: FAKE if the headline is suspicious or contradictory, "
            f"REAL if it appears authentic. Follow with one-sentence reasoning."
        ),
        "rag_user": (
            f"Recent verified articles about {{entity}}:\n"
            f"{{context}}\n\n"
            f"Headline to verify:\n"
            f"{{headline}}\n\n"
            f"{question}"
        ),
        "non_rag_system": (
            f"You are a {role}. Analyze {ctx}. "
            f"Output 'FAKE' for anomalies/misinformation and 'REAL' for authentic news."
        ),
        "cot_rag_system": (
            f"You are a {role} with access to two context sources:\n\n"
            f"1. A verified corpus of facts about companies/organizations.\n"
            f"2. A real-time social media stream showing early public reaction.\n\n"
            f"Analyze the headline and determine whether the news is FAKE "
            f"(a hoax/misinformation), REAL (authentic news), or if you need "
            f"to ESCALATE to a human reviewer (Grey Swan).\n\n"
            f"Step 1 — Known facts: What do verified articles tell us about "
            f"{{entity}}? Summarize key facts.\n\n"
            f"Step 2 — Claim analysis: What does the headline claim about "
            f"{{entity}}? Identify specific details, numbers, and events.\n\n"
            f"Step 3 — Contradiction detection: Identify contradictions between "
            f"the headline and verified articles.\n\n"
            f"Step 4 — Social consensus evaluation: Analyze social media posts. "
            f"Are early posts showing panic, skepticism/debunking, or "
            f"amplification/confirmation? What is the dominant direction?\n\n"
            f"Step 5 — Verdict: Choose EXACTLY one of:\n"
            f"- FAKE: Clear evidence of hoax — reverse the trade.\n"
            f"- REAL: Consistent with facts and consensus — no intervention.\n"
            f"- ESCALATE: Grey Swan — anomalous but unconfirmed; escalate.\n\n"
            f"Output in EXACTLY this format:\n"
            f"Verdict: FAKE or REAL or ESCALATE\n"
            f"Confidence: <0-100>\n"
            f"Flags: [contradiction|entity_mismatch|temporal_inconsistency|"
            f"metric_implausibility|social_velocity_anomaly|none]\n"
            f"Reasoning: <one-sentence rationale>"
        ),
        "cot_rag_user": (
            "Recent verified articles about {entity}:\n"
            "{context}\n\n"
            "Social Media Posts (before T1 verification window):\n"
            "{social_context}\n\n"
            "Headline to verify:\n"
            "{headline}\n\n"
            "Follow the five-step reasoning process. End with your verdict "
            "in the exact format specified. Choose FAKE, REAL, or ESCALATE."
        ),
    }
    return prompts
