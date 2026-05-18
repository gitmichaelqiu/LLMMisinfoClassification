"""RAG prompt templates for financial news authenticity verification.

Domain-aware via get_domain_prompts(domain) for cross-domain generalization.
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

COT_RAG_SYSTEM_PROMPT = """You are a Financial News Authenticity Verifier. Analyze the headline against verified articles to determine authenticity.

Step 1 — Known facts: What do the verified articles tell us about {entity}? Summarize the key facts.

Step 2 — Claim analysis: What does the headline claim about {entity}? Identify specific metrics, numbers, and events.

Step 3 — Contradiction detection: Identify contradictions between the headline and verified articles. Check:
- Metric plausibility: Are the numbers (revenue, valuation, headcount) consistent with known facts?
- Temporal consistency: Do timelines and schedules make sense?
- Operational feasibility: Could the entity realistically achieve what's claimed?
- Entity-specific facts: Does the claim conflict with entity-specific knowledge?

Step 4 — Verdict: Output your verdict in EXACTLY this format:

Verdict: FAKE or REAL
Confidence: <0-100>
Flags: [contradiction|entity_mismatch|temporal_inconsistency|metric_implausibility|source_unverifiable|none]"""

COT_RAG_USER_PROMPT = """Recent verified articles about {entity}:
{context}

Headline to verify:
{headline}

Follow the four-step reasoning process above. End with your verdict in the exact format specified."""


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
            f"You are a {role}. Analyze the headline against verified "
            f"articles to determine authenticity.\n\n"
            f"Step 1 — Known facts: What do the verified articles tell us "
            f"about {{entity}}? Summarize the key facts.\n\n"
            f"Step 2 — Claim analysis: What does the headline claim about "
            f"{{entity}}? Identify specific details, numbers, and events.\n\n"
            f"Step 3 — Contradiction detection: Identify contradictions between "
            f"the headline and verified articles.\n\n"
            f"Step 4 — Verdict: Output your verdict in EXACTLY this format:\n\n"
            f"Verdict: FAKE or REAL\n"
            f"Confidence: <0-100>\n"
            f"Flags: [contradiction|entity_mismatch|temporal_inconsistency|"
            f"metric_implausibility|source_unverifiable|none]"
        ),
        "cot_rag_user": (
            f"Recent verified articles about {{entity}}:\n"
            f"{{context}}\n\n"
            f"Headline to verify:\n"
            f"{{headline}}\n\n"
            f"Follow the four-step reasoning process above. End with your "
            f"verdict in the exact format specified."
        ),
    }
    return prompts
