"""CoT output parser for structured flag extraction from LLM reasoning."""

import re

FLAG_NAMES = [
    "contradiction",
    "entity_mismatch",
    "temporal_inconsistency",
    "metric_implausibility",
    "source_unverifiable",
]


def parse_cot_output(response_text: str) -> dict:
    """Extract structured fields from CoT LLM response.

    Handles truncated responses: if the model was cut off before reaching
    the structured output, infers verdict from reasoning body.

    Returns dict with:
        verdict (int): 0=REAL, 1=FAKE
        confidence (float): 0.0-1.0
        contradiction_flag (int): 0/1
        entity_mismatch (int): 0/1
        temporal_inconsistency (int): 0/1
        metric_implausibility (int): 0/1
        source_unverifiable (int): 0/1
    """
    result = {
        "verdict": 0,
        "confidence": 0.5,
        "contradiction_flag": 0,
        "entity_mismatch": 0,
        "temporal_inconsistency": 0,
        "metric_implausibility": 0,
        "source_unverifiable": 0,
    }

    if not response_text:
        return result

    text = response_text.strip()

    # --- Structured format parsing (preferred) ---
    verdict_match = re.search(r'Verdict:\s*(FAKE|REAL)', text, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = 1 if verdict_match.group(1).upper() == "FAKE" else 0

    conf_match = re.search(r'Confidence:\s*(\d+)', text, re.IGNORECASE)
    if conf_match:
        conf = int(conf_match.group(1))
        result["confidence"] = max(0.0, min(1.0, conf / 100.0))

    flags_match = re.search(r'Flags:\s*\[([^\]]*)\]', text, re.IGNORECASE)
    if flags_match:
        raw_flags = flags_match.group(1).lower()
        if "none" not in raw_flags:
            for flag_name in FLAG_NAMES:
                if flag_name in raw_flags:
                    result[flag_name] = 1
        return result  # Structured format found and parsed

    # --- Fallback: no structured format (truncated or loose output) ---
    # Search entire text for FAKE/REAL keywords
    text_upper = text.upper()
    has_fake = "FAKE" in text_upper
    has_real = "REAL" in text_upper

    # Keyword-based inference from reasoning body
    contradiction_keywords = [
        "contradict", "implausible", "impossible", "unrealistic",
        "far beyond", "wildly above", "not plausible", "seems fabricated",
        "is fabricated", "appears fake", "is fake", "anomaly",
        "cannot be true", "not credible", "misleading", "fictitious",
    ]
    real_keywords = [
        "appears authentic", "is authentic", "appears real", "is real",
        "consistent with", "plausible", "no contradiction",
    ]

    body = text.lower()
    fake_score = sum(1 for kw in contradiction_keywords if kw in body)
    real_score = sum(1 for kw in real_keywords if kw in body)

    if has_fake and not has_real:
        result["verdict"] = 1
    elif has_real and not has_fake:
        result["verdict"] = 0
    else:
        # Both present or neither — use keyword scoring
        result["verdict"] = 1 if fake_score > real_score else 0

    # Infer confidence from contradiction density
    n_contradictions = body.count("contradict")
    if n_contradictions >= 2:
        result["confidence"] = min(1.0, result["confidence"] + 0.2 * n_contradictions)

    # Auto-detect flags from reasoning body
    if re.search(r'contradict', body):
        result["contradiction_flag"] = 1
    if re.search(r'entity|company|organization', body) and re.search(r'mismatch|wrong|incorrect|not.*(entity|company)', body):
        result["entity_mismatch"] = 1
    if re.search(r'temporal|timeline|schedule|month|quarter.*ahead|ahead.*schedule', body):
        result["temporal_inconsistency"] = 1
    if re.search(r'metric|implausib|revenue.*(far|wildly|impossible|beyond)|valuation.*(too|inflated)', body):
        result["metric_implausibility"] = 1
    if re.search(r'source|unverifiable|cannot be verified|no source', body):
        result["source_unverifiable"] = 1

    return result
