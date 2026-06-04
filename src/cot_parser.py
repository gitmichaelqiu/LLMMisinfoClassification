"""CoT output parser for MFT verification arbitrage.

Parses structured LLM outputs supporting three verdicts:
- FAKE (verdict=1): Hoax detected, reverse the trade
- REAL (verdict=0): Authentic news, no intervention
- ESCALATE (verdict=2): Grey Swan, flag for human reviewer
"""

import re
import json
import os

FLAG_NAMES = [
    "contradiction",
    "entity_mismatch",
    "temporal_inconsistency",
    "metric_implausibility",
    "social_velocity_anomaly",
]

# Verdict constants
VERDICT_REAL = 0
VERDICT_FAKE = 1
VERDICT_ESCALATE = 2

VERDICT_MAP = {
    "FAKE": VERDICT_FAKE,
    "REAL": VERDICT_REAL,
    "ESCALATE": VERDICT_ESCALATE,
}


def parse_cot_output(response_text: str) -> dict:
    """Extract structured fields from CoT LLM response.

    Handles truncated responses: if the model was cut off before reaching
    the structured output, infers verdict from reasoning body.

    Returns dict with:
        verdict (int): 0=REAL, 1=FAKE, 2=ESCALATE
        confidence (float): 0.0-1.0
        contradiction_flag (int): 0/1
        entity_mismatch (int): 0/1
        temporal_inconsistency (int): 0/1
        metric_implausibility (int): 0/1
        social_velocity_anomaly (int): 0/1
    """
    result = {
        "verdict": VERDICT_REAL,
        "confidence": 0.5,
        "contradiction_flag": 0,
        "entity_mismatch": 0,
        "temporal_inconsistency": 0,
        "metric_implausibility": 0,
        "social_velocity_anomaly": 0,
    }

    if not response_text:
        return result

    text = response_text.strip()

    # --- Structured format parsing (preferred) ---
    verdict_match = re.search(r'Verdict:\s*(FAKE|REAL|ESCALATE)', text, re.IGNORECASE)
    if verdict_match:
        raw_verdict = verdict_match.group(1).upper()
        result["verdict"] = VERDICT_MAP.get(raw_verdict, VERDICT_REAL)

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
                    result[flag_name.replace("social_velocity_anomaly", "social_velocity_anomaly")] = 1
                    # Map human-readable flag names to internal keys
                    if flag_name == "social_velocity_anomaly":
                        result["social_velocity_anomaly"] = 1
                if flag_name in raw_flags:
                    result[flag_name] = 1
        return result  # Structured format found and parsed

    # --- Fallback: no structured format (truncated or loose output) ---
    text_upper = text.upper()
    has_fake = "FAKE" in text_upper
    has_real = "REAL" in text_upper
    has_escalate = "ESCALATE" in text_upper

    # If ESCALATE appears and neither FAKE nor REAL is decisive
    if has_escalate and not has_fake:
        result["verdict"] = VERDICT_ESCALATE
        result["confidence"] = 0.5
        return result

    # Keyword-based inference from reasoning body
    contradiction_keywords = [
        "contradict", "implausible", "impossible", "unrealistic",
        "far beyond", "wildly above", "not plausible", "seems fabricated",
        "is fabricated", "appears fake", "is fake", "anomaly",
        "cannot be true", "not credible", "misleading", "fictitious",
        "hoax", "debunk", "fabricated", "spoofed",
    ]
    real_keywords = [
        "appears authentic", "is authentic", "appears real", "is real",
        "consistent with", "plausible", "no contradiction", "genuine",
    ]
    escalate_keywords = [
        "ambiguous", "uncertain", "grey swan", "gray swan",
        "unclear", "cannot determine", "not enough information",
        "escalate", "further review", "human review",
    ]

    body = text.lower()
    fake_score = sum(1 for kw in contradiction_keywords if kw in body)
    real_score = sum(1 for kw in real_keywords if kw in body)
    escalate_score = sum(1 for kw in escalate_keywords if kw in body)

    if has_fake and not has_real:
        result["verdict"] = VERDICT_FAKE
    elif has_real and not has_fake:
        result["verdict"] = VERDICT_REAL
    elif escalate_score > max(fake_score, real_score):
        result["verdict"] = VERDICT_ESCALATE
    else:
        result["verdict"] = VERDICT_FAKE if fake_score > real_score else VERDICT_REAL

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
    if re.search(r'social.*velocity|social.*consensus|debunk.*velocity|amplification.*rate|sentiment.*shift', body):
        result["social_velocity_anomaly"] = 1

    return result


def save_parsed_samples(samples, output_path="./output/phase3_prompt_samples.json"):
    """Save a sample of parsed LLM outputs for audit/inspection.

    Args:
        samples: list of dicts, each with keys:
            - event_id: str
            - raw_output: str
            - parsed: dict (output of parse_cot_output)
            - headline: str
        output_path: JSON output path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Only keep last 300 chars of raw output for readability
    display_samples = []
    for s in samples:
        display = {
            "event_id": s.get("event_id", "unknown"),
            "headline_preview": s.get("headline", "")[:100],
            "raw_output_suffix": s.get("raw_output", "")[-300:],
            "parsed": {k: v for k, v in s.get("parsed", {}).items()},
        }
        display_samples.append(display)

    with open(output_path, "w") as f:
        json.dump(display_samples, f, indent=2)
    print(f"[CoTParser] Saved {len(display_samples)} parsed samples to {output_path}")
