"""Repair-and-rerun phase: repaired Voting, MoA, RAG on the exact same 30 items.

CRITICAL: Loads original test items from output/real_eval/raw_outputs/ so that
before/after comparison uses IDENTICAL items. This is necessary because the
finance adapter now loads from CSV (was synthetic fallback during the original run).

Design decisions implemented:
1. Voting N=3 rebalanced: 3 distinct roles (neutral, skeptical, base-rate-aware),
   soft aggregation allowing REAL/FAKE/ESCALATE, no forced binary majority.
2. RAG redesigned around evidence categories (supporting, contradicting,
   source reliability, insufficient-evidence) — no label leakage.
3. Political-domain weakness treated as N=10 pilot signal.
4. Single-Shot re-run unchanged as control.

Output: output/repair_rerun/report.json with before/after comparison.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# Bypass system proxy
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

from src.datasets import load_dataset
from src.metrics import (
    classification_metrics,
    compute_confusion_matrix,
    compute_ece,
    compute_ppv,
)
from src.prompts import (
    SINGLE_SHOT_SYSTEM,
    SINGLE_SHOT_USER,
    format_user_prompt,
)
from src.schemas import (
    Verdict,
    VerificationItem,
    VerificationResult,
)

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

SAMPLE_SIZE = 10
SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 512

DOMAINS = ["finance", "healthcare", "political"]

KAGGLE_FAKE_NEWS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "political", "kaggle_fake_news.csv"
)
FINANCIAL_NEWS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "finance", "financial_news.csv"
)

ORIGINAL_RAW_DIR = os.path.join(
    os.path.dirname(__file__), "..", "output", "real_eval", "raw_outputs"
)

PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]

OUTPUT_DIR = "output/repair_rerun"


# ── Thread-safe API helpers ─────────────────────────────────────────────────

def _make_openai_client():
    import httpx
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    http_client = httpx.Client(
        proxy=None,
        timeout=httpx.Timeout(60.0, connect=15.0),
        follow_redirects=True,
    )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        http_client=http_client,
    )


def _llm_call(system_prompt: str, user_prompt: str, model: str = MODEL,
              temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> Tuple[str, float]:
    client = _make_openai_client()
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        elapsed = time.time() - start
        return resp.choices[0].message.content or "", elapsed
    except Exception as e:
        elapsed = time.time() - start
        raise RuntimeError(f"API call failed after {elapsed:.1f}s: {e}")


# ── Response Parser ─────────────────────────────────────────────────────────

_VERDICT_PATTERN = re.compile(r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)
_FLAGS_PATTERN = re.compile(r"Flags:\s*\[([^\]]*)\]", re.IGNORECASE)
_REASONING_PATTERN = re.compile(r"Reasoning:\s*(.+)", re.IGNORECASE)

_VERDICT_MAP = {
    "REAL": Verdict.REAL,
    "FAKE": Verdict.FAKE,
    "ESCALATE": Verdict.ESCALATE,
    "EXAGGERATED": Verdict.EXAGGERATED,
}


def _parse_response(raw: str, item_id: str, latency_s: float,
                    extra_metadata: Optional[Dict[str, Any]] = None) -> VerificationResult:
    verdict_match = _VERDICT_PATTERN.search(raw)
    verdict = Verdict.REAL
    if verdict_match:
        key = verdict_match.group(1).upper()
        verdict = _VERDICT_MAP.get(key, Verdict.REAL)

    confidence_match = _CONFIDENCE_PATTERN.search(raw)
    confidence = 0.5
    if confidence_match:
        confidence = int(confidence_match.group(1)) / 100.0

    flags_match = _FLAGS_PATTERN.search(raw)
    flags = flags_match.group(1) if flags_match else ""

    reasoning_match = _REASONING_PATTERN.search(raw)
    reasoning = reasoning_match.group(1) if reasoning_match else ""

    evidence = []
    if flags:
        evidence.append(f"Flags: {flags}")
    if reasoning:
        evidence.append(reasoning)

    metadata = {"model": MODEL, "prompt_template": "repair_rerun"}
    if extra_metadata:
        metadata.update(extra_metadata)

    return VerificationResult(
        item_id=item_id,
        verdict=verdict,
        confidence=min(max(confidence, 0.0), 1.0),
        latency_s=latency_s,
        evidence=evidence,
        metadata=metadata,
    )


# ══════════════════════════════════════════════════════════════════════════
#  ORIGINAL ITEM LOADER — reads the exact same test items as the first run
# ══════════════════════════════════════════════════════════════════════════

def _load_original_items_from_raw(domain: str) -> Tuple[List[VerificationItem], List[VerificationItem], str]:
    """Load the exact test items from the original run's saved raw outputs.

    Returns (test_items, rag_corpus, rag_notes).
    """
    raw_path = os.path.join(ORIGINAL_RAW_DIR, f"{domain}_single_shot.json")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Original raw output not found: {raw_path}")

    with open(raw_path, "r") as f:
        data = json.load(f)

    items_raw = data.get("items", [])
    test_items = []
    for it in items_raw:
        claim = it.get("claim", "")
        gt_raw = it.get("ground_truth")
        if gt_raw and gt_raw != "None":
            gt = Verdict[gt_raw]
        else:
            # Map known claims from synthetic fallback
            gt = _infer_ground_truth(claim, domain)
        test_items.append(
            VerificationItem.create(
                claim_text=claim,
                ground_truth=gt,
                metadata={"domain": domain, "source": "original_run_reconstructed"},
            )
        )

    # Build RAG corpus
    rag_corpus, rag_notes = _build_rag_corpus(domain, test_items)

    return test_items, rag_corpus, rag_notes


_SYNTHETIC_FAKE_HEADLINES = [
    "Breaking: Federal Reserve declares all bank deposits void effective immediately",
    "SEC announces Apple is under investigation for accounting fraud",
    "Elon Musk announces Tesla is filing for bankruptcy",
    "JP Morgan reports $50 billion trading loss due to algorithm error",
    "Amazon to lay off 40% of workforce in cost-cutting measure",
]

_SYNTHETIC_REAL_HEADLINES = [
    "Federal Reserve holds interest rates steady at 5.25%",
    "Apple reports Q4 revenue of $89.5 billion, beating estimates",
    "Tesla delivers 435,000 vehicles in Q3 2024",
    "Microsoft announces $10 billion share buyback program",
    "Goldman Sachs upgrades S&P 500 target to 6,000",
]

_SYNTHETIC_ALL = set(_SYNTHETIC_FAKE_HEADLINES + _SYNTHETIC_REAL_HEADLINES)


def _infer_ground_truth(claim: str, domain: str) -> Optional[Verdict]:
    """Infer ground truth for known synthetic claims or None if unknown."""
    # Check finance synthetic items
    if claim in _SYNTHETIC_FAKE_HEADLINES:
        return Verdict.FAKE
    if claim in _SYNTHETIC_REAL_HEADLINES:
        return Verdict.REAL
    return None


def _build_rag_corpus(domain: str, test_items: List[VerificationItem]) -> Tuple[List[VerificationItem], str]:
    """Build RAG corpus matching the original run's approach."""
    test_ids = {it.id for it in test_items}

    if domain == "finance":
        # Read from CSV for corpus (same as original run)
        corpus = []
        try:
            with open(FINANCIAL_NEWS_PATH, "r", encoding="utf-8") as f:
                import csv
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = row.get("text", row.get("title", ""))
                    if len(text) > 50:
                        corpus.append(VerificationItem.create(
                            claim_text=text[:2000],
                            ground_truth=Verdict.REAL,
                            metadata={"domain": "finance", "source": "financial_news_.csv"},
                        ))
                        if i >= 199:
                            break
            return corpus, f"Corpus: {len(corpus)} real financial news articles"
        except Exception as e:
            return [], f"RAG skipped: {e}"

    elif domain == "healthcare":
        adapter = load_dataset("healthcare")
        all_items = adapter.load()
        corpus = [it for it in all_items if it.id not in test_ids]
        return corpus, f"Corpus: {len(corpus)} healthcare items"

    elif domain == "political":
        # Read from CSV for corpus
        rng = random.Random(SEED)
        try:
            with open(fix_bom(KAGGLE_FAKE_NEWS_PATH), "r", encoding="utf-8-sig") as f:
                import csv
                reader = csv.DictReader(f)
                lines = list(reader)
            fake, real = [], []
            for row in lines:
                title = row.get("title", row.get("text", ""))
                label_raw = row.get("label", "0")
                if not title:
                    continue
                try:
                    label = int(label_raw.strip())
                except (ValueError, AttributeError):
                    continue
                item = VerificationItem.create(
                    claim_text=title[:500],
                    ground_truth=Verdict.FAKE if label == 1 else Verdict.REAL,
                    metadata={"domain": "political", "source": "kaggle_fake_news_FULL.csv"},
                )
                (fake if label == 1 else real).append(item)
            test_ids_set = {it.id for it in test_items}
            corpus = [it for it in fake + real if it.id not in test_ids_set][:200]
            return corpus, f"Corpus: {len(corpus)} political items"
        except Exception as e:
            return [], f"RAG skipped: {e}"

    return [], "Unknown domain"


def fix_bom(path: str) -> str:
    """Read file, strip BOM, return path to a temp file without BOM."""
    # If the file has a BOM, create a temp file without it
    import tempfile
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b'\xef\xbb\xbf'):
        tmp = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv')
        tmp.write(raw[3:])
        tmp.close()
        return tmp.name
    return path


# ══════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE RUNNERS
# ══════════════════════════════════════════════════════════════════════════

# -- Repaired Voting N=3 --

VOTER_NEUTRAL_SYSTEM = """You are a Neutral Evidence Verifier. Examine the following claim for evidence patterns without bias.

Analyze systematically:
1. Internal consistency: Are the claims logically self-consistent?
2. Plausibility: Is the magnitude within reasonable bounds?
3. Specificity: Does it include specific, verifiable details?

Weigh supporting and contradicting evidence equally. Report what you actually find.

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""

VOTER_SKEPTIC_SYSTEM = """You are a Skeptical Misinformation Detector. Your job is to scrutinize the claim for specific red flags.

Look for:
1. Logical contradictions or impossibilities
2. Exaggerated or implausible magnitudes/numbers
3. Temporal inconsistencies or anachronisms
4. Hallmarks of misinformation: emotional manipulation, lack of specifics, conspiracy framing

Only flag as FAKE when you find CONCRETE, SPECIFIC evidence of misinformation.
If the claim seems plausible with no clear red flags, report REAL.
If uncertain, report ESCALATE.

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""

VOTER_CALIBRATED_SYSTEM = """You are a Base-Rate-Calibrated Verifier. You understand that most news is genuine.

Remember that genuine, authentic news is the majority in established reporting. Before flagging a claim as FAKE, ensure the evidence against authenticity is clear and unambiguous. The cost of false positives (flagging real news as fake) is significant.

Consider:
1. Is there clear evidence this is false, or is it merely surprising?
2. Could this be a real event that sounds unusual but is actually plausible?
3. Would a reasonable person with domain knowledge accept this claim?

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>"""

VOTER_PROMPTS = [
    ("neutral", VOTER_NEUTRAL_SYSTEM),
    ("skeptical", VOTER_SKEPTIC_SYSTEM),
    ("calibrated", VOTER_CALIBRATED_SYSTEM),
]


def _run_repaired_voting(items: List[VerificationItem]) -> List[VerificationResult]:
    """Repaired Voting N=3 with balanced roles and soft aggregation."""
    results = []
    for item in items:
        start = time.time()
        user_prompt = format_user_prompt(
            SINGLE_SHOT_USER,
            claim_text=item.claim_text,
            context=item.context or "",
        )

        voter_outputs = []
        for role_name, system_prompt in VOTER_PROMPTS:
            try:
                raw, lat = _llm_call(system_prompt, user_prompt, temperature=0.3)
                parsed = _parse_response(raw, item.id, lat,
                                         extra_metadata={"voter_role": role_name})
            except Exception as e:
                parsed = VerificationResult(
                    item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                    latency_s=0, metadata={"voter_role": role_name, "error": str(e)},
                )
            voter_outputs.append(parsed)

        total_latency = time.time() - start

        # Soft aggregation: 2+ majority wins; else ESCALATE
        verdicts = [v.verdict for v in voter_outputs]
        counter = Counter(verdicts)
        most_common = counter.most_common(1)

        if most_common and most_common[0][1] >= 2:
            final_verdict = most_common[0][0]
        else:
            final_verdict = Verdict.ESCALATE

        avg_conf = float(np.mean([v.confidence for v in voter_outputs]))
        agreement = max(counter.values()) / len(verdicts) if verdicts else 0.0

        evidence = [
            f"Repaired Voting N=3: {len(voter_outputs)} voters",
            f"Aggregation: soft (2+ majority, else ESCALATE)",
            f"Agreement rate: {agreement:.2f}",
            f"Verdict distribution: {dict(counter)}",
        ]
        for role_name, _ in VOTER_PROMPTS:
            matches = [v for v in voter_outputs if v.metadata.get("voter_role") == role_name]
            if matches:
                m = matches[0]
                evidence.append(f"{role_name}: {m.verdict.name} (conf={m.confidence:.2f})")

        results.append(VerificationResult(
            item_id=item.id, verdict=final_verdict, confidence=avg_conf,
            latency_s=total_latency, evidence=evidence,
            metadata={
                "model": MODEL,
                "architecture": "repaired_voting_n3",
                "n_voters": len(voter_outputs),
                "agreement_rate": agreement,
                "verdict_distribution": {k.name: v for k, v in counter.items()},
                "per_voter": [
                    {"role": role_name, "verdict": v.verdict.name, "confidence": v.confidence}
                    for (role_name, _), v in zip(VOTER_PROMPTS, voter_outputs)
                ],
            },
        ))
    return results


# -- Repaired MoA --

REPAIRED_BELIEVER_SYSTEM = """You are Agent 1 (The Believer) in an information verification debate.
Your role is to find evidence SUPPORTING the authenticity of the claim. Argue the claim is REAL.

Build the strongest possible case for authenticity by examining:
1. Source credibility: Could this originate from a legitimate source?
2. Factual alignment: Does the claim match known facts and reasonable expectations?
3. Internal consistency: Are the claims internally logical and coherent?
4. Plausibility: Is the magnitude within the realm of possibility?

Be thorough — list specific supporting evidence. It's OK to challenge the claim if you find no supporting evidence.

Output:
Verdict: REAL or UNCERTAIN
Evidence: <bulleted list of supporting points>
Confidence: <0-100>"""

REPAIRED_SKEPTIC_SYSTEM = """You are Agent 2 (The Skeptic) in an information verification debate.
Your role is to find evidence that the claim is FALSE or misleading. Argue the claim is FAKE.

Build the strongest possible case against authenticity by examining:
1. Contradictions: Does the claim contradict known facts?
2. Implausibility: Are specific numbers or claims unrealistic or extreme?
3. Inconsistencies: Are there temporal or logical inconsistencies?
4. Red flags: Are there hallmarks of misinformation (clickbait language, lack of sources, emotional manipulation)?

Be thorough — list specific problematic evidence. Only argue FAKE if you find concrete issues.

Output:
Verdict: FAKE or UNCERTAIN
Evidence: <bulleted list of problematic points>
Confidence: <0-100>"""

REPAIRED_RISK_OFFICER_SYSTEM = """You are Agent 3 (The Risk Officer) in an information verification debate.

You have received analyses from two agents:
- The Believer argues the claim is REAL.
- The Skeptic argues the claim is FAKE.

Your job is to evaluate BOTH arguments critically and deliver a final verdict.

IMPORTANT GUIDELINES:
1. Weigh evidence QUALITY, not just presence. Specific verifiable evidence outweighs vague suspicion.
2. The cost of false positives (flagging REAL as FAKE) is HIGH — it causes unnecessary reversals.
3. The cost of false negatives (missing FAKE) is also HIGH — it causes holding fake-news positions.
4. Default to REAL when the Believer provides more specific evidence than the Skeptic.
5. Default to FAKE when the Skeptic provides clear, specific contradictions.
6. Default to ESCALATE when evidence is roughly balanced or both sides are weak.
7. Be calibrated — your confidence should reflect your actual certainty.

Output your final verdict in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale weighing both sides>"""


def _run_repaired_moa(items: List[VerificationItem]) -> List[VerificationResult]:
    """Repaired MoA with calibrated Risk Officer and degeneracy tracking."""
    results = []
    for item in items:
        start = time.time()
        user_prompt = f"Claim to verify:\n{item.claim_text}\n\nContext:\n{item.context or 'None provided.'}"

        try:
            bel_raw, bel_lat = _llm_call(REPAIRED_BELIEVER_SYSTEM, user_prompt)
        except Exception as e:
            bel_raw = f"Error: {e}"
            bel_lat = 0

        try:
            skep_raw, skep_lat = _llm_call(REPAIRED_SKEPTIC_SYSTEM, user_prompt)
        except Exception as e:
            skep_raw = f"Error: {e}"
            skep_lat = 0

        synthesis_prompt = (
            f"Claim to verify:\n{item.claim_text}\n\n"
            f"=== Believer's Analysis (pro-REAL) ===\n{bel_raw}\n\n"
            f"=== Skeptic's Analysis (pro-FAKE) ===\n{skep_raw}\n\n"
            f"Based on both analyses above, deliver your final verdict."
        )

        try:
            ro_raw, ro_lat = _llm_call(REPAIRED_RISK_OFFICER_SYSTEM, synthesis_prompt)
        except Exception as e:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=time.time() - start,
                evidence=[f"Risk Officer error: {e}"],
                metadata={"error": str(e), "believer_output": bel_raw[:200],
                          "skeptic_output": skep_raw[:200],
                          "architecture": "repaired_moa"},
            ))
            continue

        total_latency = time.time() - start
        r = _parse_response(ro_raw, item.id, total_latency,
                            extra_metadata={
                                "architecture": "repaired_moa",
                                "believer_output": bel_raw[:200],
                                "skeptic_output": skep_raw[:200],
                                "risk_officer_output": ro_raw[:200],
                            })
        bel_conf = re.search(r"Confidence:\s*(\d+)", bel_raw)
        skep_conf = re.search(r"Confidence:\s*(\d+)", skep_raw)
        r.evidence = [
            f"Believer confidence: {bel_conf.group(1) if bel_conf else 'unknown'}",
            f"Skeptic confidence: {skep_conf.group(1) if skep_conf else 'unknown'}",
        ] + r.evidence
        results.append(r)
    return results


# -- Repaired RAG (Evidence-Centric) --

EVIDENCE_RAG_SYSTEM = """You are an Evidence-Centric Retrieval-Augmented Verifier.

You receive:
1. A CLAIM to verify.
2. RETRIEVED EVIDENCE from a knowledge corpus — documents ranked by relevance to the claim.

Your task is to analyze the evidence by category and reach a verdict:

A. SUPPORTING EVIDENCE — Do any retrieved documents contain facts that align with or corroborate the claim?
B. CONTRADICTING EVIDENCE — Do any retrieved documents contain facts that directly contradict the claim?
C. SOURCE RELIABILITY — What can you infer about the sources? Are they authoritative?
D. EVIDENCE GAPS — Does the corpus cover this topic? If not, note insufficient evidence.

VERDICT RULES:
- If supporting evidence found AND no contradicting evidence → REAL
- If contradicting evidence found AND it clearly disproves the claim → FAKE
- If evidence is mixed (both support and contradict) → ESCALATE (weigh both sides)
- If no relevant evidence found (evidence gap) → ESCALATE (insufficient information)
- If the claim is obviously true/false independent of corpus → use your best judgment

Output in EXACTLY this format:
Verdict: REAL or FAKE or ESCALATE
Confidence: <0-100>
Flags: [contradiction|implausibility|inconsistency|none]
Reasoning: <one-sentence rationale>
Evidence quality: <supporting|contradicting|mixed|insufficient>"""

EVIDENCE_RAG_USER = """Claim to verify:
{claim_text}

── Retrieved Evidence (from knowledge corpus) ──
{context}
──────────────────────────────────────────────

Analyze the evidence above by category (supporting, contradicting, source reliability, gaps).
Then deliver your verdict."""


def _run_repaired_rag(items: List[VerificationItem],
                      corpus: List[VerificationItem]) -> List[VerificationResult]:
    """Repaired RAG with evidence-category analysis — no label leakage."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus_texts = [it.claim_text for it in corpus]
    corpus_ids = [it.id for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vec.fit_transform(corpus_texts)

    results = []
    for item in items:
        start = time.time()

        query_vec = vec.transform([item.claim_text])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_k = 3
        top_indices = np.argsort(sims)[::-1][:top_k]

        hits = []
        for idx in top_indices:
            if sims[idx] <= 0:
                continue
            hits.append((corpus_ids[idx], corpus_texts[idx], float(sims[idx])))

        retrieval_lat = time.time() - start

        context_parts = []
        for doc_id, doc_text, score in hits:
            context_parts.append(
                f"[Document {doc_id}] (relevance={score:.3f})\n{doc_text}\n"
            )
        context = "".join(context_parts)[:2000]

        user_prompt = EVIDENCE_RAG_USER.format(claim_text=item.claim_text, context=context)
        try:
            raw, llm_lat = _llm_call(EVIDENCE_RAG_SYSTEM, user_prompt)
        except Exception as e:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=time.time() - start,
                evidence=[f"RAG LLM error: {e}"],
                metadata={"retrieval_hits": len(hits), "error": str(e),
                          "architecture": "repaired_rag"},
            ))
            continue

        total_latency = time.time() - start
        r = _parse_response(raw, item.id, total_latency, extra_metadata={
            "architecture": "repaired_rag",
            "retriever_type": "sparse",
            "top_k": top_k,
            "retrieval_hits": len(hits),
            "evidence_ids": [doc_id for doc_id, _, _ in hits],
            "relevance_scores": [s for _, _, s in hits],
            "retrieval_latency_s": round(retrieval_lat, 4),
            "llm_latency_s": round(llm_lat, 4),
            "context_length_chars": len(context),
        })
        retrieval_evidence = [
            f"[Retrieved] doc={doc_id} score={s:.3f}: {text[:120]}"
            for doc_id, text, s in hits
        ]
        r.evidence = retrieval_evidence + r.evidence
        results.append(r)
    return results


# -- Unchanged Single-Shot (control) --

def _run_single_shot(items: List[VerificationItem]) -> List[VerificationResult]:
    """Sequential single-shot verification (unchanged)."""
    results = []
    for item in items:
        user_prompt = format_user_prompt(SINGLE_SHOT_USER, claim_text=item.claim_text,
                                         context=item.context or "")
        try:
            raw, lat = _llm_call(SINGLE_SHOT_SYSTEM, user_prompt)
            r = _parse_response(raw, item.id, lat)
        except Exception as e:
            r = VerificationResult(item_id=item.id, verdict=Verdict.REAL,
                                   confidence=0.5, latency_s=0, evidence=[str(e)],
                                   metadata={"error": str(e)})
        results.append(r)
    return results


# ── Metrics helpers ─────────────────────────────────────────────────────────


def _latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    arr = sorted(latencies)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "min": float(min(arr)), "max": float(max(arr))}


def _compute_ppv_table(tp: int, fp: int, tn: int, fn: int) -> List[Dict[str, float]]:
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    ppvs = compute_ppv(sensitivity=sens, specificity=spec, base_rates=PPV_BASE_RATES)
    return [{"base_rate": br, "ppv": ppv, "npv": npv} for br, ppv, npv in ppvs
            if br in PPV_BASE_RATES]


def find_failures(results: List[VerificationResult], items: List[VerificationItem]) -> List[Dict[str, Any]]:
    failures = []
    for r, it in zip(results, items):
        if it.ground_truth is None:
            continue
        if r.verdict != it.ground_truth:
            ft = "FN" if it.ground_truth == Verdict.FAKE else "FP"
            failures.append({"item_id": it.id, "claim": it.claim_text[:150],
                             "domain": it.metadata.get("domain", ""), "type": ft,
                             "truth": it.ground_truth.name, "predicted": r.verdict.name,
                             "confidence": r.confidence, "latency_s": r.latency_s})
    return failures


def diagnose_moa_degeneracy(results: List[VerificationResult], items: List[VerificationItem],
                            domain: str) -> Dict[str, Any]:
    fake_preds = sum(1 for r in results if r.verdict == Verdict.FAKE)
    total = len(results)
    fake_fraction = fake_preds / total if total else 0.0
    base_rate = sum(1 for it in items if it.ground_truth == Verdict.FAKE) / len(items) if items else 0.0
    cm = compute_confusion_matrix(results, [it.ground_truth or Verdict.REAL for it in items])
    prec = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0

    believer_confs, skeptic_confs = [], []
    for r in results:
        bel = r.metadata.get("believer_output", "")
        skep = r.metadata.get("skeptic_output", "")
        m = re.search(r"Confidence:\s*(\d+)", bel)
        if m:
            believer_confs.append(int(m.group(1)))
        m = re.search(r"Confidence:\s*(\d+)", skep)
        if m:
            skeptic_confs.append(int(m.group(1)))

    info_gap = prec - base_rate

    return {
        "domain": domain,
        "fake_prediction_rate": fake_fraction,
        "test_set_fake_base_rate": base_rate,
        "precision": prec,
        "precision_above_base_rate": prec > base_rate,
        "info_gap_precision_minus_base_rate": info_gap,
        "likely_degenerate": prec <= base_rate,
        "avg_believer_confidence": float(np.mean(believer_confs)) if believer_confs else None,
        "avg_skeptic_confidence": float(np.mean(skeptic_confs)) if skeptic_confs else None,
        "believer_skeptic_divergence": (
            float(np.mean(skeptic_confs) - np.mean(believer_confs))
            if believer_confs and skeptic_confs else None),
    }


def rag_evidence_quality(results: List[VerificationResult]) -> Dict[str, Any]:
    verdicts = [r.verdict.name for r in results]
    vc = Counter(verdicts)
    escalate_rate = vc.get("ESCALATE", 0) / len(results) if results else 0.0
    fake_rate = vc.get("FAKE", 0) / len(results) if results else 0.0
    real_rate = vc.get("REAL", 0) / len(results) if results else 0.0

    retrieval_hits = [r.metadata.get("retrieval_hits", 0) for r in results]
    avg_hits = float(np.mean(retrieval_hits)) if retrieval_hits else 0.0

    avg_conf = float(np.mean([r.confidence for r in results])) if results else 0.0

    return {
        "n_items": len(results),
        "verdict_distribution": dict(vc),
        "escalate_rate": escalate_rate,
        "fake_prediction_rate": fake_rate,
        "real_prediction_rate": real_rate,
        "avg_confidence": avg_conf,
        "retrieval_avg_hits": avg_hits,
        "assessment": "Good retrieval quality" if avg_hits >= 2 else "Limited retrieval",
    }


# ── Evaluation runner ──────────────────────────────────────────────────────


@dataclass
class EvalResult:
    architecture: str
    domain: str
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    verdict_distribution: Dict[str, int] = field(default_factory=dict)
    ppv_table: List[Dict[str, float]] = field(default_factory=list)
    ece: float = 0.0
    failure_examples: List[Dict[str, Any]] = field(default_factory=list)
    raw_output_path: str = ""


def run_evaluation(arch_name: str, domain_data: Tuple,
                   runner_fn: Callable, output_dir: str,
                   api_count: int = 0) -> Tuple[EvalResult, int]:
    items, _, _ = domain_data
    domain = items[0].metadata.get("domain", "unknown") if items else "unknown"
    result = EvalResult(architecture=arch_name, domain=domain)

    print(f"\n{'='*60}")
    print(f"  [{domain}] Running {arch_name} on {len(items)} items...")
    print(f"{'='*60}")

    try:
        raw_results = runner_fn(items)
    except Exception as e:
        traceback.print_exc()
        result.error = str(e)
        return result, 0

    latencies = []
    result_list = []
    for r in raw_results:
        latencies.append(r.latency_s)
        result_list.append({
            "item_id": r.item_id, "verdict": r.verdict.name,
            "confidence": r.confidence, "latency_s": r.latency_s,
            "evidence": r.evidence[:5],
            "metadata": {k: v for k, v in r.metadata.items()
                         if k not in ("raw_output_first_100",)},
        })

    result.results = result_list
    result.latency = _latency_stats(latencies)

    truths = [it.ground_truth or Verdict.REAL for it in items]
    try:
        cm = compute_confusion_matrix(raw_results, truths)
        metrics = classification_metrics(raw_results, truths)
        result.metrics = asdict(metrics)
        result.verdict_distribution = dict(Counter(r.verdict.name for r in raw_results))
        result.ppv_table = _compute_ppv_table(cm.tp, cm.fp, cm.tn, cm.fn)
        result.failure_examples = find_failures(raw_results, items)
        correct = [r.verdict == t for r, t in zip(raw_results, truths)]
        ece, _, _, _ = compute_ece([r.confidence for r in raw_results], correct, n_bins=5)
        result.ece = ece
    except Exception as e:
        result.error = f"Metrics error: {e}"

    os.makedirs(os.path.join(output_dir, "raw_outputs"), exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_outputs", f"{domain}_{arch_name}.json")
    with open(raw_path, "w") as f:
        json.dump({"architecture": arch_name, "domain": domain, "model": MODEL,
                   "items": [{"id": it.id, "claim": it.claim_text[:200],
                              "ground_truth": it.ground_truth.name if it.ground_truth else None}
                             for it in items],
                   "results": [{"item_id": r.item_id, "verdict": r.verdict.name,
                                "confidence": r.confidence, "latency_s": r.latency_s,
                                "evidence": r.evidence, "metadata": r.metadata}
                               for r in raw_results]},
                  f, indent=2, default=str)
    result.raw_output_path = raw_path

    m = result.metrics
    print(f"  └─ F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  "
          f"R={m.get('recall', 0):.4f}  "
          f"Lat={result.latency.get('mean', 0):.1f}s  ECE={result.ece:.4f}")
    if result.failure_examples:
        fn_c = sum(1 for f in result.failure_examples if f['type'] == 'FN')
        fp_c = sum(1 for f in result.failure_examples if f['type'] == 'FP')
        print(f"     Failures: {len(result.failure_examples)} ({fn_c} FN, {fp_c} FP)")

    return result, api_count or len(items)


# ── Load prior results for before/after comparison ─────────────────────────


def load_prior_results(path: str = "output/real_eval/report.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def build_before_after(prior: Dict[str, Any],
                       current: Dict[str, Any]) -> Dict[str, Any]:
    comparison = {"domains": {}, "cross_domain": {}}

    for domain in DOMAINS:
        domain_cmp = {}
        for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
            old_name = {
                "repaired_voting_n3": "voting_n3",
                "repaired_moa": "moa",
                "repaired_rag": "rag",
            }.get(arch_name, arch_name)

            prior_f1 = prior_p = prior_r = prior_ece = prior_lat = None
            if prior:
                pd = prior.get("domains", {}).get(domain, {}).get(old_name, {})
                pm = pd.get("metrics", {})
                if pm:
                    prior_f1 = pm.get("f1")
                    prior_p = pm.get("precision")
                    prior_r = pm.get("recall")
                    prior_ece = pd.get("ece")
                    prior_lat = pd.get("latency", {}).get("mean")

            cd = current.get("domains", {}).get(domain, {}).get(arch_name, {})
            cm = cd.get("metrics", {})
            current_f1 = cm.get("f1")
            current_p = cm.get("precision")
            current_r = cm.get("recall")
            current_ece = cd.get("ece")
            current_lat = cd.get("latency", {}).get("mean")

            delta_f1 = (current_f1 - prior_f1) if (current_f1 is not None and prior_f1 is not None) else None

            domain_cmp[arch_name] = {
                "before": {"f1": prior_f1, "precision": prior_p, "recall": prior_r,
                           "ece": prior_ece, "mean_latency_s": prior_lat},
                "after": {"f1": current_f1, "precision": current_p, "recall": current_r,
                          "ece": current_ece, "mean_latency_s": current_lat},
                "delta_f1": delta_f1,
                "improved": delta_f1 > 0 if delta_f1 is not None else None,
            }
        comparison["domains"][domain] = domain_cmp

    for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
        old_name = {
            "repaired_voting_n3": "voting_n3",
            "repaired_moa": "moa",
            "repaired_rag": "rag",
        }.get(arch_name, arch_name)

        current_f1s = {}
        for d in DOMAINS:
            cd = current.get("domains", {}).get(d, {}).get(arch_name, {})
            cm = cd.get("metrics", {})
            if cm.get("f1") is not None:
                current_f1s[d] = cm["f1"]

        prior_f1s = {}
        if prior:
            for d in DOMAINS:
                pd = prior.get("domains", {}).get(d, {}).get(old_name, {})
                pm = pd.get("metrics", {})
                if pm.get("f1") is not None:
                    prior_f1s[d] = pm["f1"]

        cv = list(current_f1s.values())
        pv = list(prior_f1s.values())

        ss_mean = current.get("cross_domain", {}).get("single_shot", {}).get("mean_f1", 0)

        comparison["cross_domain"][arch_name] = {
            "before_mean_f1": float(np.mean(pv)) if pv else None,
            "after_mean_f1": float(np.mean(cv)) if cv else None,
            "delta": (float(np.mean(cv)) - float(np.mean(pv))) if cv and pv else None,
            "beats_single_shot": float(np.mean(cv)) > ss_mean if cv else None,
        }

    return comparison


# ── Main ────────────────────────────────────────────────────────────────────


def main(output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Verify connectivity
    print("Verifying DeepSeek API connectivity...")
    try:
        raw, lat = _llm_call("You are a test assistant.", "Reply with a single word: OK.")
        print(f"  API OK — response: {raw[:50]} ({lat:.1f}s)")
    except Exception as e:
        print(f"  API ERROR: {e}")
        return

    # Load original test items
    print(f"\n{'='*60}")
    print("  Loading original test items from saved raw outputs...")
    print(f"{'='*60}")
    domain_data: Dict[str, Tuple] = {}
    for domain_name in DOMAINS:
        test_items, rag_corpus, rag_notes = _load_original_items_from_raw(domain_name)
        domain_data[domain_name] = (test_items, rag_corpus, rag_notes)
        n_fake = sum(1 for it in test_items if it.ground_truth == Verdict.FAKE)
        n_real = sum(1 for it in test_items if it.ground_truth == Verdict.REAL)
        n_unlabeled = sum(1 for it in test_items if it.ground_truth is None)
        print(f"  {domain_name}: {len(test_items)} test items "
              f"(F:{n_fake} R:{n_real} ?:{n_unlabeled})  "
              f"RAG: {rag_notes}")

    # ── Run architectures ────────────────────────────────────────────

    domain_results: Dict[str, Dict[str, EvalResult]] = {}
    all_api_counts: Dict[str, int] = {}

    for domain_name in DOMAINS:
        data = domain_data[domain_name]
        items, rag_corpus, rag_notes = data
        domain_results[domain_name] = {}

        # Single-Shot (unchanged control)
        result, _ = run_evaluation("single_shot", data, _run_single_shot, output_dir, SAMPLE_SIZE)
        domain_results[domain_name]["single_shot"] = result
        all_api_counts["single_shot"] = all_api_counts.get("single_shot", 0) + SAMPLE_SIZE

        # Repaired Voting N=3
        result, _ = run_evaluation("repaired_voting_n3", data, _run_repaired_voting, output_dir, SAMPLE_SIZE * 3)
        domain_results[domain_name]["repaired_voting_n3"] = result
        all_api_counts["repaired_voting_n3"] = all_api_counts.get("repaired_voting_n3", 0) + SAMPLE_SIZE * 3

        # Repaired MoA
        result, _ = run_evaluation("repaired_moa", data, _run_repaired_moa, output_dir, SAMPLE_SIZE * 3)
        domain_results[domain_name]["repaired_moa"] = result
        all_api_counts["repaired_moa"] = all_api_counts.get("repaired_moa", 0) + SAMPLE_SIZE * 3

        # Repaired RAG (needs corpus)
        if rag_corpus and len(rag_corpus) >= 3:
            def make_rag_runner(corpus=rag_corpus):
                return lambda its: _run_repaired_rag(its, corpus)
            result, _ = run_evaluation("repaired_rag", data, make_rag_runner(), output_dir, SAMPLE_SIZE)
        else:
            result = EvalResult(architecture="repaired_rag", domain=domain_name,
                                skipped=True, skip_reason=rag_notes or "No corpus")
        domain_results[domain_name]["repaired_rag"] = result
        all_api_counts["repaired_rag"] = all_api_counts.get("repaired_rag", 0) + SAMPLE_SIZE

    # ── Build report ──
    report = {
        "metadata": {
            "model": MODEL,
            "timestamp": datetime.utcnow().isoformat(),
            "sample_size_per_domain": SAMPLE_SIZE,
            "total_api_calls": sum(all_api_counts.values()),
            "api_calls_by_architecture": all_api_counts,
            "estimated_cost_usd": sum(all_api_counts.values()) * 0.00015,
            "note": "Repair-and-rerun phase. Items loaded from original raw outputs for identical comparison.",
        },
        "domains": {},
        "cross_domain": {},
        "moa_degeneracy": {},
        "rag_evidence_quality": {},
    }

    for domain_name in DOMAINS:
        arch_results = domain_results[domain_name]
        domain_section = {}
        for arch_name, er in arch_results.items():
            section = {"skipped": er.skipped, "skip_reason": er.skip_reason,
                       "error": er.error, "metrics": er.metrics, "latency": er.latency,
                       "verdict_distribution": er.verdict_distribution,
                       "ece": er.ece, "ppv": er.ppv_table,
                       "failure_count": len(er.failure_examples),
                       "failures": er.failure_examples[:5],
                       "raw_output_path": er.raw_output_path}
            domain_section[arch_name] = section
        report["domains"][domain_name] = domain_section

        # MoA degeneracy
        ar_moa = arch_results.get("repaired_moa")
        if ar_moa and not ar_moa.skipped and not ar_moa.error:
            items, _, _ = domain_data[domain_name]
            moa_res = [VerificationResult(item_id=r["item_id"], verdict=Verdict[r["verdict"]],
                                          confidence=r["confidence"], latency_s=r["latency_s"],
                                          evidence=r.get("evidence", []),
                                          metadata=r.get("metadata", {}))
                       for r in ar_moa.results]
            report["moa_degeneracy"][domain_name] = diagnose_moa_degeneracy(moa_res, items, domain_name)

        # RAG evidence quality
        ar_rag = arch_results.get("repaired_rag")
        if ar_rag and not ar_rag.skipped and not ar_rag.error:
            rag_res = [VerificationResult(item_id=r["item_id"], verdict=Verdict[r["verdict"]],
                                          confidence=r["confidence"], latency_s=r["latency_s"],
                                          evidence=r.get("evidence", []),
                                          metadata=r.get("metadata", {}))
                       for r in ar_rag.results]
            report["rag_evidence_quality"][domain_name] = rag_evidence_quality(rag_res)

    # Cross-domain metrics
    for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
        f1s = {}
        for d in DOMAINS:
            er = domain_results[d].get(arch_name)
            if er and not er.skipped and not er.error:
                f1 = er.metrics.get("f1", 0)
                if f1 is not None:
                    f1s[d] = f1
        if f1s:
            vals = list(f1s.values())
            report["cross_domain"][arch_name] = {
                "per_domain": f1s,
                "mean_f1": float(np.mean(vals)),
                "std_f1": float(np.std(vals)),
                "range_f1": max(vals) - min(vals),
            }

    # Before/after comparison
    prior = load_prior_results()
    if prior:
        report["before_after"] = build_before_after(prior, report)

    # Save report
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # ── Print compact report ──
    print_compact_report(report, prior, output_dir)


def print_compact_report(report: Dict[str, Any], prior: Dict[str, Any], output_dir: str):
    """Print the compact before/after report."""
    print(f"\n{'='*60}")
    print("  COMPACT REPAIR-AND-RERUN REPORT")
    print(f"{'='*60}")

    print(f"\n**Model**: {report['metadata']['model']}")
    print(f"**Total API calls**: {report['metadata']['total_api_calls']}")
    print(f"**Estimated cost**: ~${report['metadata']['estimated_cost_usd']:.4f}")
    print(f"**Items**: Loaded from original raw outputs — IDENTICAL to first run.")

    # After metrics table
    print(f"\n### REPAIRED METRICS (After)")
    for domain in DOMAINS:
        print(f"\n**{domain.upper()}**")
        headers = ["Architecture", "F1", "Prec", "Rec", "FPR", "ECE", "Mean Lat"]
        rows = []
        for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
            er = report.get("domains", {}).get(domain, {}).get(arch_name)
            if er is None:
                continue
            if er.get("skipped"):
                rows.append([arch_name, "SKIPPED", "", "", "", "", er.get("skip_reason", "")[:40]])
                continue
            if not er.get("metrics"):
                rows.append([arch_name, "ERROR", er.get("error", "")[:40], "", "", "", ""])
                continue
            m = er["metrics"]
            label = {"repaired_voting_n3": "voting_n3", "repaired_moa": "moa",
                     "repaired_rag": "rag", "single_shot": "single_shot"}.get(arch_name, arch_name)
            rows.append([label,
                         f"{m.get('f1', 0):.4f}", f"{m.get('precision', 0):.4f}",
                         f"{m.get('recall', 0):.4f}", f"{m.get('fpr', 0):.4f}",
                         f"{er.get('ece', 0):.4f}",
                         f"{er.get('latency', {}).get('mean', 0):.1f}s"])
        col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
        header_line = "  " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(f"\n  {header_line}")
        print(f"  " + "-" * len(header_line))
        for row in rows:
            print("  " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row)))

    # Before/After Delta
    if "before_after" in report:
        print(f"\n### BEFORE vs AFTER DELTA (ΔF1)")
        for domain in DOMAINS:
            print(f"\n**{domain.upper()}**")
            deltas = report["before_after"]["domains"][domain]
            for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
                d = deltas.get(arch_name, {})
                dd = d.get("delta_f1")
                before = d.get("before", {})
                after = d.get("after", {})
                bf = before.get("f1")
                af = after.get("f1")
                label = {"repaired_voting_n3": "voting_n3", "repaired_moa": "moa",
                         "repaired_rag": "rag", "single_shot": "single_shot"}.get(arch_name, arch_name)
                if dd is not None:
                    arrow = "↑" if dd > 0 else ("↓" if dd < 0 else "→")
                    print(f"  {label:15s} {bf:.4f} → {af:.4f}  ({arrow} {dd:+.4f})")
                elif bf is not None and af is None:
                    print(f"  {label:15s} {bf:.4f} → SKIPPED")
                else:
                    print(f"  {label:15s} N/A → {af:.4f} (new)")

    # Cross-domain summary
    print(f"\n### CROSS-DOMAIN MEAN F1")
    cd = report.get("cross_domain", {})
    for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
        acd = cd.get(arch_name, {})
        mf = acd.get("mean_f1")
        if mf is not None:
            label = {"repaired_voting_n3": "voting_n3", "repaired_moa": "moa",
                     "repaired_rag": "rag", "single_shot": "single_shot"}.get(arch_name, arch_name)
            print(f"  {label:15s}  F1={mf:.4f}  ±{acd.get('std_f1', 0):.4f}")

    # Does repaired beat Single-Shot?
    print(f"\n### DOES REPAIRED ARCHITECTURE BEAT SINGLE-SHOT?")
    ss_mean = cd.get("single_shot", {}).get("mean_f1", 0)
    for arch_name in ["repaired_voting_n3", "repaired_moa", "repaired_rag"]:
        acd = cd.get(arch_name, {})
        mf = acd.get("mean_f1")
        label = {"repaired_voting_n3": "voting_n3", "repaired_moa": "moa",
                 "repaired_rag": "rag"}.get(arch_name, arch_name)
        if mf is not None:
            beats = mf > ss_mean
            print(f"  {label:15s}  F1={mf:.4f} vs SS={ss_mean:.4f}  "
                  f"{'✅ BEATS' if beats else '❌ DOES NOT BEAT'} Single-Shot"
                  f" ({'+' if beats else ''}{mf - ss_mean:+.4f})")

    # Prediction distributions
    print(f"\n### PREDICTION DISTRIBUTIONS")
    for domain in DOMAINS:
        print(f"\n**{domain.upper()}**")
        for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
            er = report.get("domains", {}).get(domain, {}).get(arch_name, {})
            vd = er.get("verdict_distribution", {})
            if vd:
                label = {"repaired_voting_n3": "voting_n3", "repaired_moa": "moa",
                         "repaired_rag": "rag", "single_shot": "single_shot"}.get(arch_name, arch_name)
                parts = [f"{k}={v}" for k, v in sorted(vd.items())]
                print(f"  {label:15s}  {'  '.join(parts)}")

    # MoA degeneracy diagnostics
    print(f"\n### MoA DEGENERACY (Repaired)")
    for domain in DOMAINS:
        dd = report.get("moa_degeneracy", {}).get(domain, {})
        if dd:
            flag = "⚠️  DEGENERATE" if dd.get("likely_degenerate") else "✅ NOT DEGENERATE"
            print(f"  {domain}: FAKE pred rate={dd['fake_prediction_rate']:.2%} "
                  f"(base={dd['test_set_fake_base_rate']:.2%}) "
                  f"prec={dd['precision']:.4f} info_gap={dd.get('info_gap_precision_minus_base_rate', 0):+.4f} — {flag}")
            if dd.get("avg_believer_confidence") is not None:
                print(f"    Believer avg={dd['avg_believer_confidence']:.1f}  "
                      f"Skeptic avg={dd['avg_skeptic_confidence']:.1f}  "
                      f"div={dd['believer_skeptic_divergence']:.1f}")

    # RAG evidence quality
    print(f"\n### RAG EVIDENCE QUALITY (Repaired)")
    for domain in DOMAINS:
        eq = report.get("rag_evidence_quality", {}).get(domain, {})
        if eq:
            escalate = eq.get("escalate_rate", 0)
            fake = eq.get("fake_prediction_rate", 0)
            real = eq.get("real_prediction_rate", 0)
            print(f"  {domain}: ESC={escalate:.0%} FAKE={fake:.0%} REAL={real:.0%}  "
                  f"avg_hits={eq.get('retrieval_avg_hits', 0):.1f}  {eq.get('assessment', '')}")

    # Scaling recommendation
    print(f"\n### SCALING RECOMMENDATION")
    ss_mean = cd.get("single_shot", {}).get("mean_f1", 0)
    for arch_name in ["repaired_voting_n3", "repaired_moa", "repaired_rag"]:
        acd = cd.get(arch_name, {})
        mf = acd.get("mean_f1")
        label = {"repaired_voting_n3": "voting_n3", "repaired_moa": "moa",
                 "repaired_rag": "rag"}.get(arch_name, arch_name)
        if mf is not None:
            beats = mf > ss_mean
            if beats:
                print(f"  ✅ {label} (F1={mf:.4f}) beats Single-Shot ({ss_mean:.4f}) — "
                      f"worth scaling to N=50/domain for statistical confirmation.")
            else:
                delta = ss_mean - mf
                if delta < 0.05:
                    print(f"  ⚠️  {label} (F1={mf:.4f}) is within 0.05 of Single-Shot ({ss_mean:.4f}) — "
                          f"marginal, may not justify extra latency.")
                else:
                    print(f"  ❌ {label} (F1={mf:.4f}) is {delta:.4f} below Single-Shot ({ss_mean:.4f}) — "
                          f"does not justify extra complexity.")

    political_note = (
        "\n  **Political domain note**: All political results are N=10 pilot signals. "
        "The political dataset uses 2016-era news headlines which differ stylistically "
        "from finance/healthcare. Larger evaluation needed before drawing conclusions."
    )
    print(political_note)

    # Top failures
    print(f"\n### TOP FAILURE MODES (Repaired)")
    all_fails = []
    for domain in DOMAINS:
        for arch_name in ["single_shot", "repaired_voting_n3", "repaired_moa", "repaired_rag"]:
            er = report.get("domains", {}).get(domain, {}).get(arch_name, {})
            for f in er.get("failures", []):
                f["architecture"] = arch_name
                all_fails.append(f)
    all_fails.sort(key=lambda x: x.get("confidence", 1.0))
    for i, f in enumerate(all_fails[:8]):
        label = {"repaired_voting_n3": "v3", "repaired_moa": "moa",
                 "repaired_rag": "rag", "single_shot": "ss"}.get(f.get("architecture", ""), f.get("architecture", "?"))
        print(f"  {i+1}. [{f['type']}] {f.get('domain','?')}/{label}: "
              f"\"{f.get('claim', '')[:80]}...\" "
              f"(conf={f['confidence']:.2f})")

    print(f"\nFull report: {os.path.join(output_dir, 'report.json')}")
    print()


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_DIR
    main(output_dir)
