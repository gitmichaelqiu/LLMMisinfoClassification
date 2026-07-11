"""Final validation sprint: N=50 balanced evaluation with concurrent API calls.

Scales the pilot N=10 results to N=50 (or max available) for finance and
healthcare. Tests Single-Shot, Voting N=3, MoA+RAG, Voting N=3+RAG.

Uses ThreadPoolExecutor (max 400 concurrent) for fast API calling.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

from src.metrics import compute_confusion_matrix, compute_ece
from src.schemas import Verdict, VerificationItem, VerificationResult

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

SEED = 42
MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 512
DOMAINS = ["finance", "healthcare"]
OUTPUT_DIR = "results/final_validation"
MAX_CONCURRENCY = 400
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)

PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
COST_RATIOS = [(1, 1), (1, 5), (1, 10), (1, 25)]

SAMPLE_CONFIG = {
    "finance": {"n": 50, "n_real": 25, "n_fake": 25},
    "healthcare": {"n": 40, "n_real": 20, "n_fake": 20},
}

COST_PER_CALL = 0.00015
FINANCE_CSV = os.path.join("data", "raw", "finance", "financial_news.csv")
HEALTH_CSV = os.path.join("data", "raw", "health", "health_headlines.csv")


# ── API Call (creates new client per call — thread-safe) ─────────────────────


def _llm_call(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> Tuple[str, float]:
    import httpx
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    http_client = httpx.Client(
        proxy=None,
        timeout=httpx.Timeout(180.0, connect=30.0),
        follow_redirects=True,
    )
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        http_client=http_client,
    )

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
    finally:
        http_client.close()


# ── Response Parser ─────────────────────────────────────────────────────────

_VERDICT_RE = re.compile(r"Verdict:\s*(REAL|FAKE|ESCALATE|EXAGGERATED)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"Confidence:\s*(\d+)", re.IGNORECASE)
_VERDICT_MAP = {
    "REAL": Verdict.REAL,
    "FAKE": Verdict.FAKE,
    "ESCALATE": Verdict.ESCALATE,
    "EXAGGERATED": Verdict.EXAGGERATED,
}


def _parse_response(
    raw: str,
    item_id: str,
    latency_s: float,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    verdict = Verdict.REAL
    vm = _VERDICT_RE.search(raw)
    if vm:
        verdict = _VERDICT_MAP.get(vm.group(1).upper(), Verdict.REAL)
    confidence = 0.5
    cm = _CONFIDENCE_RE.search(raw)
    if cm:
        confidence = int(cm.group(1)) / 100.0
    return VerificationResult(
        item_id=item_id,
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        latency_s=latency_s,
        evidence=[],
        metadata=extra_metadata or {},
    )


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _sample_balanced(domain: str) -> Tuple[List[VerificationItem], List[VerificationItem]]:
    rng = np.random.RandomState(SEED)
    cfg = SAMPLE_CONFIG[domain]
    path = FINANCE_CSV if domain == "finance" else HEALTH_CSV
    rows = _load_csv_rows(path)

    real = [r for r in rows if r.get("label", "").strip() == "0"]
    fake = [r for r in rows if r.get("label", "").strip() == "1"]

    n_real = min(cfg["n_real"], len(real))
    n_fake = min(cfg["n_fake"], len(fake))

    rng.shuffle(real)
    rng.shuffle(fake)

    sel_real = real[:n_real]
    sel_fake = fake[:n_fake]

    def to_item(r: Dict[str, str], gt: Verdict) -> VerificationItem:
        # BUGFIX: 'text' column contains multiple concatenated articles;
        # 'title' is the actual headline/claim. Prefer 'headline' > 'title' > 'text'.
        claim = r.get("headline", r.get("title", r.get("text", "")))
        return VerificationItem.create(
            claim_text=claim, ground_truth=gt,
            metadata={"domain": domain, "source": "csv_sample"},
        )

    test_items = [to_item(r, Verdict.FAKE) for r in sel_fake] + [to_item(r, Verdict.REAL) for r in sel_real]
    rng.shuffle(test_items)

    corpus_items = [to_item(r, Verdict.FAKE) for r in fake[n_fake:]] + [to_item(r, Verdict.REAL) for r in real[n_real:]]
    if len(corpus_items) > 1000:
        rng.shuffle(corpus_items)
        corpus_items = corpus_items[:1000]

    print(f"  [{domain}] Sampled: test={len(test_items)} "
          f"(R={n_real} F={n_fake}), corpus={len(corpus_items)} "
          f"(avail: R={len(real)} F={len(fake)})")
    return test_items, corpus_items


# ══════════════════════════════════════════════════════════════════════════
#  PARALLEL EXECUTION HELPER
# ══════════════════════════════════════════════════════════════════════════


def _run_parallel(callables: List[Callable], desc: str = "") -> List[Any]:
    """Run a list of callables in parallel and return results in order.

    Args:
        callables: List of zero-argument callables.
        desc: Optional description for logging.

    Returns:
        List of return values (or None on exception).
    """
    n = len(callables)
    results = [None] * n
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, n)) as ex:
        fut_to_idx = {ex.submit(fn): i for i, fn in enumerate(callables)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = None
    return results


# ══════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════

SS_SYSTEM = (
    "You are an Information Authenticity Verifier. Determine whether a given claim "
    "is authentic (REAL) or contains misinformation (FAKE).\n\n"
    "Analyze the claim for:\n"
    "1. Internal consistency\n"
    "2. Plausibility\n"
    "3. Specificity\n"
    "4. Temporal consistency\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [contradiction|implausibility|inconsistency|none]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_NEUTRAL = (
    "You are a Neutral Evidence Verifier. Examine the following claim for evidence "
    "patterns without bias.\n\n"
    "Analyze systematically:\n"
    "1. Internal consistency\n"
    "2. Plausibility\n"
    "3. Specificity\n\n"
    "Weigh supporting and contradicting evidence equally.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_SKEPTIC = (
    "You are a Skeptical Misinformation Detector. Scrutinize the claim for red flags.\n\n"
    "Look for:\n"
    "1. Logical contradictions\n"
    "2. Exaggerated magnitudes\n"
    "3. Temporal inconsistencies\n"
    "4. Hallmarks of misinformation\n\n"
    "Only flag FAKE when you find CONCRETE evidence.\n"
    "If uncertain, report ESCALATE.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_CALIBRATED = (
    "You are a Base-Rate-Calibrated Verifier. Most news is genuine.\n\n"
    "Before flagging FAKE, ensure the evidence against authenticity is clear. "
    "The cost of false positives is significant.\n\n"
    "Consider:\n"
    "1. Is there clear evidence this is false, or merely surprising?\n"
    "2. Could a reasonable person accept this claim?\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_PROMPTS = [
    ("neutral", VOTER_NEUTRAL),
    ("skeptical", VOTER_SKEPTIC),
    ("calibrated", VOTER_CALIBRATED),
]

# MoA+RAG prompts
MOA_BELIEVER_RAG = (
    "You are Agent 1 (The Believer). Argue the claim is REAL.\n\n"
    "You have access to RETRIEVED EVIDENCE from a knowledge corpus. Use it to build your case.\n\n"
    "Build the strongest case for authenticity:\n"
    "1. Does any evidence support the claim?\n"
    "2. Source credibility\n"
    "3. Factual alignment\n"
    "4. Plausibility\n\n"
    "Output:\n"
    "Verdict: REAL or UNCERTAIN\n"
    "Evidence: <bulleted list>\n"
    "Confidence: <0-100>"
)

MOA_SKEPTIC_RAG = (
    "You are Agent 2 (The Skeptic). Argue the claim is FAKE.\n\n"
    "You have access to RETRIEVED EVIDENCE. Scrutinize it for contradictions.\n\n"
    "Build the strongest case against authenticity:\n"
    "1. Does any evidence contradict the claim?\n"
    "2. Implausibility\n"
    "3. Inconsistencies\n"
    "4. Red flags\n\n"
    "Output:\n"
    "Verdict: FAKE or UNCERTAIN\n"
    "Evidence: <bulleted list>\n"
    "Confidence: <0-100>"
)

MOA_RISK_OFFICER_RAG = (
    "You are Agent 3 (The Risk Officer). Evaluate both sides and deliver a final verdict.\n\n"
    "You have analyses from:\n"
    "- The Believer (argues REAL)\n"
    "- The Skeptic (argues FAKE)\n"
    "Both have access to retrieved evidence.\n\n"
    "Weigh evidence QUALITY, not just presence.\n"
    "Default to REAL if Believer has stronger evidence.\n"
    "Default to FAKE if Skeptic has clear contradictions.\n"
    "Default to ESCALATE if evidence is balanced.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_NEUTRAL_RAG = (
    "You are a Neutral Evidence Verifier with access to RETRIEVED EVIDENCE.\n\n"
    "Analyze:\n"
    "1. Retrieved evidence: does it support or contradict?\n"
    "2. Internal consistency\n"
    "3. Plausibility\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_SKEPTIC_RAG = (
    "You are a Skeptical Misinformation Detector with access to RETRIEVED EVIDENCE.\n\n"
    "Scrutinize if the evidence contradicts the claim.\n"
    "Look for:\n"
    "1. Does evidence directly contradict the claim?\n"
    "2. Logical contradictions\n"
    "3. Exaggerated magnitudes\n"
    "4. Hallmarks of misinformation\n\n"
    "Only flag FAKE with concrete evidence.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_CALIBRATED_RAG = (
    "You are a Base-Rate-Calibrated Verifier with access to RETRIEVED EVIDENCE.\n\n"
    "Most news is genuine. Compare the claim against evidence.\n"
    "Consider:\n"
    "1. Does the evidence support or contradict?\n"
    "2. Is this clearly false, or merely surprising?\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

VOTER_RAG_PROMPTS = [
    ("neutral", VOTER_NEUTRAL_RAG),
    ("skeptical", VOTER_SKEPTIC_RAG),
    ("calibrated", VOTER_CALIBRATED_RAG),
]


# ══════════════════════════════════════════════════════════════════════════
#  TF-IDF RETRIEVER
# ══════════════════════════════════════════════════════════════════════════


def _build_retriever(corpus: List[VerificationItem]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [it.claim_text for it in corpus]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(texts)
    return vec, tfidf, texts


def _retrieve(claim: str, vec, tfidf, texts, top_k=3) -> str:
    from sklearn.metrics.pairwise import cosine_similarity

    qv = vec.transform([claim])
    sims = cosine_similarity(qv, tfidf).flatten()
    parts = []
    for idx in np.argsort(sims)[::-1][:top_k]:
        if sims[idx] > 0:
            parts.append(f"[Doc {idx}] (rel={sims[idx]:.3f})\n{texts[idx]}\n")
    return "".join(parts)[:2000]


# ══════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE RUNNERS  (all use _run_parallel for concurrency)
# ══════════════════════════════════════════════════════════════════════════


# ── 1. Single-Shot ────────────────────────────────────────────────────────

def run_single_shot(items: List[VerificationItem]) -> List[VerificationResult]:
    def make_call(item: VerificationItem) -> VerificationResult:
        up = f"Claim to verify:\n{item.claim_text}\n\nIs this claim REAL or FAKE?"
        raw, lat = _llm_call(SS_SYSTEM, up)
        return _parse_response(raw, item.id, lat, {"architecture": "single_shot"})

    callables = [lambda it=item: make_call(it) for item in items]
    raw = _run_parallel(callables, "SS")
    return [r if r else VerificationResult(
        item_id=it.id, verdict=Verdict.REAL, confidence=0.5, latency_s=0,
        metadata={"architecture": "single_shot", "error": "parallel_failure"},
    ) for r, it in zip(raw, items)]


# ── 2. Voting N=3 ─────────────────────────────────────────────────────────

def run_voting_n3(items: List[VerificationItem]) -> List[VerificationResult]:
    """All voters for all items run in parallel. Track wall time for latency."""
    import time as _time
    batch_start = _time.time()

    callables = []
    for item in items:
        up = f"Claim to verify:\n{item.claim_text}\n\nIs this claim REAL or FAKE?"
        for role_name, sys_prompt in VOTER_PROMPTS:
            def _vote(it=item, sp=sys_prompt, rn=role_name, u=up):
                txt, _ = _llm_call(sp, u)
                return _parse_response(txt, it.id, 0,
                    {"voter_role": rn, "architecture": "voting_n3"})
            callables.append(_vote)

    raw_results = _run_parallel(callables, "Voting")
    batch_elapsed = _time.time() - batch_start

    # Group by item
    item_votes: Dict[str, List[VerificationResult]] = {}
    for r in raw_results:
        if r is not None:
            item_votes.setdefault(r.item_id, []).append(r)

    results = []
    for item in items:
        votes = item_votes.get(item.id, [])
        if not votes:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=batch_elapsed,
                metadata={"architecture": "voting_n3", "error": "no_voters"},
            ))
            continue

        counter = Counter(v.verdict for v in votes)
        mc = counter.most_common(1)
        if mc and mc[0][1] >= 2:
            fv = mc[0][0]
        else:
            fv = Verdict.ESCALATE

        avg_conf = float(np.mean([v.confidence for v in votes]))
        results.append(VerificationResult(
            item_id=item.id, verdict=fv, confidence=avg_conf, latency_s=batch_elapsed,
            evidence=[f"Voting: {dict(counter)}"],
            metadata={
                "architecture": "voting_n3",
                "n_voters": len(votes),
                "verdict_distribution": {k.name: v for k, v in counter.items()},
            },
        ))
    return results


# ── 3. MoA + RAG ─────────────────────────────────────────────────────────

def run_moa_rag(items: List[VerificationItem], corpus: List[VerificationItem]) -> List[VerificationResult]:
    import time as _time
    batch_start = _time.time()

    vec, tfidf, texts = _build_retriever(corpus)

    # Phase 1: Retrieve evidence + run Believer + Skeptic for all items
    evidence_map = {}
    phase1_callables = []
    for item in items:
        evidence = _retrieve(item.claim_text, vec, tfidf, texts)
        evidence_map[item.id] = evidence
        ep = f"\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n" if evidence else ""
        up = f"Claim to verify:\n{item.claim_text}{ep}"

        def _bel(u=up, iid=item.id):
            txt, _ = _llm_call(MOA_BELIEVER_RAG, u)
            return ("believer", iid, txt)

        def _ske(u=up, iid=item.id):
            txt, _ = _llm_call(MOA_SKEPTIC_RAG, u)
            return ("skeptic", iid, txt)

        phase1_callables.append(_bel)
        phase1_callables.append(_ske)

    raw_p1 = _run_parallel(phase1_callables, "MoA+RAG P1")

    # Organize phase 1 results
    believer_out = {}
    skeptic_out = {}
    for result in raw_p1:
        if result is None:
            continue
        role, iid, text = result
        if role == "believer":
            believer_out[iid] = text
        else:
            skeptic_out[iid] = text

    # Phase 2: Risk Officer for all items
    phase2_callables = []
    for item in items:
        evidence = evidence_map.get(item.id, "")
        bel = believer_out.get(item.id, "Error: believer failed")
        skt = skeptic_out.get(item.id, "Error: skeptic failed")
        ro_ctx = (
            f"Claim to verify:\n{item.claim_text}\n\n"
            f"── Retrieved Evidence ──\n{evidence}\n────────────────────\n"
            f"── Believer's Analysis ──\n{bel}\n\n"
            f"── Skeptic's Analysis ──\n{skt}\n"
        )
        def _ro(iid=item.id, ctx=ro_ctx):
            txt, _ = _llm_call(MOA_RISK_OFFICER_RAG, ctx)
            return (iid, _parse_response(txt, iid, 0, {"architecture": "moa_rag"}))
        phase2_callables.append(_ro)

    raw_p2 = _run_parallel(phase2_callables, "MoA+RAG P2")
    batch_elapsed = _time.time() - batch_start

    results = []
    for result in raw_p2:
        if result is None:
            continue
        _, vr = result
        vr.latency_s = batch_elapsed
        results.append(vr)

    return results


# ── 4. Voting N=3 + RAG ──────────────────────────────────────────────────

def run_voting_rag(items: List[VerificationItem], corpus: List[VerificationItem]) -> List[VerificationResult]:
    import time as _time
    batch_start = _time.time()

    vec, tfidf, texts = _build_retriever(corpus)

    # Retrieve evidence for each item (parallel)
    retrieve_calls = [
        lambda it=item: _retrieve(it.claim_text, vec, tfidf, texts)
        for item in items
    ]
    evidence_list = _run_parallel(retrieve_calls, "Retrieval")

    # All voters for all items with evidence
    callables = []
    for item, evidence in zip(items, evidence_list):
        evidence = evidence or ""
        ep = f"\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n" if evidence else ""
        up = f"Claim to verify:\n{item.claim_text}{ep}\nIs this claim REAL or FAKE?"

        for role_name, sys_prompt in VOTER_RAG_PROMPTS:
            def _vote(it=item, sp=sys_prompt, rn=role_name, u=up):
                txt, _ = _llm_call(sp, u)
                return _parse_response(txt, it.id, 0,
                    {"voter_role": rn, "architecture": "voting_n3_rag"})
            callables.append(_vote)

    raw_results = _run_parallel(callables, "Voting+RAG")
    batch_elapsed = _time.time() - batch_start

    # Group by item
    item_votes: Dict[str, List[VerificationResult]] = {}
    for r in raw_results:
        if r is not None:
            item_votes.setdefault(r.item_id, []).append(r)

    results = []
    for item in items:
        votes = item_votes.get(item.id, [])
        if not votes:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=batch_elapsed,
                metadata={"architecture": "voting_n3_rag", "error": "no_voters"},
            ))
            continue

        vf = sum(1 for v in votes if v.verdict == Verdict.FAKE)
        vr = sum(1 for v in votes if v.verdict == Verdict.REAL)
        ve = sum(1 for v in votes if v.verdict == Verdict.ESCALATE)
        avg_conf = float(np.mean([v.confidence for v in votes]))

        if vf >= 2:
            verdict = Verdict.FAKE
        elif vr >= 2:
            verdict = Verdict.REAL
        else:
            verdict = Verdict.ESCALATE

        results.append(VerificationResult(
            item_id=item.id, verdict=verdict, confidence=avg_conf, latency_s=batch_elapsed,
            evidence=[f"voting_rag: FAKE={vf} REAL={vr} ESC={ve}"],
            metadata={
                "architecture": "voting_n3_rag",
                "votes": {"FAKE": vf, "REAL": vr, "ESCALATE": ve},
            },
        ))
    return results


# ══════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════


def _latency_stats(latencies):
    if not latencies:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
    arr = sorted(latencies)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(min(arr)),
        "max": float(max(arr)),
    }


def _bootstrap_ci(items, results, metric_fn, n_iter=1000, alpha=0.05):
    truths = [it.ground_truth or Verdict.REAL for it in items]
    paired = [(r, t) for r, t in zip(results, truths)]
    if len(paired) < 2:
        return (0.0, 0.0)
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n_iter):
        idx = rng.choice(len(paired), len(paired), replace=True)
        sr = [paired[i][0] for i in idx]
        st = [paired[i][1] for i in idx]
        try:
            vals.append(metric_fn(sr, st))
        except Exception:
            continue
    if len(vals) < 2:
        return (0.0, 0.0)
    return (float(np.percentile(vals, alpha / 2 * 100)),
            float(np.percentile(vals, (1 - alpha / 2) * 100)))


def _f1_metric(r, t):
    cm = compute_confusion_matrix(r, t)
    p = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    r_ = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0
    return 2 * p * r_ / (p + r_) if (p + r_) else 0.0


def _prec_metric(r, t):
    cm = compute_confusion_matrix(r, t)
    return cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0


def _rec_metric(r, t):
    cm = compute_confusion_matrix(r, t)
    return cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0


def compute_ppv_curve(sens, fpr, base_rates):
    return [
        {"base_rate": br, "ppv": round((sens * br) / (sens * br + fpr * (1 - br)), 6)
         if (sens * br + fpr * (1 - br)) > 0 else 0.0}
        for br in base_rates
    ]


def compute_expected_cost(fpr, fnr, base_rates, cfp, cfn):
    return [
        {"base_rate": br, "expected_cost": round((1 - br) * fpr * cfp + br * fnr * cfn, 6)}
        for br in base_rates
    ]


def compute_routing(arch_metrics):
    routing = {}
    for budget_label, budget_s in [("<5s", 5), ("<15s", 15), ("<30s", 30)]:
        feasible = [
            {"architecture": an, "f1": d.get("f1", 0), "latency_s": d.get("latency", {}).get("mean", 999)}
            for an, d in arch_metrics.items()
            if d.get("latency", {}).get("mean", 999) < budget_s
        ]
        feasible.sort(key=lambda x: -x["f1"])
        routing[budget_label] = {
            "feasible_count": len(feasible),
            "recommended": feasible[0]["architecture"] if feasible else None,
            "recommended_f1": feasible[0]["f1"] if feasible else None,
            "alternatives": feasible[1:],
        }
    return routing


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════


def run_architecture(arch_name, items, corpus, runner_fn, api_per_item, output_dir):
    domain = items[0].metadata.get("domain", "unknown") if items else "unknown"
    n_fake = sum(1 for it in items if it.ground_truth == Verdict.FAKE)
    n_real = sum(1 for it in items if it.ground_truth == Verdict.REAL)

    print(f"\n  [{domain}] Running {arch_name} on {len(items)} items "
          f"(R={n_real} F={n_fake})...")
    sys.stdout.flush()

    start_t = time.time()
    try:
        raw_results = runner_fn(items, corpus) if corpus is not None else runner_fn(items)
    except Exception as e:
        traceback.print_exc()
        return {"skipped": True, "error": str(e)}

    elapsed = time.time() - start_t
    latencies = [r.latency_s for r in raw_results]

    # Serialize
    result_list = [{
        "item_id": r.item_id, "verdict": r.verdict.name,
        "confidence": r.confidence, "latency_s": r.latency_s,
        "evidence": r.evidence[:5], "metadata": r.metadata,
    } for r in raw_results]

    truths = [it.ground_truth or Verdict.REAL for it in items]
    section = {"n": len(items), "n_fake": n_fake, "n_real": n_real,
               "class_balance": f"{n_fake}F/{n_real}R"}

    try:
        cm = compute_confusion_matrix(raw_results, truths)
        from src.metrics import classification_metrics
        m = classification_metrics(raw_results, truths)
        section["metrics"] = {
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "fpr": m.fpr, "fnr": m.fnr, "accuracy": m.accuracy,
            "n_total": m.n_total,
            "confusion": {"tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn},
        }
        section["verdict_distribution"] = dict(Counter(r.verdict.name for r in raw_results))
        n_esc = sum(1 for r in raw_results if r.verdict == Verdict.ESCALATE)
        section["escalate_rate"] = n_esc / len(raw_results) if raw_results else 0.0

        correct = [r.verdict == t for r, t in zip(raw_results, truths)]
        ece_val, _, _, _ = compute_ece([r.confidence for r in raw_results], correct, n_bins=5)
        section["ece"] = ece_val

        section["ppv"] = compute_ppv_curve(m.recall, m.fpr, PPV_BASE_RATES)
        section["expected_cost"] = {
            f"FP{cfp}_FN{cfn}": compute_expected_cost(m.fpr, m.fnr, PPV_BASE_RATES, cfp, cfn)
            for cfp, cfn in COST_RATIOS
        }
        section["confidence_intervals"] = {
            "f1_95": list(_bootstrap_ci(items, raw_results, _f1_metric)),
            "precision_95": list(_bootstrap_ci(items, raw_results, _prec_metric)),
            "recall_95": list(_bootstrap_ci(items, raw_results, _rec_metric)),
        }
    except Exception as e:
        section["metrics_error"] = str(e)

    section["latency"] = _latency_stats(latencies)
    section["total_api_calls"] = len(items) * api_per_item
    section["elapsed_s"] = round(elapsed, 1)

    # Save raw
    raw_path = os.path.join(output_dir, "raw_outputs", f"{domain}_{arch_name}.json")
    with open(raw_path, "w") as f:
        json.dump({
            "architecture": arch_name, "domain": domain, "model": MODEL,
            "items": [{"id": it.id, "claim": it.claim_text[:200],
                       "ground_truth": it.ground_truth.name if it.ground_truth else None}
                      for it in items],
            "results": result_list,
        }, f, indent=2, default=str)
    section["raw_output_path"] = raw_path

    m = section.get("metrics", {})
    ci = section.get("confidence_intervals", {})
    ci_str = f"  F1 CI=({ci.get('f1_95', [0,0])[0]:.3f}, {ci.get('f1_95', [0,0])[1]:.3f})" if ci.get("f1_95") else ""
    print(f"    F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  "
          f"R={m.get('recall', 0):.4f}  FPR={m.get('fpr', 0):.4f}  "
          f"FNR={m.get('fnr', 0):.4f}  Acc={m.get('accuracy', 0):.4f}  "
          f"ESC={section.get('escalate_rate', 0):.2%}  ECE={section.get('ece', 0):.4f}  "
          f"Lat={section.get('latency', {}).get('mean', 0):.1f}s  "
          f"({section.get('elapsed_s', 0):.0f}s wall){ci_str}")
    sys.stdout.flush()
    return section


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    global_start = time.time()

    print("=" * 70)
    print("FINAL VALIDATION SPRINT (CONCURRENT)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Concurrency: {MAX_CONCURRENCY}")
    print(f"Output: {OUTPUT_DIR}/")
    sys.stdout.flush()

    # Verify API
    print("\nVerifying DeepSeek API connectivity...")
    sys.stdout.flush()
    try:
        raw, lat = _llm_call("You are a test assistant.", "Reply with OK.")
        print(f"  API OK — {raw[:30]} ({lat:.1f}s)")
    except Exception as e:
        print(f"  API ERROR: {e}")
        return
    sys.stdout.flush()

    # Load data
    all_test: Dict[str, List[VerificationItem]] = {}
    all_corpus: Dict[str, List[VerificationItem]] = {}
    print("\n" + "=" * 70)
    print("LOADING AND SAMPLING DATA")
    print("=" * 70)
    for domain in DOMAINS:
        test, corpus = _sample_balanced(domain)
        all_test[domain] = test
        all_corpus[domain] = corpus
    sys.stdout.flush()

    # Architectures
    architectures = [
        ("single_shot", run_single_shot, None, 1),
        ("voting_n3", run_voting_n3, None, 3),
        ("moa_rag", run_moa_rag, all_corpus, 4),
        ("voting_n3_rag", run_voting_rag, all_corpus, 3),
    ]

    report = {
        "metadata": {
            "phase": "final_validation",
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "concurrency": MAX_CONCURRENCY,
            "domains": DOMAINS,
            "sample_config": SAMPLE_CONFIG,
            "pilot_note": "Healthcare N=40 (only 20 REAL available). Finance N=50 balanced 25/25.",
        },
        "domains": {},
        "cross_domain": {},
        "routing": {},
    }

    total_api_calls = 0
    all_sections: Dict[str, Dict[str, Any]] = {}

    for domain in DOMAINS:
        print(f"\n{'=' * 70}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'=' * 70}")
        sys.stdout.flush()

        domain_sections = {}
        for arch_name, runner_fn, corpus_source, api_per_item in architectures:
            corpus = corpus_source.get(domain) if corpus_source else None
            section = run_architecture(
                arch_name, all_test[domain], corpus,
                runner_fn, api_per_item, OUTPUT_DIR,
            )
            domain_sections[arch_name] = section
            total_api_calls += section.get("total_api_calls", 0) if isinstance(section.get("total_api_calls"), int) else api_per_item * len(all_test[domain])

        all_sections[domain] = domain_sections

    # Assemble report
    total_est_cost = total_api_calls * COST_PER_CALL
    report["metadata"]["total_api_calls"] = total_api_calls
    report["metadata"]["estimated_cost_usd"] = round(total_est_cost, 4)

    for domain in DOMAINS:
        report["domains"][domain] = all_sections[domain]
        # Per-domain routing
        arch_metrics = {
            an: s.get("metrics", {})
            for an, s in all_sections[domain].items()
            if not s.get("skipped") and s.get("metrics")
        }
        # Add latency for routing
        for an in arch_metrics:
            arch_metrics[an]["latency"] = all_sections[domain][an].get("latency", {})
        report["routing"][domain] = compute_routing(arch_metrics)

    # Cross-domain
    for arch_name, _, _, _ in architectures:
        f1s = {}
        for d in DOMAINS:
            m = all_sections[d].get(arch_name, {}).get("metrics", {})
            if m.get("f1") is not None:
                f1s[d] = m["f1"]
        if f1s:
            vals = list(f1s.values())
            report["cross_domain"][arch_name] = {
                "per_domain_f1": f1s,
                "mean_f1": float(np.mean(vals)),
                "std_f1": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }

    # Cross-domain routing
    cd_metrics = {}
    for arch_name in [a[0] for a in architectures]:
        m = report["cross_domain"].get(arch_name, {})
        if m:
            cd_metrics[arch_name] = {
                "f1": m.get("mean_f1", 0),
                "latency": {"mean": np.mean([
                    all_sections[d][arch_name].get("latency", {}).get("mean", 0)
                    for d in DOMAINS
                ])},
            }
    report["routing"]["cross_domain"] = compute_routing(cd_metrics)

    # Compute total runtime so far
    running = time.time() - global_start

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total runtime: {running:.0f}s ({running/60:.1f} min)")
    print(f"Total API calls: {total_api_calls}")
    print(f"Estimated cost: ${total_est_cost:.4f}")

    for domain in DOMAINS:
        print(f"\n{'─' * 70}")
        print(f"{domain.upper()} (N={all_sections[domain].get('single_shot', {}).get('n', 0)})")
        print(f"{'─' * 70}")
        h = f"{'Arch':20s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Lat':7s} {'Wall':7s}"
        print(h)
        print("-" * len(h))
        for arch_name, _, _, _ in architectures:
            s = all_sections[domain].get(arch_name, {})
            if s.get("skipped"):
                print(f"{arch_name:20s} SKIPPED")
                continue
            m = s.get("metrics", {})
            lt = s.get("latency", {})
            print(f"{arch_name:20s} {m.get('f1',0):.4f}  {m.get('precision',0):.4f}  "
                  f"{m.get('recall',0):.4f}  {m.get('fpr',0):.4f}  {m.get('fnr',0):.4f}  "
                  f"{m.get('accuracy',0):.4f}  {s.get('escalate_rate',0):.2%}  "
                  f"{s.get('ece',0):.4f}  {lt.get('mean',0):.1f}s  {s.get('elapsed_s',0):.0f}s")

    # CIs
    print(f"\n{'─' * 70}")
    print("95% BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'─' * 70}")
    for domain in DOMAINS:
        print(f"\n{domain}:")
        for arch_name, _, _, _ in architectures:
            s = all_sections[domain].get(arch_name, {})
            ci = s.get("confidence_intervals", {})
            f1c = ci.get("f1_95", [])
            pc = ci.get("precision_95", [])
            rc = ci.get("recall_95", [])
            if f1c:
                print(f"  {arch_name:20s} F1=({f1c[0]:.3f},{f1c[1]:.3f})  "
                      f"P=({pc[0]:.3f},{pc[1]:.3f})  R=({rc[0]:.3f},{rc[1]:.3f})")

    # Routing
    print(f"\n{'─' * 70}")
    print("ROUTING RECOMMENDATION")
    print(f"{'─' * 70}")
    for budget in ["<5s", "<15s", "<30s"]:
        print(f"\nLatency budget: {budget}")
        for domain in DOMAINS:
            r = report["routing"][domain].get(budget, {})
            rec = r.get("recommended", "N/A")
            f1v = r.get("recommended_f1", "N/A")
            print(f"  {domain:15s} → {str(rec):20s} (F1={f1v})")

    total_runtime = time.time() - global_start
    print(f"\n{'=' * 70}")
    print(f"VALIDATION COMPLETE ({total_runtime:.0f}s / {total_runtime/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
