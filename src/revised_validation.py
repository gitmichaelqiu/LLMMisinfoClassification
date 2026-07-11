"""Revised verification experiment with canonical architecture definitions.

Implements per the specification:
- Single-Shot: one call, no aggregation, REAL/FAKE/ESCALATE + confidence
- Voting/Self-Consistency: N independent calls, same prompt, temp>0, mechanical majority
- MoA: Believer/Skeptic/Judge deliberation, one round, no majority vote
- RAG: evidence factor (OFF/ON), not a standalone architecture

Stage 1 — Voting-size sensitivity (N=1,3,5,7, RAG OFF)
Stage 2 — 2×3 factorial (Evidence OFF/ON × SS/VotingN*/MoA)
Stage 3 — Final comparison across all 6 cells
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
TEMPERATURE = 0.7        # canonical: same for SS and Voting (N=1 single call ≡ SS)
                          # Per audit: temperature was the only difference between SS and Voting N=1.
                          # With T=0 for SS and T=0.7 for Voting, they produced different outputs.
                          # Fix: use identical T=0.7 — Voting N=1 is a single call at T=0.7,
                          # and SS is also a single call at T=0.7. They are the same configuration.
VOTING_TEMPERATURE = 0.7  # alias kept for clarity; always equals TEMPERATURE
MAX_TOKENS = 512
DOMAINS = ["finance", "healthcare"]
OUTPUT_DIR = "results/revised_validation"
MAX_CONCURRENCY = 400
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "raw_outputs"), exist_ok=True)

PPV_BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
COST_RATIOS = [(1, 1), (1, 5), (1, 10), (1, 25)]
VOTING_N_VALUES = [1, 3, 5, 7]

SAMPLE_CONFIG = {
    "finance": {"n": 50, "n_real": 25, "n_fake": 25},
    "healthcare": {"n": 40, "n_real": 20, "n_fake": 20},
}

COST_PER_CALL = 0.00015
FINANCE_CSV = os.path.join("data", "raw", "finance", "financial_news.csv")
HEALTH_CSV = os.path.join("data", "raw", "health", "health_headlines.csv")


# ── API Call ────────────────────────────────────────────────────────────────


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


# ── Data Loading ────────────────────────────────────────────────────────────


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


# ── Parallel Execution ──────────────────────────────────────────────────────


def _run_parallel(callables: List[Callable], desc: str = "") -> List[Any]:
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
#  PROMPTS  (Canonical definitions)
# ══════════════════════════════════════════════════════════════════════════


# ── Single-Shot prompt (also used as the uniform Voting prompt) ────────

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

# ── MoA prompts (without RAG) ─────────────────────────────────────────

MOA_BELIEVER = (
    "You are Agent 1 (The Believer). Argue the claim is REAL.\n\n"
    "Build the strongest case for authenticity:\n"
    "1. Internal consistency\n"
    "2. Plausibility\n"
    "3. Factual alignment with known information\n"
    "4. Source credibility signals\n\n"
    "Output:\n"
    "Verdict: REAL or UNCERTAIN\n"
    "Confidence: <0-100>\n"
    "Reasoning: <your argument>"
)

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

MOA_SKEPTIC = (
    "You are Agent 2 (The Skeptic). Argue the claim is FAKE.\n\n"
    "Build the strongest case against authenticity:\n"
    "1. Logical contradictions\n"
    "2. Implausibility\n"
    "3. Inconsistencies\n"
    "4. Red flags or hallmarks of misinformation\n\n"
    "Output:\n"
    "Verdict: FAKE or UNCERTAIN\n"
    "Confidence: <0-100>\n"
    "Reasoning: <your argument>"
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

MOA_JUDGE = (
    "You are the Judge. Read both arguments and deliver a final verdict.\n\n"
    "You have analyses from:\n"
    "- The Believer (argues REAL)\n"
    "- The Skeptic (argues FAKE)\n\n"
    "Weigh evidence QUALITY, not just presence.\n"
    "Default to REAL if Believer has stronger evidence.\n"
    "Default to FAKE if Skeptic has clear contradictions.\n"
    "Default to ESCALATE if arguments are balanced or uncertain.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)

MOA_JUDGE_RAG = (
    "You are the Judge. Read both arguments and the retrieved evidence, then deliver a final verdict.\n\n"
    "You have analyses from:\n"
    "- The Believer (argues REAL, with evidence)\n"
    "- The Skeptic (argues FAKE, with evidence)\n"
    "Evidence from the knowledge corpus is available.\n\n"
    "Weigh evidence QUALITY, not just presence.\n"
    "Default to REAL if Believer has stronger evidence.\n"
    "Default to FAKE if Skeptic has clear contradictions.\n"
    "Default to ESCALATE if arguments are balanced or uncertain.\n\n"
    "Output in EXACTLY this format:\n"
    "Verdict: REAL or FAKE or ESCALATE\n"
    "Confidence: <0-100>\n"
    "Flags: [...]\n"
    "Reasoning: <one-sentence rationale>"
)


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
#  ARCHITECTURE RUNNERS  (Canonical implementations)
# ══════════════════════════════════════════════════════════════════════════

# ── 1. Single-Shot ──────────────────────────────────────────────────────

def run_single_shot(items: List[VerificationItem],
                    rag_on: bool = False,
                    corpus: Optional[List[VerificationItem]] = None) -> List[VerificationResult]:
    """Canonical Single-Shot: one verifier call, no aggregation."""

    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = _build_retriever(corpus)

    def make_call(item: VerificationItem) -> VerificationResult:
        evidence = ""
        if rag_on and vec is not None:
            evidence = _retrieve(item.claim_text, vec, tfidf, texts)
        ep = f"\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n" if evidence else ""
        up = f"Claim to verify:\n{item.claim_text}{ep}\n\nIs this claim REAL or FAKE?"
        raw, lat = _llm_call(SS_SYSTEM, up)
        return _parse_response(raw, item.id, lat, {
            "architecture": "single_shot",
            "rag": rag_on,
        })

    callables = [lambda it=item: make_call(it) for item in items]
    raw = _run_parallel(callables, "SS")
    return [r if r else VerificationResult(
        item_id=it.id, verdict=Verdict.REAL, confidence=0.5, latency_s=0,
        metadata={"architecture": "single_shot", "rag": rag_on, "error": "parallel_failure"},
    ) for r, it in zip(raw, items)]


# ── 2. Voting / Self-Consistency ────────────────────────────────────────

def run_voting(items: List[VerificationItem],
               n_voters: int,
               temperature: float = VOTING_TEMPERATURE,
               rag_on: bool = False,
               corpus: Optional[List[VerificationItem]] = None) -> List[VerificationResult]:
    """Canonical Voting / Self-Consistency.

    - N independent calls using the SAME verifier prompt (SS_SYSTEM)
    - Same model and evidence context
    - Calls do not see each other
    - No role specialization, no judge
    - Mechanical aggregation: majority REAL/FAKE, tie/insufficient → ESCALATE
    - Uses fixed non-zero sampling temperature so calls can differ
    """
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = _build_retriever(corpus)

    callables = []
    for item in items:
        evidence = ""
        if rag_on and vec is not None:
            evidence = _retrieve(item.claim_text, vec, tfidf, texts)
        ep = f"\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n" if evidence else ""
        up = f"Claim to verify:\n{item.claim_text}{ep}\n\nIs this claim REAL or FAKE?"

        for v in range(n_voters):
            def _vote(it=item, u=up, voter_idx=v):
                txt, lat = _llm_call(SS_SYSTEM, u, temperature=temperature)
                return _parse_response(txt, it.id, lat, {
                    "voter_idx": voter_idx,
                    "architecture": f"voting_n{n_voters}",
                    "rag": rag_on,
                })
            callables.append(_vote)

    raw = _run_parallel(callables, f"Voting-N{n_voters}")

    # Group by item
    item_results: Dict[str, List[VerificationResult]] = defaultdict(list)
    for r in raw:
        if r is not None:
            item_results[r.item_id].append(r)

    # Mechanical majority aggregation
    threshold = n_voters // 2 + 1  # > N/2

    results = []
    for item in items:
        votes = item_results.get(item.id, [])
        if not votes:
            results.append(VerificationResult(
                item_id=item.id, verdict=Verdict.REAL, confidence=0.5,
                latency_s=0, metadata={"architecture": f"voting_n{n_voters}",
                                       "rag": rag_on, "error": "no_voters"},
            ))
            continue

        counter = Counter(v.verdict for v in votes)
        most_common = counter.most_common(1)

        if most_common and most_common[0][1] >= threshold:
            verdict = most_common[0][0]
        else:
            verdict = Verdict.ESCALATE

        per_voter_latencies = [v.latency_s for v in votes]
        item_lat = max(per_voter_latencies)  # parallel: wall time = slowest voter
        avg_conf = float(np.mean([v.confidence for v in votes]))
        disagreement = 1.0 - (most_common[0][1] / len(votes)) if most_common else 1.0

        results.append(VerificationResult(
            item_id=item.id, verdict=verdict, confidence=avg_conf,
            latency_s=item_lat,
            evidence=[f"Voting-N{n_voters}: {dict(counter)}"],
            metadata={
                "architecture": f"voting_n{n_voters}",
                "rag": rag_on,
                "n_voters": len(votes),
                "threshold": threshold,
                "verdict_distribution": {k.name: v for k, v in counter.items()},
                "voter_latencies": per_voter_latencies,
                "disagreement_rate": disagreement,
            },
        ))
    return results


# ── 3. MoA (Mixture of Agents) ──────────────────────────────────────────

def run_moa(items: List[VerificationItem],
            rag_on: bool = False,
            corpus: Optional[List[VerificationItem]] = None) -> List[VerificationResult]:
    """Canonical MoA.

    - Role-specialized agents: Believer (argues REAL), Skeptic (argues FAKE)
    - Judge reads both agents' arguments
    - Judge outputs REAL/FAKE/ESCALATE + confidence
    - One deliberation round only
    - No mechanical majority vote
    - RAG is an evidence toggle passed to all agents
    """
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = _build_retriever(corpus)

    # Phase 1: Retrieve evidence and run Believer + Skeptic (parallel within item, parallel across items)
    evidence_map: Dict[str, str] = {}
    phase1_callables = []

    for item in items:
        evidence = ""
        if rag_on and vec is not None:
            evidence = _retrieve(item.claim_text, vec, tfidf, texts)
        evidence_map[item.id] = evidence
        ep = f"\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n" if evidence else ""
        up = f"Claim to verify:\n{item.claim_text}{ep}"

        if rag_on:
            believer_prompt = MOA_BELIEVER_RAG
            skeptic_prompt = MOA_SKEPTIC_RAG
        else:
            believer_prompt = MOA_BELIEVER
            skeptic_prompt = MOA_SKEPTIC

        def _bel(u=up, iid=item.id):
            txt, lat = _llm_call(believer_prompt, u)
            return ("believer", iid, txt, lat)

        def _ske(u=up, iid=item.id):
            txt, lat = _llm_call(skeptic_prompt, u)
            return ("skeptic", iid, txt, lat)

        phase1_callables.append(_bel)
        phase1_callables.append(_ske)

    raw_p1 = _run_parallel(phase1_callables, "MoA P1")

    # Organize phase 1 results
    believer_out: Dict[str, Tuple[str, float]] = {}
    skeptic_out: Dict[str, Tuple[str, float]] = {}
    for result in raw_p1:
        if result is None:
            continue
        role, iid, text, lat = result
        if role == "believer":
            believer_out[iid] = (text, lat)
        else:
            skeptic_out[iid] = (text, lat)

    # Phase 2: Judge for all items (parallel across items)
    judge_prompt = MOA_JUDGE_RAG if rag_on else MOA_JUDGE

    phase2_callables = []
    for item in items:
        evidence = evidence_map.get(item.id, "")
        bel_text, bel_lat = believer_out.get(item.id, ("Error: believer failed", 0))
        ske_text, ske_lat = skeptic_out.get(item.id, ("Error: skeptic failed", 0))

        ep = f"\n\n── Retrieved Evidence ──\n{evidence}\n────────────────────\n" if evidence else ""
        judge_ctx = (
            f"Claim to verify:\n{item.claim_text}{ep}\n\n"
            f"── Believer's Analysis ──\n{bel_text}\n\n"
            f"── Skeptic's Analysis ──\n{ske_text}\n"
        )

        # Per-item MoA latency = max(believer, skeptic) + judge
        p1_lat = max(bel_lat, ske_lat)

        def _judge(iid=item.id, ctx=judge_ctx, p1_lat_=p1_lat):
            txt, lat = _llm_call(judge_prompt, ctx)
            vr = _parse_response(txt, iid, lat, {
                "architecture": "moa",
                "rag": rag_on,
            })
            vr.latency_s = p1_lat_ + lat  # total = P1 max + judge
            return (iid, vr)

        phase2_callables.append(_judge)

    raw_p2 = _run_parallel(phase2_callables, "MoA P2")

    results = []
    for result in raw_p2:
        if result is None:
            continue
        _, vr = result
        results.append(vr)

    return results


# ══════════════════════════════════════════════════════════════════════════
#  METRICS & EVALUATION
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


def _bootstrap_diff(results_a, truths_a, results_b, truths_b,
                    metric_fn, n_iter=1000, alpha=0.05):
    """Paired bootstrap CI for difference (metric_a - metric_b)."""
    n = len(truths_a)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        va = metric_fn([results_a[i] for i in idx], [truths_a[i] for i in idx])
        vb = metric_fn([results_b[i] for i in idx], [truths_b[i] for i in idx])
        diffs.append(va - vb)
    diffs = sorted(diffs)
    return (float(np.percentile(diffs, alpha / 2 * 100)),
            float(np.percentile(diffs, (1 - alpha / 2) * 100)))


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


def evaluate_architecture(items: List[VerificationItem],
                          results: List[VerificationResult],
                          arch_name: str,
                          total_api_calls: int) -> Dict[str, Any]:
    """Compute all metrics for a single architecture run."""
    truths = [it.ground_truth or Verdict.REAL for it in items]
    latencies = [r.latency_s for r in results if r.latency_s > 0]

    section = {
        "n": len(items),
        "n_fake": sum(1 for it in items if it.ground_truth == Verdict.FAKE),
        "n_real": sum(1 for it in items if it.ground_truth == Verdict.REAL),
    }

    try:
        cm = compute_confusion_matrix(results, truths)
        from src.metrics import classification_metrics
        m = classification_metrics(results, truths)
        section["metrics"] = {
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "fpr": m.fpr, "fnr": m.fnr, "accuracy": m.accuracy,
            "n_total": m.n_total,
            "confusion": {"tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn},
        }
        section["verdict_distribution"] = dict(Counter(r.verdict.name for r in results))
        n_esc = sum(1 for r in results if r.verdict == Verdict.ESCALATE)
        section["escalate_rate"] = n_esc / len(results) if results else 0.0

        correct = [r.verdict == t for r, t in zip(results, truths)]
        ece_val, _, _, _ = compute_ece([r.confidence for r in results], correct, n_bins=5)
        section["ece"] = ece_val

        # Disagreement rate (for voting)
        disagreement_rates = []
        for r in results:
            meta = r.metadata or {}
            dr = meta.get("disagreement_rate")
            if dr is not None:
                disagreement_rates.append(dr)
        section["mean_disagreement"] = float(np.mean(disagreement_rates)) if disagreement_rates else 0.0

        section["ppv"] = compute_ppv_curve(m.recall, m.fpr, PPV_BASE_RATES)
        section["expected_cost"] = {
            f"FP{cfp}_FN{cfn}": compute_expected_cost(m.fpr, m.fnr, PPV_BASE_RATES, cfp, cfn)
            for cfp, cfn in COST_RATIOS
        }
        section["confidence_intervals"] = {
            "f1_95": list(_bootstrap_ci(items, results, _f1_metric)),
            "precision_95": list(_bootstrap_ci(items, results, _prec_metric)),
            "recall_95": list(_bootstrap_ci(items, results, _rec_metric)),
        }
    except Exception as e:
        section["metrics_error"] = str(e)

    section["latency"] = _latency_stats(latencies)
    section["total_api_calls"] = total_api_calls
    return section


def save_raw_output(domain: str, arch_name: str, items: List[VerificationItem],
                    results: List[VerificationResult], output_dir: str):
    """Save raw per-item predictions to JSON."""
    raw_path = os.path.join(output_dir, "raw_outputs", f"{domain}_{arch_name}.json")
    with open(raw_path, "w") as f:
        json.dump({
            "architecture": arch_name, "domain": domain, "model": MODEL,
            "items": [{"id": it.id, "claim": it.claim_text[:200],
                       "ground_truth": it.ground_truth.name if it.ground_truth else None}
                      for it in items],
            "results": [{
                "item_id": r.item_id, "verdict": r.verdict.name,
                "confidence": r.confidence, "latency_s": r.latency_s,
                "evidence": r.evidence[:5], "metadata": r.metadata,
            } for r in results],
        }, f, indent=2, default=str)
    return raw_path


def print_metrics(section: Dict[str, Any], label: str = ""):
    """Print a one-line metrics summary."""
    m = section.get("metrics", {})
    ci = section.get("confidence_intervals", {})
    ci_str = (f"  F1 CI=({ci.get('f1_95', [0, 0])[0]:.3f}, "
              f"{ci.get('f1_95', [0, 0])[1]:.3f})") if ci.get("f1_95") else ""
    print(f"    {label:20s} F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  "
          f"R={m.get('recall', 0):.4f}  FPR={m.get('fpr', 0):.4f}  "
          f"FNR={m.get('fnr', 0):.4f}  Acc={m.get('accuracy', 0):.4f}  "
          f"ESC={section.get('escalate_rate', 0):.2%}  "
          f"ECE={section.get('ece', 0):.4f}  "
          f"Lat={section.get('latency', {}).get('mean', 0):.1f}s{ci_str}")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 1: VOTING-SIZE SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════


def run_voting_sensitivity(items: List[VerificationItem],
                           domain: str,
                           output_dir: str) -> Dict[int, Dict[str, Any]]:
    """Test voting at N=1,3,5,7 with RAG OFF. Returns {N: section} dict."""
    print(f"\n  {'─' * 60}")
    print(f"  STAGE 1: Voting-size sensitivity — {domain}")
    print(f"  {'─' * 60}")
    print(f"  Temperature: {VOTING_TEMPERATURE}")
    print(f"  Prompt: SS_SYSTEM (same for all voters)")
    print(f"  Aggregation: majority (threshold=N//2+1), else ESCALATE")
    print(f"  {'─' * 60}")

    results: Dict[int, Dict[str, Any]] = {}

    for n in VOTING_N_VALUES:
        print(f"\n  Voting N={n}...")
        sys.stdout.flush()

        start_t = time.time()
        raw_results = run_voting(items, n, temperature=VOTING_TEMPERATURE,
                                 rag_on=False, corpus=None)
        elapsed = time.time() - start_t

        api_calls = len(items) * n
        section = evaluate_architecture(items, raw_results, f"voting_n{n}", api_calls)
        section["elapsed_s"] = round(elapsed, 1)

        # Save raw
        save_raw_output(domain, f"voting_n{n}", items, raw_results, output_dir)

        # Marginal gain vs N=1 (or previous N)
        prev_n = max(v for v in VOTING_N_VALUES if v < n) if n > 1 else None
        if prev_n and prev_n in results:
            f1_curr = section["metrics"]["f1"]
            f1_prev = results[prev_n]["metrics"]["f1"]
            section["marginal_gain"] = f1_curr - f1_prev

            # Paired bootstrap of difference vs N=1
            truths = [it.ground_truth or Verdict.REAL for it in items]
            prev_raw_results = results[prev_n].get("_raw_results", [])
            if prev_raw_results:
                ci = _bootstrap_diff(raw_results, truths, prev_raw_results, truths, _f1_metric)
                section["bootstrap_diff_vs_n1"] = list(ci)
        else:
            section["marginal_gain"] = 0.0
            section["bootstrap_diff_vs_n1"] = []

        # Bootstrap vs N=1 specifically
        truths = [it.ground_truth or Verdict.REAL for it in items]
        if n > 1 and 1 in results:
            n1_raw = results[1].get("_raw_results", [])
            if n1_raw:
                ci = _bootstrap_diff(raw_results, truths, n1_raw, truths, _f1_metric)
                section["bootstrap_diff_vs_n1"] = list(ci)

        section["_raw_results"] = raw_results  # store for cross-comparisons
        results[n] = section
        print_metrics(section, f"N={n}")

    # Selection
    print(f"\n  {'─' * 40}")
    print(f"  Voting-N selection for {domain}")
    print(f"  {'─' * 40}")

    best_n = max(results, key=lambda n: results[n].get("metrics", {}).get("f1", 0))
    best_f1 = results[best_n].get("metrics", {}).get("f1", 0)
    print(f"  Best-performing N:     {best_n} (F1={best_f1:.4f})")

    # Smallest N within 0.02 F1 of best
    candidates = [n for n in VOTING_N_VALUES
                  if results[n].get("metrics", {}).get("f1", 0) >= best_f1 - 0.02]
    smallest_within_002 = min(candidates) if candidates else best_n
    print(f"  Smallest N within 0.02: {smallest_within_002} "
          f"(F1={results[smallest_within_002].get('metrics', {}).get('f1', 0):.4f})")

    # Best latency-adjusted N
    def f1_per_sec(n_val):
        f1 = results[n_val].get("metrics", {}).get("f1", 0)
        lat = results[n_val].get("latency", {}).get("mean", 1)
        return f1 / lat if lat > 0 else 0

    best_lat_adj = max(VOTING_N_VALUES, key=f1_per_sec)
    print(f"  Best latency-adjusted: {best_lat_adj} "
          f"(F1/s={f1_per_sec(best_lat_adj):.4f})")

    results["_selection"] = {
        "best_n": best_n,
        "best_f1": best_f1,
        "smallest_within_002": smallest_within_002,
        "smallest_within_002_f1": results[smallest_within_002].get("metrics", {}).get("f1", 0),
        "best_latency_adjusted_n": best_lat_adj,
        "best_latency_adjusted_score": f1_per_sec(best_lat_adj),
    }

    return results


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 2+3: 2×3 FACTORIAL DESIGN
# ══════════════════════════════════════════════════════════════════════════


def run_factorial_comparison(items: List[VerificationItem],
                              corpus: List[VerificationItem],
                              domain: str,
                              voting_n: int,
                              output_dir: str) -> Dict[str, Any]:
    """Run the full 2×3 factorial comparison.

    Factors:
    - Evidence: OFF / ON
    - Architecture: Single-Shot / Voting N* / MoA

    Returns {cell_label: section} dict.
    """
    print(f"\n  {'═' * 60}")
    print(f"  STAGE 2+3: 2×3 Factorial Comparison — {domain}")
    print(f"  {'═' * 60}")
    print(f"  Voting N* = {voting_n} (selected from Stage 1)")
    print(f"  {'═' * 60}")

    runs = [
        ("single_shot_rag_off", run_single_shot, False, len(items) * 1),
        ("single_shot_rag_on",  run_single_shot, True,  len(items) * 1),
        (f"voting_n{voting_n}_rag_off", lambda its, c: run_voting(its, voting_n, rag_on=False), False, len(items) * voting_n),
        (f"voting_n{voting_n}_rag_on",  lambda its, c: run_voting(its, voting_n, rag_on=True), True, len(items) * voting_n),
        ("moa_rag_off", run_moa, False, len(items) * 3),  # Believer + Skeptic + Judge
        ("moa_rag_on",  run_moa, True,  len(items) * 3),
    ]

    all_sections: Dict[str, Any] = {}
    total_api = 0

    for arch_name, runner_fn, rag_flag, api_count in runs:
        print(f"\n  Running {arch_name}...")
        sys.stdout.flush()

        start_t = time.time()
        try:
            raw_results = runner_fn(items, corpus)
        except Exception as e:
            traceback.print_exc()
            print(f"    ERROR: {e}")
            all_sections[arch_name] = {"skipped": True, "error": str(e)}
            continue

        elapsed = time.time() - start_t
        section = evaluate_architecture(items, raw_results, arch_name, api_count)
        section["elapsed_s"] = round(elapsed, 1)
        section["_raw_results"] = raw_results

        save_raw_output(domain, arch_name, items, raw_results, output_dir)
        all_sections[arch_name] = section
        total_api += api_count

        print_metrics(section, arch_name)

    all_sections["_total_api_calls"] = total_api
    return all_sections


# ══════════════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════


def print_voting_table(voting_results: Dict[int, Dict[str, Any]], domain: str):
    """Print formatted Stage 1 voting sensitivity table."""
    print(f"\n{'=' * 70}")
    print(f"STAGE 1: VOTING SENSITIVITY — {domain.upper()}")
    print(f"{'=' * 70}")

    h = f"{'N':6s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Disagr':7s} {'Lat':7s} {'Wall':7s}"
    print(h)
    print("-" * len(h))

    for n in VOTING_N_VALUES:
        s = voting_results.get(n, {})
        if s.get("skipped"):
            print(f"{n:6d} SKIPPED")
            continue
        m = s.get("metrics", {})
        lt = s.get("latency", {})
        dr = s.get("mean_disagreement", 0)
        print(f"{n:6d} {m.get('f1', 0):.4f}  {m.get('precision', 0):.4f}  "
              f"{m.get('recall', 0):.4f}  {m.get('fpr', 0):.4f}  "
              f"{m.get('fnr', 0):.4f}  {m.get('accuracy', 0):.4f}  "
              f"{s.get('escalate_rate', 0):.2%}  {s.get('ece', 0):.4f}  "
              f"{dr:.4f}  {lt.get('mean', 0):.1f}s  {s.get('elapsed_s', 0):.0f}s")

    # Marginal gains
    print(f"\n  Marginal gains:")
    for n in VOTING_N_VALUES:
        mg = voting_results.get(n, {}).get("marginal_gain", 0)
        prev = max(v for v in VOTING_N_VALUES if v < n) if n > VOTING_N_VALUES[0] else None
        from_label = f"N={prev}" if prev else "N/A"
        print(f"    N={n} (Δ from {from_label}): {mg:+.4f} F1")

    # Bootstrap vs N=1
    print(f"\n  Paired bootstrap 95% CI vs N=1:")
    for n in VOTING_N_VALUES:
        if n == 1:
            print(f"    N=1: (baseline)")
            continue
        ci = voting_results.get(n, {}).get("bootstrap_diff_vs_n1", [])
        if ci:
            f1 = voting_results.get(n, {}).get("metrics", {}).get("f1", 0)
            f1_1 = voting_results.get(1, {}).get("metrics", {}).get("f1", 0)
            sig = ci[0] > 0 or ci[1] < 0
            print(f"    N={n}: ΔF1={f1 - f1_1:+.4f}  CI=({ci[0]:.4f}, {ci[1]:.4f})  "
                  f"{'✓' if sig else '✗'}")
        else:
            print(f"    N={n}: CI not computed")

    # Selection
    sel = voting_results.get("_selection", {})
    print(f"\n  Selection:")
    print(f"    Best-performing N:           {sel.get('best_n', 'N/A')} (F1={sel.get('best_f1', 0):.4f})")
    print(f"    Smallest N within 0.02 F1:   {sel.get('smallest_within_002', 'N/A')} (F1={sel.get('smallest_within_002_f1', 0):.4f})")
    print(f"    Best latency-adjusted N:     {sel.get('best_latency_adjusted_n', 'N/A')} (F1/s={sel.get('best_latency_adjusted_score', 0):.4f})")


def print_factorial_table(factorial_results: Dict[str, Any], domain: str, voting_n: int):
    """Print formatted 2×3 factorial results table."""
    print(f"\n{'=' * 70}")
    print(f"STAGE 2+3: 2×3 FACTORIAL — {domain.upper()}  (N*={voting_n})")
    print(f"{'=' * 70}")

    cells = [
        ("single_shot_rag_off", "SS RAG OFF"),
        ("single_shot_rag_on",  "SS RAG ON"),
        (f"voting_n{voting_n}_rag_off", f"Voting N={voting_n} RAG OFF"),
        (f"voting_n{voting_n}_rag_on",  f"Voting N={voting_n} RAG ON"),
        ("moa_rag_off", "MoA RAG OFF"),
        ("moa_rag_on",  "MoA RAG ON"),
    ]

    h = (f"{'Architecture':25s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} "
         f"{'Acc':8s} {'ESC':7s} {'ECE':7s} {'Lat':7s} {'Wall':7s}")
    print(h)
    print("-" * len(h))

    for key, label in cells:
        s = factorial_results.get(key, {})
        if s.get("skipped"):
            print(f"{label:25s} SKIPPED ({s.get('error', '')})")
            continue
        m = s.get("metrics", {})
        lt = s.get("latency", {})
        print(f"{label:25s} {m.get('f1', 0):.4f}  {m.get('precision', 0):.4f}  "
              f"{m.get('recall', 0):.4f}  {m.get('fpr', 0):.4f}  "
              f"{m.get('fnr', 0):.4f}  {m.get('accuracy', 0):.4f}  "
              f"{s.get('escalate_rate', 0):.2%}  {s.get('ece', 0):.4f}  "
              f"{lt.get('mean', 0):.1f}s  {s.get('elapsed_s', 0):.0f}s")

    # CIs
    print(f"\n  95% Bootstrap CIs:")
    for key, label in cells:
        s = factorial_results.get(key, {})
        ci = s.get("confidence_intervals", {})
        f1c = ci.get("f1_95", [])
        if f1c:
            m = s.get("metrics", {})
            print(f"    {label:25s} F1={m.get('f1', 0):.4f}  CI=({f1c[0]:.3f}, {f1c[1]:.3f})")

    # PPV
    print(f"\n  PPV across base rates:")
    bp_line = "    " + "".join(f"{' ' if br < 0.01 else ''}{f'{br*100:.1f}%':>8s}" for br in PPV_BASE_RATES)
    print(bp_line)
    for key, label in cells:
        s = factorial_results.get(key, {})
        ppv_data = s.get("ppv", [])
        if ppv_data:
            ppvs = "".join(f"{p['ppv']:8.4f}" for p in ppv_data)
            print(f"    {label:25s} {ppvs}")

    # Expected cost (FP:FN = 1:10)
    print(f"\n  Expected cost (FP:FN = 1:10):")
    for br in PPV_BASE_RATES[:3]:
        print(f"    base_rate={br*100:.1f}%:", end="")
        for key, label in cells:
            s = factorial_results.get(key, {})
            ec_data = s.get("expected_cost", {}).get("FP1_FN10", [])
            if ec_data:
                for entry in ec_data:
                    if entry.get("base_rate") == br:
                        print(f"  {label}={entry['expected_cost']:.4f}", end="")
        print()


def print_final_summary(all_domains_factual: Dict[str, Any],
                        all_domain_voting: Dict[str, Any],
                        voting_n: int):
    """Print cross-domain summary and comparison to previous results."""
    print(f"\n{'=' * 70}")
    print(f"STAGE 3: FINAL COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    # Per-architecture cross-domain mean
    cells = [
        ("single_shot_rag_off", "SS RAG OFF", 1),
        ("single_shot_rag_on", "SS RAG ON", 1),
        (f"voting_n{voting_n}_rag_off", f"Voting N={voting_n} RAG OFF", voting_n),
        (f"voting_n{voting_n}_rag_on", f"Voting N={voting_n} RAG ON", voting_n),
        ("moa_rag_off", "MoA RAG OFF", 3),
        ("moa_rag_on", "MoA RAG ON", 3),
    ]

    print(f"\n  Cross-domain mean F1:")
    for key, label, _ in cells:
        f1s = []
        latencies = []
        for domain in DOMAINS:
            fact = all_domains_factual.get(domain, {})
            s = fact.get(key, {})
            if s and not s.get("skipped") and s.get("metrics"):
                f1s.append(s["metrics"]["f1"])
                latencies.append(s.get("latency", {}).get("mean", 0))
        if f1s:
            mean_f1 = float(np.mean(f1s))
            mean_lat = float(np.mean(latencies))
            print(f"    {label:27s} F1={mean_f1:.4f}  Lat={mean_lat:.1f}s")

    # Total resources
    total_api = 0
    total_wall = 0
    for domain in DOMAINS:
        fact = all_domains_factual.get(domain, {})
        for key, _, _ in cells:
            s = fact.get(key, {})
            if s and not s.get("skipped"):
                total_api += s.get("total_api_calls", 0)
        # Voting sensitivity wall time
        voting_data = all_domain_voting.get(domain, {})
        for n in VOTING_N_VALUES:
            s = voting_data.get(n, {})
            total_wall += s.get("elapsed_s", 0)
            total_api += s.get("total_api_calls", 0)
        # Factorial wall time
        for key, _, _ in cells:
            s = fact.get(key, {})
            total_wall += s.get("elapsed_s", 0)

    print(f"\n  Total API calls: {total_api}")
    print(f"  Estimated cost:  ${total_api * COST_PER_CALL:.4f}")


# ══════════════════════════════════════════════════════════════════════════
#  PREVIOUS RESULTS COMPARISON
# ══════════════════════════════════════════════════════════════════════════


def compare_with_previous(revised_results: Dict[str, Any],
                          domain: str, voting_n: int):
    """Compare revised results with previous (role-diverse) results table."""
    print(f"\n  {'─' * 60}")
    print(f"  COMPARISON WITH PREVIOUS (role-diverse) RESULTS — {domain}")
    print(f"  {'─' * 60}")

    # Map old architecture names to new ones for comparison
    # Previous: single_shot, voting_n3, moa_rag, voting_n3_rag
    # New:      single_shot_rag_off, voting_n{voting_n}_rag_off, moa_rag_off, moa_rag_on, etc.
    mapping = {
        "Single-Shot (old)": ("single_shot_rag_off", "SS RAG OFF (new)"),
        "Voting N=3 (old)": (f"voting_n{voting_n}_rag_off", f"Voting N={voting_n} RAG OFF (new)"),
        "MoA+RAG (old)": ("moa_rag_on", "MoA RAG ON (new)"),
        "Voting N=3+RAG (old)": (f"voting_n{voting_n}_rag_on", f"Voting N={voting_n} RAG ON (new)"),
    }

    print(f"  {'Old Architecture':25s} {'Old F1':8s} {'New Architecture':27s} {'New F1':8s} {'ΔF1':8s}")
    print(f"  {'─' * 76}")
    for old_label, (new_key, new_label) in mapping.items():
        fact = revised_results.get(domain, {})
        new_s = fact.get(new_key, {})
        new_f1 = new_s.get("metrics", {}).get("f1", 0) if not new_s.get("skipped") else 0
        # We'll load old F1 from the existing report
        print(f"  {old_label:25s} {'?':>8s} {new_label:27s} {new_f1:.4f}  {'?':>8s}")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════


def main():
    global_start = time.time()

    print("=" * 70)
    print("REVISED VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Voting temperature: {VOTING_TEMPERATURE} (non-zero for self-consistency)")
    print(f"Single-Shot / MoA temperature: {TEMPERATURE}")
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

    # ── Stage 1: Voting-size sensitivity ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STAGE 1: VOTING-SIZE SENSITIVITY")
    print(f"{'=' * 70}")

    all_voting_results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    selected_n_per_domain: Dict[str, int] = {}

    for domain in DOMAINS:
        voting_results = run_voting_sensitivity(all_test[domain], domain, OUTPUT_DIR)
        all_voting_results[domain] = voting_results
        sel = voting_results.get("_selection", {})
        selected_n = sel.get("smallest_within_002", 3)  # default: smallest within 0.02
        selected_n_per_domain[domain] = selected_n
        print_voting_table(voting_results, domain)

    # Select global N* (use the larger of the two selected N for conservatism)
    # Or allow per-domain N*
    print(f"\n  Selected N* per domain:")
    for domain in DOMAINS:
        print(f"    {domain}: N*={selected_n_per_domain[domain]}")
    global_n = max(selected_n_per_domain.values())
    # But also consider: if one domain selects N=1 and another N=7, that's awkward
    # Use the max for conservatism
    print(f"  Global N* for Stage 2: {global_n} (max of per-domain selections)")

    # ── Stage 2+3: Factorial comparison ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("STAGE 2+3: 2×3 FACTORIAL COMPARISON")
    print(f"{'=' * 70}")

    all_factorial_results: Dict[str, Dict[str, Any]] = {}

    for domain in DOMAINS:
        vn = selected_n_per_domain[domain]
        factorial_results = run_factorial_comparison(
            all_test[domain], all_corpus[domain], domain, vn, OUTPUT_DIR,
        )
        all_factorial_results[domain] = factorial_results
        print_factorial_table(factorial_results, domain, vn)

    # ── Cross-domain summary ─────────────────────────────────────────────
    print_final_summary(all_factorial_results, all_voting_results, global_n)

    # ── Comparison with previous results ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH PREVIOUS (ROLE-DIVERSE) RESULTS")
    print(f"{'=' * 70}")

    # Load previous report if available
    prev_report_path = "results/final_validation/report.json"
    prev_data = None
    if os.path.exists(prev_report_path):
        with open(prev_report_path) as f:
            prev_data = json.load(f)
        print(f"  Loaded previous results from {prev_report_path}")
    else:
        print(f"  Previous report not found at {prev_report_path}")

    for domain in DOMAINS:
        compare_with_previous(all_factorial_results, domain,
                              selected_n_per_domain[domain])
        if prev_data and domain in prev_data.get("domains", {}):
            prev_domain = prev_data["domains"][domain]
            print(f"\n  Previous {domain} results:")
            for arch_name, arch_data in prev_domain.items():
                if isinstance(arch_data, dict) and "metrics" in arch_data:
                    m = arch_data["metrics"]
                    print(f"    {arch_name:20s} F1={m.get('f1', 0):.4f}  "
                          f"P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}  "
                          f"FPR={m.get('fpr', 0):.4f}  ESC={arch_data.get('escalate_rate', 0):.2%}")

    # ── Save master report ──────────────────────────────────────────────
    report = {
        "metadata": {
            "phase": "revised_validation",
            "model": MODEL,
            "timestamp": datetime.now().isoformat(),
            "voting_temperature": VOTING_TEMPERATURE,
            "concurrency": MAX_CONCURRENCY,
            "domains": DOMAINS,
            "description": "Canonical architectures: SS (one call), Voting (same prompt, temp>0, self-consistency), "
                           "MoA (Believer/Skeptic/Judge). RAG=evidence factor (OFF/ON). "
                           "Stage 1: voting N=1/3/5/7. Stage 2+3: 2×3 factorial.",
        },
        "voting_sensitivity": {},
        "factorial_comparison": {},
        "selected_n_per_domain": selected_n_per_domain,
        "global_n": global_n,
    }

    for domain in DOMAINS:
        v = all_voting_results[domain]
        # Strip raw results from serialization
        clean_v = {}
        for k in VOTING_N_VALUES + ["_selection"]:
            if k in v:
                d = {kk: vv for kk, vv in v[k].items() if kk != "_raw_results"}
                clean_v[str(k)] = d
        report["voting_sensitivity"][domain] = clean_v

        f = all_factorial_results[domain]
        clean_f = {}
        for k, vv in f.items():
            if k.startswith("_"):
                continue
            clean_f[k] = {kk: val for kk, val in vv.items() if kk != "_raw_results"}
        report["factorial_comparison"][domain] = clean_f

    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    total_runtime = time.time() - global_start
    print(f"\n{'=' * 70}")
    print(f"REVISED VALIDATION COMPLETE")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
