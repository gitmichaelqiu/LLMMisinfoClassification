"""Multi-model sensitivity analysis via OpenRouter.

Runs all architectures (Single-Shot, Voting N=1/3/5/7, MoA) on a reduced
test set across multiple LLMs through a single OpenRouter endpoint.
Every API call is persisted immediately so interrupted runs resume
without re-doing completed work.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from src.api import _llm_call, _parse_response, _run_parallel
from src.config import (
    COVID_CORPUS,
    COVID_TEST,
    FINANCE_CORPUS,
    FINANCE_TEST,
    SENSITIVITY_MODELS,
    SENSITIVITY_OUTPUT_DIR,
    SENSITIVITY_TEST_SIZE,
    TEMPERATURE,
    MAX_TOKENS,
    SEED,
)
from src.data import load_corpus_csv, load_test_csv
from src.evaluation import evaluate_architecture, analyze_voter_agreement
from src.prompts import (
    CANONICAL_SYSTEM,
    MOA_JUDGE,
    MOA_JUDGE_RAG,
    MOA_SKEPTIC,
    MOA_SKEPTIC_RAG,
    MOA_SUPPORTER,
    MOA_SUPPORTER_RAG,
)
from src.reporting import print_metrics
from src.retrieval import build_retriever, make_user_prompt
from src.schemas import Verdict, VerificationItem, VerificationResult
from src.storage import CallRecorder


# ── Model pricing (USD per 1M tokens) — used for cost reporting ─────
# Based on OpenRouter pricing as of July 2026.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.6-luna": {"input": 1.00, "output": 6.00},
    "glm-5.2": {"input": 1.40, "output": 4.40},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
}

# Architectures used as the key when storing individual API calls in the
# JSONL files.  ``token_counts`` iterates over these; voting uses a single
# shared name (not per-N) because the 6 new voter outputs share one file
# (the 7th vote reuses the SS call stored under ``single_shot_rag_*``).
_CALL_STORAGE_ARCHS: list[str] = [
    "single_shot_rag_off",
    "single_shot_rag_on",
    "voting_rag_off",
    "voting_rag_on",
    "moa_rag_off",
    "moa_rag_on",
]


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _sensitivity_call(
    model_cfg: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, float, dict]:
    """Make an API call through OpenRouter for the configured model.

    Returns ``(response_text, latency_s, usage_dict)``.
    """
    api_key = os.environ.get(model_cfg["api_key_var"], "")
    return _llm_call(
        system_prompt,
        user_prompt,
        model=model_cfg["openrouter_id"],
        api_key=api_key,
        base_url=model_cfg["base_url"],
    )


def _model_slug(key: str) -> str:
    """Return a filesystem-safe slug for a model key."""
    return key.replace("/", "-").replace("_", "-").replace(".", "-")


def _estimate_cost(
    model_key: str, total_input: int, total_output: int
) -> float:
    """Estimate USD cost from token counts using known per-model pricing."""
    p = _MODEL_PRICING.get(model_key, {"input": 1.0, "output": 6.0})
    return (total_input * p["input"] + total_output * p["output"]) / 1_000_000


def _reorder_results(
    items: list[VerificationItem],
    results: list[VerificationResult],
) -> list[VerificationResult]:
    """Re-order *results* to match the order of *items* by ``item_id``.

    Items without a matching result get a default ``VerificationResult`` with
    ``VERDICT.REAL``, confidence 0.5, and an ``"error": "missing_result"``
    metadata tag.  This ensures ``evaluate_architecture`` pairs truths with
    the correct predictions.
    """
    by_id = {r.item_id: r for r in results}
    return [
        by_id.get(
            it.id,
            VerificationResult(
                item_id=it.id,
                verdict=Verdict.REAL,
                confidence=0.5,
                latency_s=0,
                metadata={"error": "missing_result"},
            ),
        )
        for it in items
    ]


# ══════════════════════════════════════════════════════════════════════
#  Architecture runners  (with per-call persistence + recovery)
# ══════════════════════════════════════════════════════════════════════

def _run_ss(
    items: list[VerificationItem],
    rag_on: bool,
    corpus: list[VerificationItem] | None,
    model_key: str,
    model_cfg: dict[str, Any],
    domain: str,
    recorder: CallRecorder,
) -> list[VerificationResult]:
    """Single-Shot with per-call recording.

    Skips items that already have a saved result (recovery).
    """
    slug = _model_slug(model_key)
    arch = f"single_shot_rag_{'on' if rag_on else 'off'}"

    completed = recorder.completed_item_ids(domain, slug, arch)
    pending = [it for it in items if it.id not in completed]
    if not pending:
        return recorder.load_results(domain, slug, arch)

    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = build_retriever(corpus)

    def make_call(item: VerificationItem) -> VerificationResult:
        up = make_user_prompt(item.claim_text, vec, tfidf, texts)
        raw, lat, usage = _sensitivity_call(model_cfg, CANONICAL_SYSTEM, up)
        vr = _parse_response(
            raw, item.id, lat, {"architecture": "single_shot", "rag": rag_on}
        )
        recorder.record_call(
            domain, slug, arch, item.id,
            CANONICAL_SYSTEM, up, raw,
            vr.verdict.name, vr.confidence, lat, usage,
        )
        recorder.record_result(
            domain, slug, arch, item.id,
            vr.verdict.name, vr.confidence, vr.latency_s,
            {"architecture": "single_shot", "rag": rag_on},
        )
        return vr

    _run_parallel(
        [lambda it=item: make_call(it) for item in pending],
        f"SS {domain} {arch}",
    )
    return _reorder_results(items, recorder.load_results(domain, slug, arch))


def _run_voting(
    items: list[VerificationItem],
    rag_on: bool,
    corpus: list[VerificationItem] | None,
    model_key: str,
    model_cfg: dict[str, Any],
    domain: str,
    recorder: CallRecorder,
    n_new: int = 6,
) -> dict[int, tuple[list[VerificationResult], list[list[VerificationResult]]]]:
    """Voting with per-call recording.

    Reuses the already-run Single-Shot result as voter 0 (the first vote),
    then generates *n_new* (default 6) additional independent votes, for a
    total of 7 votes per item.  This saves one API call per item compared
    to generating all 7 from scratch.

    If *any* of the N=1/3/5/7 aggregation results are missing, regenerates
    the *n_new* votes and re-aggregates.  Every generated call is saved
    immediately, so interrupted runs resume without repeating work.
    """
    slug = _model_slug(model_key)
    rl = "rag_on" if rag_on else "rag_off"

    # ── Load SS results to reuse as voter 0 ──────────────────────
    ss_arch = f"single_shot_rag_{rl}"
    ss_results = recorder.load_results(domain, slug, ss_arch)
    ss_by_id: dict[str, VerificationResult] = {r.item_id: r for r in ss_results}
    all_ss_available = all(it.id in ss_by_id for it in items)

    # ── Check which N-aggregations are complete ──────────────────
    all_n_complete = True
    for N in [1, 3, 5, 7]:
        arch = f"voting_n{N}_{rl}"
        completed = recorder.completed_item_ids(domain, slug, arch)
        item_ids = {it.id for it in items}
        if not item_ids.issubset(completed):
            all_n_complete = False
            break

    if all_n_complete:
        result_map: dict[int, tuple[list[VerificationResult], list[list[VerificationResult]]]] = {}
        for N in [1, 3, 5, 7]:
            arch = f"voting_n{N}_{rl}"
            results = _reorder_results(items, recorder.load_results(domain, slug, arch))
            result_map[N] = (results, [])
        return result_map

    # ── Generate n_new voter outputs ────────────────────────────
    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = build_retriever(corpus)

    callables: list[Any] = []
    for item in items:
        up = make_user_prompt(item.claim_text, vec, tfidf, texts)
        for v in range(n_new):
            # New voters get indices 1..n_new; SS is index 0
            voter_idx = v + 1

            def _vote(it: VerificationItem = item, u: str = up, vidx: int = voter_idx):
                txt, lat, usage = _sensitivity_call(model_cfg, CANONICAL_SYSTEM, u)
                vr = _parse_response(
                    txt, it.id, lat,
                    {"voter_idx": vidx, "architecture": "voting", "rag": rag_on},
                )
                recorder.record_call(
                    domain, slug, f"voting_{rl}", it.id,
                    CANONICAL_SYSTEM, u, txt,
                    vr.verdict.name, vr.confidence, lat, usage,
                    voter_idx=vidx,
                )
                return (it.id, vidx, vr)
            callables.append(_vote)

    raw = _run_parallel(callables, f"Voting {domain} {rl}")

    # Group new voters by item
    item_voters_new: dict[str, list[VerificationResult]] = defaultdict(list)
    for r in raw:
        if r is not None:
            _, vidx, vr = r
            item_voters_new[vr.item_id].append(vr)
    for iid in item_voters_new:
        item_voters_new[iid].sort(key=lambda x: x.metadata.get("voter_idx", 0))

    # ── Combine: SS + new voters → 7 total votes per item ─────────
    n_total = n_new + 1  # 7 total (1 SS + 6 new)
    result_map = {}
    for N in [1, 3, 5, 7]:
        threshold = N // 2 + 1
        aggregated: list[VerificationResult] = []
        per_item_voters: list[list[VerificationResult]] = []
        arch = f"voting_n{N}_{rl}"
        for item in items:
            ss_vote = ss_by_id.get(item.id)
            new_votes = item_voters_new.get(item.id, [])
            if ss_vote and all_ss_available:
                # Prepend SS as voter 0
                ss_vote.metadata = {**ss_vote.metadata, "voter_idx": 0}
                all_votes = [ss_vote] + new_votes
            else:
                # Fallback: use only new votes (should not happen in normal flow)
                all_votes = new_votes
            voters = all_votes[:N]
            per_item_voters.append(voters)
            if not voters:
                vr = VerificationResult(
                    item_id=item.id, verdict=Verdict.REAL,
                    confidence=0.5, latency_s=0,
                    metadata={"architecture": arch, "error": "no_voters"},
                )
            else:
                counter = Counter(v.verdict for v in voters)
                mc = counter.most_common(1)
                verdict = (
                    mc[0][0] if (mc and mc[0][1] >= threshold)
                    else Verdict.ESCALATE
                )
                vr = VerificationResult(
                    item_id=item.id, verdict=verdict,
                    confidence=float(np.mean([v.confidence for v in voters])),
                    latency_s=max(v.latency_s for v in voters),
                    evidence=[f"N={N}: {dict(counter)}"],
                    metadata={
                        "architecture": arch, "rag": rag_on,
                        "n_voters": len(voters), "threshold": threshold,
                        "verdict_distribution": {
                            k.name: v for k, v in counter.items()
                        },
                    },
                )
            aggregated.append(vr)
            recorder.record_result(
                domain, slug, arch, item.id,
                vr.verdict.name, vr.confidence, vr.latency_s,
                vr.metadata,
            )
        result_map[N] = (aggregated, per_item_voters)
    return result_map


def _run_moa(
    items: list[VerificationItem],
    rag_on: bool,
    corpus: list[VerificationItem] | None,
    model_key: str,
    model_cfg: dict[str, Any],
    domain: str,
    recorder: CallRecorder,
) -> list[VerificationResult]:
    """MoA (Supporter+Skeptic → Judge) with per-call recording.

    Skips items with a saved Judge result.
    """
    slug = _model_slug(model_key)
    arch = f"moa_rag_{'on' if rag_on else 'off'}"

    completed = recorder.completed_item_ids(domain, slug, arch)
    pending = [it for it in items if it.id not in completed]
    if not pending:
        return recorder.load_results(domain, slug, arch)

    vec = tfidf = texts = None
    if rag_on and corpus:
        vec, tfidf, texts = build_retriever(corpus)

    evidence_map: dict[str, str] = {}
    p1_callables: list[Any] = []
    for item in pending:
        up = make_user_prompt(item.claim_text, vec, tfidf, texts)
        evidence_map[item.id] = up
        sp = MOA_SUPPORTER_RAG if rag_on else MOA_SUPPORTER
        sk = MOA_SKEPTIC_RAG if rag_on else MOA_SKEPTIC

        def _sup(iid: str = item.id, u: str = up, p: str = sp):
            txt, lat, usage = _sensitivity_call(model_cfg, p, u)
            recorder.record_call(
                domain, slug, arch, iid,
                p, u, txt, "N/A", 0.0, lat, usage,
                extra_metadata={"moa_role": "supporter"},
            )
            return ("supporter", iid, txt, lat)

        def _ske(iid: str = item.id, u: str = up, p: str = sk):
            txt, lat, usage = _sensitivity_call(model_cfg, p, u)
            recorder.record_call(
                domain, slug, arch, iid,
                p, u, txt, "N/A", 0.0, lat, usage,
                extra_metadata={"moa_role": "skeptic"},
            )
            return ("skeptic", iid, txt, lat)

        p1_callables.append(_sup)
        p1_callables.append(_ske)

    raw_p1 = _run_parallel(p1_callables, f"MoA P1 {domain} {arch}")

    supporter_out: dict[str, tuple[str, float]] = {}
    skeptic_out: dict[str, tuple[str, float]] = {}
    for result in raw_p1:
        if result is None:
            continue
        role, iid, text, lat = result
        if role == "supporter":
            supporter_out[iid] = (text, lat)
        else:
            skeptic_out[iid] = (text, lat)

    judge_prompt = MOA_JUDGE_RAG if rag_on else MOA_JUDGE
    p2_callables: list[Any] = []
    for item in pending:
        up = evidence_map.get(item.id, f"Claim to verify:\n{item.claim_text}")
        sup_text, sup_lat = supporter_out.get(item.id, ("Error", 0))
        ske_text, ske_lat = skeptic_out.get(item.id, ("Error", 0))
        p1_lat = max(sup_lat, ske_lat)
        ctx = (
            f"{up}\n\n── Supporter's Analysis ──\n{sup_text}\n\n"
            f"── Skeptic's Analysis ──\n{ske_text}\n"
        )

        def _judge(iid: str = item.id, c: str = ctx, p1: float = p1_lat):
            txt, lat, usage = _sensitivity_call(model_cfg, judge_prompt, c)
            vr = _parse_response(
                txt, iid, lat, {"architecture": "moa", "rag": rag_on}
            )
            vr.latency_s = p1 + lat
            recorder.record_call(
                domain, slug, arch, iid,
                judge_prompt, c, txt,
                vr.verdict.name, vr.confidence, lat, usage,
                extra_metadata={"moa_role": "judge"},
            )
            recorder.record_result(
                domain, slug, arch, iid,
                vr.verdict.name, vr.confidence, vr.latency_s,
                vr.metadata,
            )
            return (iid, vr)

        p2_callables.append(_judge)

    raw_p2 = _run_parallel(p2_callables, f"MoA P2 {domain} {arch}")
    return _reorder_results(items, recorder.load_results(domain, slug, arch))


# ══════════════════════════════════════════════════════════════════════
#  Comparison table
# ══════════════════════════════════════════════════════════════════════

def _print_comparison(
    all_results: dict[str, dict[str, dict[str, Any]]],
    model_keys: list[str],
) -> None:
    """Print cross-model comparison table (the five key findings)."""
    findings = [
        ("SS RAG OFF F1", "single_shot_rag_off", "metrics", "f1"),
        ("SS RAG ON F1", "single_shot_rag_on", "metrics", "f1"),
        ("Vot N=3 OFF F1", "voting_n3_rag_off", "metrics", "f1"),
        ("Vot N=3 ON F1", "voting_n3_rag_on", "metrics", "f1"),
        ("MoA OFF F1", "moa_rag_off", "metrics", "f1"),
        ("MoA ON F1", "moa_rag_on", "metrics", "f1"),
        ("SS OFF ESC", "single_shot_rag_off", "escalate_rate", None),
        ("MoA OFF ESC", "moa_rag_off", "escalate_rate", None),
        ("Vot N=7 OFF ESC", "voting_n7_rag_off", "escalate_rate", None),
    ]

    print(f"\n{'=' * 70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 70}")

    # Header
    header = f"  {'Metric':22s}"
    for mk in model_keys:
        d = SENSITIVITY_MODELS[mk]["display"]
        header += f"  {d:18s}"
    print(header)
    print(f"  {'-' * (22 + 20 * len(model_keys))}")

    for label, arch_key, section, field in findings:
        row = f"  {label:22s}"
        for mk in model_keys:
            val = None
            for domain in ["finance", "healthcare"]:
                ar = all_results.get(mk, {}).get(domain, {}).get(arch_key, {})
                if section == "metrics":
                    m = ar.get("metrics", {})
                    v = m.get(field, None) if field else None
                elif section == "escalate_rate":
                    v = ar.get("escalate_rate", None)
                else:
                    v = None
                if v is not None:
                    val = v
            if val is not None:
                row += f"  {val:>8.4f}       "
            else:
                row += f"  {'—':>8s}       "
        print(row)

    # Cross-domain mean F1
    print(f"\n  {'Cross-domain Mean F1':22s}")
    for label, arch_key in [
        ("SS OFF", "single_shot_rag_off"),
        ("SS ON", "single_shot_rag_on"),
        ("Vot N=3 OFF", "voting_n3_rag_off"),
        ("Vot N=3 ON", "voting_n3_rag_on"),
        ("Vot N=7 OFF", "voting_n7_rag_off"),
        ("Vot N=7 ON", "voting_n7_rag_on"),
        ("MoA OFF", "moa_rag_off"),
        ("MoA ON", "moa_rag_on"),
    ]:
        row = f"  {label:22s}"
        for mk in model_keys:
            f1s = []
            for domain in ["finance", "healthcare"]:
                s = all_results.get(mk, {}).get(domain, {}).get(arch_key, {})
                m = s.get("metrics", {})
                if m.get("f1") is not None:
                    f1s.append(m["f1"])
            if f1s:
                mean_f1 = float(np.mean(f1s))
                row += f"  {mean_f1:>8.4f}       "
            else:
                row += f"  {'—':>8s}       "
        print(row)

    # Operational: reviews per 1,000
    print(f"\n  {'Reviews / 1,000 claims':22s}")
    for mk in model_keys:
        display = SENSITIVITY_MODELS[mk]["display"]
        esc_rates = []
        for domain in ["finance", "healthcare"]:
            for arch_key in ["single_shot_rag_off", "moa_rag_off", "voting_n7_rag_off"]:
                s = all_results.get(mk, {}).get(domain, {}).get(arch_key, {})
                er = s.get("escalate_rate", None)
                if er is not None:
                    esc_rates.append(er)
        if esc_rates:
            avg_esc = float(np.mean(esc_rates))
            print(f"  {display:22s}  {avg_esc * 1000:>8.0f}           ")

    print()


def _save_report(
    all_results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Write a JSON report with all metrics and cost info."""
    report: dict[str, Any] = {
        "metadata": {
            "experiment": "model_sensitivity",
            "models": [
                {
                    "key": k,
                    "display": v["display"],
                    "openrouter_id": v["openrouter_id"],
                }
                for k, v in SENSITIVITY_MODELS.items()
            ],
            "test_size_per_domain": SENSITIVITY_TEST_SIZE,
            "timestamp": datetime.now().isoformat(),
        },
    }
    for model_key, domain_results in all_results.items():
        report[model_key] = {}
        for domain, archs in domain_results.items():
            report[model_key][domain] = {
                k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in archs.items()
            }

    report_path = os.path.join(SENSITIVITY_OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")


# ══════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis() -> None:
    """Orchestrate the full multi-model sensitivity sweep."""
    global_start = time.time()
    print("=" * 70)
    print("MODEL SENSITIVITY ANALYSIS — 3 models × 2 domains × all archs")
    print(f"Test size: {SENSITIVITY_TEST_SIZE} items per domain")
    models_to_run = list(SENSITIVITY_MODELS.keys())
    print(f"Models: {', '.join(SENSITIVITY_MODELS[m]['display'] for m in models_to_run)}")
    print("=" * 70)

    # ── Verify API ────────────────────────────────────────────────
    print("\nVerifying API (OpenRouter)...")
    try:
        first_cfg = SENSITIVITY_MODELS[models_to_run[0]]
        raw, lat, _ = _sensitivity_call(first_cfg, "You are a test assistant.", "Reply with OK.")
        print(f"  OpenRouter OK ({lat:.1f}s)")
    except Exception as e:
        print(f"  OpenRouter ERROR: {e}")
        print("  Ensure OPENROUTER_API_KEY is set in .env")
        return

    # ── Load data ─────────────────────────────────────────────────
    print("\nLoading data...")
    finance_test_full = load_test_csv(FINANCE_TEST, "finance")
    finance_corpus = load_corpus_csv(FINANCE_CORPUS, "finance")
    covid_test_full = load_test_csv(COVID_TEST, "healthcare")
    covid_corpus = load_corpus_csv(COVID_CORPUS, "healthcare")

    n = SENSITIVITY_TEST_SIZE
    finance_test = finance_test_full[:n]
    covid_test = covid_test_full[:n]
    print(f"  Finance: {len(finance_test)} test/{len(finance_corpus)} corpus items")
    print(f"  COVID:   {len(covid_test)} test/{len(covid_corpus)} corpus items")

    domains: list[tuple[str, list[VerificationItem], list[VerificationItem]]] = [
        ("finance", finance_test, finance_corpus),
        ("healthcare", covid_test, covid_corpus),
    ]

    # ── Run models ────────────────────────────────────────────────
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    total_estimated_cost = 0.0

    for model_key in models_to_run:
        model_cfg = SENSITIVITY_MODELS[model_key]
        display = model_cfg["display"]
        print(f"\n{'=' * 65}")
        print(f"{display} ({model_key})")
        print(f"{'=' * 65}")

        recorder = CallRecorder(SENSITIVITY_OUTPUT_DIR)
        domain_results: dict[str, dict[str, Any]] = {}

        for domain_name, test_items, corpus_items in domains:
            print(f"\n  ── {domain_name.upper()} ──")
            sys.stdout.flush()

            arch_results: dict[str, Any] = {}

            # Single-Shot RAG OFF
            t0 = time.time()
            ss_off = _run_ss(test_items, False, None, model_key, model_cfg, domain_name, recorder)
            arch = "single_shot_rag_off"
            arch_results[arch] = evaluate_architecture(test_items, ss_off, arch, len(test_items))
            arch_results[arch]["elapsed_s"] = time.time() - t0
            print_metrics(arch_results[arch], f"  SS RAG OFF")

            # Single-Shot RAG ON
            t0 = time.time()
            ss_on = _run_ss(test_items, True, corpus_items, model_key, model_cfg, domain_name, recorder)
            arch = "single_shot_rag_on"
            arch_results[arch] = evaluate_architecture(test_items, ss_on, arch, len(test_items))
            arch_results[arch]["elapsed_s"] = time.time() - t0
            print_metrics(arch_results[arch], f"  SS RAG ON")

            # Voting
            for rag_flag, rag_label in [(False, "rag_off"), (True, "rag_on")]:
                t0 = time.time()
                voting_map = _run_voting(
                    test_items, rag_flag,
                    corpus_items if rag_flag else None,
                    model_key, model_cfg, domain_name, recorder,
                )
                v_elapsed = time.time() - t0
                for N in [1, 3, 5, 7]:
                    agg, per_voter = voting_map[N]
                    arch_n = f"voting_n{N}_{rag_label}"
                    arch_results[arch_n] = evaluate_architecture(
                        test_items, agg, arch_n, len(test_items) * 7,
                    )
                    arch_results[arch_n]["elapsed_s"] = round(v_elapsed, 1)
                    print_metrics(arch_results[arch_n], f"  Vot N={N} {rag_label}")

            # MoA
            for rag_flag, rag_label in [(False, "rag_off"), (True, "rag_on")]:
                t0 = time.time()
                moa_r = _run_moa(
                    test_items, rag_flag,
                    corpus_items if rag_flag else None,
                    model_key, model_cfg, domain_name, recorder,
                )
                arch_m = f"moa_{rag_label}"
                arch_results[arch_m] = evaluate_architecture(
                    test_items, moa_r, arch_m, len(test_items) * 3,
                )
                arch_results[arch_m]["elapsed_s"] = time.time() - t0
                print_metrics(arch_results[arch_m], f"  MoA {rag_label}")

            domain_results[domain_name] = arch_results

        # ── Cost summary for this model ───────────────────────────
        total_in = total_out = 0
        for domain_name, _test_items, _corpus_items in domains:
            for call_arch in _CALL_STORAGE_ARCHS:
                inp, out = recorder.token_counts(
                    domain_name, _model_slug(model_key), call_arch
                )
                total_in += inp
                total_out += out

        est_cost = _estimate_cost(model_key, total_in, total_out)
        total_estimated_cost += est_cost
        print(f"\n  ── {display} token summary ──")
        print(f"    Total input tokens:  {total_in:,}")
        print(f"    Total output tokens: {total_out:,}")
        print(f"    Estimated API cost:  ${est_cost:.2f}")

        all_results[model_key] = domain_results

    # ── Final comparison ──────────────────────────────────────────
    _print_comparison(all_results, models_to_run)
    _save_report(all_results)

    total_runtime = time.time() - global_start
    print(f"\n{'=' * 70}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime / 60:.1f} min)")
    print(f"Total estimated cost: ${total_estimated_cost:.2f}")
    print(f"Results: {SENSITIVITY_OUTPUT_DIR}/")
