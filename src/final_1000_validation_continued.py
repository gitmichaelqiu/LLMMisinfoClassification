"""Continuation entry point: run only missing experiments.

Loads existing raw output files, checks which architectures are still
missing, and runs only those.  Useful for resuming after a partial run.

Usage:
    python -m src.final_1000_validation_continued
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from src.api import _llm_call
from src.architectures import run_moa, run_single_shot, run_tfidf_baseline, run_voting_all_n
from src.config import (
    COVID_CORPUS,
    COVID_TEST,
    FINANCE_CORPUS,
    FINANCE_TEST,
    MODEL,
    OUTPUT_DIR,
    TEMPERATURE,
    MAX_CONCURRENCY,
)
from src.data import load_test_csv, load_corpus_csv
from src.evaluation import analyze_voter_agreement, evaluate_architecture
from src.reporting import print_metrics, save_raw_output
from src.schemas import Verdict, VerificationResult

load_dotenv()


# ── Load previously saved results ───────────────────────────────
def load_existing_results(
    domain: str, output_dir: str = OUTPUT_DIR
) -> dict[str, list[VerificationResult]]:
    """Reconstruct ``VerificationResult`` lists from saved JSON files."""
    prefix = f"{domain}_"
    results: dict[str, list[VerificationResult]] = {}
    raw_dir = os.path.join(output_dir, "raw_outputs")
    if not os.path.isdir(raw_dir):
        return results
    for fname in os.listdir(raw_dir):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        arch_name = fname[len(prefix) : -5]
        with open(os.path.join(raw_dir, fname)) as f:
            data = json.load(f)
        loaded: list[VerificationResult] = []
        for rd in data.get("results", []):
            meta = dict(rd.get("metadata", {}) or {})
            vr = VerificationResult(
                item_id=rd["item_id"],
                verdict=Verdict[rd["verdict"]],
                confidence=rd.get("confidence", 0.5),
                latency_s=rd.get("latency_s", 0),
                evidence=[],
                metadata=meta,
            )
            loaded.append(vr)
        results[arch_name] = loaded
        print(f"  Loaded existing: {arch_name} ({len(loaded)} items)")
    return results


# ── Continuation orchestrator ───────────────────────────────────
def main() -> None:
    global_start = time.time()
    print("=" * 70)
    print("FINAL 1000-ITEM VALIDATION — CONTINUATION")
    print("=" * 70)
    print(f"Model: {MODEL}, Temperature: {TEMPERATURE}, Concurrency: {MAX_CONCURRENCY}")
    sys.stdout.flush()

    # Verify API
    print("\nVerifying API...")
    try:
        raw, lat = _llm_call("You are a test assistant.", "Reply with OK.")
        print(f"  API OK ({lat:.1f}s)")
    except Exception as e:
        print(f"  API ERROR: {e}")
        return

    # Load test data
    print("\nLoading data...")
    finance_test = load_test_csv(FINANCE_TEST, "finance")
    finance_corpus = load_corpus_csv(FINANCE_CORPUS, "finance")
    covid_test = load_test_csv(COVID_TEST, "healthcare")
    covid_corpus = load_corpus_csv(COVID_CORPUS, "healthcare")
    print(f"  Finance: {len(finance_test)} test, {len(finance_corpus)} corpus")
    print(f"  COVID:   {len(covid_test)} test, {len(covid_corpus)} corpus")

    # Load existing results
    print("\nLoading existing results...")
    finance_existing = load_existing_results("finance", OUTPUT_DIR)
    covid_existing = load_existing_results("healthcare", OUTPUT_DIR)

    all_results: dict[str, dict] = {"finance": {}, "healthcare": {}}
    total_api_calls = 0

    # ═══ FINANCE ═══════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("FINANCE — running MoA RAG ON (only missing experiment)")
    print(f"{'=' * 65}")

    for arch, raw_results in finance_existing.items():
        n = len(finance_test)
        api_count = (n * 7 if "voting" in arch else n * 3 if "moa" in arch
                     else n if "single_shot" in arch else 0)
        section = evaluate_architecture(finance_test, raw_results, arch, api_count)
        section["_cached_results"] = raw_results
        all_results["finance"][arch] = section
        print_metrics(section, arch)

    if "moa_rag_on" not in finance_existing:
        print(f"\n  MoA RAG ON (finance)...")
        sys.stdout.flush()
        t0 = time.time()
        moa_on = run_moa(finance_test, rag_on=True, corpus=finance_corpus)
        elapsed = time.time() - t0
        arch = "moa_rag_on"
        all_results["finance"][arch] = evaluate_architecture(
            finance_test, moa_on, arch, len(finance_test) * 3
        )
        all_results["finance"][arch]["elapsed_s"] = round(elapsed, 1)
        all_results["finance"][arch]["_cached_results"] = moa_on
        save_raw_output("finance", arch, finance_test, moa_on, OUTPUT_DIR)
        print_metrics(all_results["finance"][arch], "MoA RAG ON")
        total_api_calls += len(finance_test) * 3
    else:
        print(f"  MoA RAG ON already exists, skipping")

    # ═══ HEALTHCARE ════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("HEALTHCARE — running all experiments")
    print(f"{'=' * 65}")

    items, corpus = covid_test, covid_corpus
    missing_health = [
        k
        for k in [
            "tfidf_baseline",
            "single_shot_rag_off",
            "single_shot_rag_on",
            "voting_n1_rag_off",
            "voting_n3_rag_off",
            "voting_n5_rag_off",
            "voting_n7_rag_off",
            "voting_n1_rag_on",
            "voting_n3_rag_on",
            "voting_n5_rag_on",
            "voting_n7_rag_on",
            "moa_rag_off",
            "moa_rag_on",
        ]
        if k not in covid_existing
    ]

    if covid_existing:
        print(
            f"\n  Found {len(covid_existing)} existing healthcare results, "
            f"missing: {missing_health}"
        )
        for arch, raw_results in covid_existing.items():
            n = len(items)
            est = n * 7 if "voting" in arch else n * 3 if "moa" in arch else n if "single_shot" in arch else 0
            all_results["healthcare"][arch] = evaluate_architecture(
                items, raw_results, arch, est
            )
            all_results["healthcare"][arch]["_cached_results"] = raw_results
            print_metrics(all_results["healthcare"][arch], arch)

    # TF-IDF
    if "tfidf_baseline" not in all_results["healthcare"]:
        print(f"\n  TF-IDF Baseline...")
        sys.stdout.flush()
        t0 = time.time()
        r = run_tfidf_baseline(items, corpus)
        all_results["healthcare"]["tfidf_baseline"] = evaluate_architecture(
            items, r, "tfidf_baseline", 0
        )
        all_results["healthcare"]["tfidf_baseline"]["_cached_results"] = r
        all_results["healthcare"]["tfidf_baseline"]["elapsed_s"] = time.time() - t0
        print_metrics(all_results["healthcare"]["tfidf_baseline"], "TF-IDF Baseline")

    # Single-Shot OFF/ON
    for rag_flag, rag_label in [(False, "RAG OFF"), (True, "RAG ON")]:
        arch = f"single_shot_{rag_label.lower().replace(' ', '_')}"
        if arch not in all_results["healthcare"]:
            print(f"\n  Single-Shot {rag_label}...")
            sys.stdout.flush()
            t0 = time.time()
            r = run_single_shot(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            all_results["healthcare"][arch] = evaluate_architecture(
                items, r, arch, len(items)
            )
            all_results["healthcare"][arch]["elapsed_s"] = time.time() - t0
            all_results["healthcare"][arch]["_cached_results"] = r
            save_raw_output("healthcare", arch, items, r, OUTPUT_DIR)
            print_metrics(all_results["healthcare"][arch], f"SS {rag_label}")

    # Voting (7 outputs → N=1,3,5,7)
    for rag_flag, rag_label in [(False, "RAG OFF"), (True, "RAG ON")]:
        rl = rag_label.lower().replace(" ", "_")
        all_exist = all(
            f"voting_n{N}_{rl}" in all_results["healthcare"] for N in [1, 3, 5, 7]
        )
        if not all_exist:
            print(f"\n  Voting {rag_label} (7 outputs per item)...")
            sys.stdout.flush()
            t0 = time.time()
            voting_map = run_voting_all_n(
                items, rag_on=rag_flag, corpus=corpus if rag_flag else None, n_total=7
            )
            v_elapsed = time.time() - t0
            for N in [1, 3, 5, 7]:
                agg, per_voter = voting_map[N]
                arch = f"voting_n{N}_{rl}"
                all_results["healthcare"][arch] = evaluate_architecture(
                    items, agg, arch, len(items) * 7
                )
                all_results["healthcare"][arch]["elapsed_s"] = round(v_elapsed, 1)
                all_results["healthcare"][arch]["_cached_results"] = agg
                all_results["healthcare"][arch]["voter_agreement"] = (
                    analyze_voter_agreement(items, per_voter, N)
                )
                save_raw_output("healthcare", arch, items, agg, OUTPUT_DIR)
                print_metrics(all_results["healthcare"][arch], f"Voting N={N} {rag_label}")
        else:
            print(f"  Voting {rag_label} already complete")

    # MoA OFF/ON
    for rag_flag, rag_label in [(False, "RAG OFF"), (True, "RAG ON")]:
        arch = f"moa_{rag_label.lower().replace(' ', '_')}"
        if arch not in all_results["healthcare"]:
            print(f"\n  MoA {rag_label}...")
            sys.stdout.flush()
            t0 = time.time()
            r = run_moa(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            all_results["healthcare"][arch] = evaluate_architecture(
                items, r, arch, len(items) * 3
            )
            all_results["healthcare"][arch]["elapsed_s"] = time.time() - t0
            all_results["healthcare"][arch]["_cached_results"] = r
            save_raw_output("healthcare", arch, items, r, OUTPUT_DIR)
            print_metrics(all_results["healthcare"][arch], f"MoA {rag_label}")

    # ── Print summary ──────────────────────────────────────────
    total_runtime = time.time() - global_start

    # Simple summary tables
    print(f"\n{'=' * 70}")
    print("VOTING SIZE SENSITIVITY (RAG OFF)")
    print(f"{'=' * 70}")
    for domain in ["finance", "healthcare"]:
        print(f"\n  {domain.upper()}:")
        print(f"  {'N':6s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} "
              f"{'Acc':8s} {'ESC':7s} {'Lat':6s}")
        print(f"  {'-' * 60}")
        for N in [1, 3, 5, 7]:
            s = all_results[domain].get(f"voting_n{N}_rag_off", {})
            if not s.get("metrics"):
                continue
            m = s["metrics"]
            lt = s["latency"]
            print(f"  {N:6d} {m.get('f1', 0):.4f}  {m.get('precision', 0):.4f}  "
                  f"{m.get('recall', 0):.4f}  {m.get('fpr', 0):.4f}  "
                  f"{m.get('fnr', 0):.4f}  {m.get('accuracy', 0):.4f}  "
                  f"{s.get('escalate_rate', 0):.2%}  {lt.get('mean', 0):.1f}s")

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime / 60:.1f} min)")
    print(f"Results: {OUTPUT_DIR}/")
    sys.stdout.flush()

    # Save report
    report = {
        "metadata": {
            "experiment": "final_1000_validation_continued",
            "model": MODEL,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "total_runtime_s": total_runtime,
        },
        "data_summary": {
            "finance": {"test": len(finance_test), "corpus": len(finance_corpus)},
            "healthcare": {"test": len(covid_test), "corpus": len(covid_corpus)},
        },
    }
    for domain in ["finance", "healthcare"]:
        report[domain] = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in all_results[domain].items()
        }
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {OUTPUT_DIR}/report.json")


if __name__ == "__main__":
    main()
