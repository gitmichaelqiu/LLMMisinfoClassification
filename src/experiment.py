"""Experiment orchestrator — runs all architectures on both domains."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from src.api import _llm_call
from src.architectures import run_moa, run_single_shot, run_tfidf_baseline, run_voting_all_n
from src.config import (
    COST_PER_CALL,
    COVID_CORPUS,
    COVID_TEST,
    FINANCE_CORPUS,
    FINANCE_TEST,
    MAX_CONCURRENCY,
    MODEL,
    OUTPUT_DIR,
    TEMPERATURE,
)
from src.data import load_all_data
from src.evaluation import analyze_voter_agreement, evaluate_architecture
from src.figures import generate_figures
from src.reporting import (
    print_cross_domain_summary,
    print_final_table,
    print_metrics,
    print_pairwise_comparisons,
    print_voting_sensitivity,
    save_raw_output,
)

load_dotenv()


def main() -> None:
    global_start = time.time()
    print("=" * 70)
    print("FINAL 1000-ITEM LARGE-SAMPLE VALIDATION")
    print("=" * 70)
    print(f"Model: {MODEL}, Temperature: {TEMPERATURE}, Concurrency: {MAX_CONCURRENCY}")
    sys.stdout.flush()

    # ── Verify API ──────────────────────────────────────────────
    print("\nVerifying API...")
    try:
        raw, lat, _ = _llm_call("You are a test assistant.", "Reply with OK.")
        print(f"  API OK ({lat:.1f}s)")
    except Exception as e:
        print(f"  API ERROR: {e}")
        return

    # ── Load data ───────────────────────────────────────────────
    print("\nLOADING DATA")
    try:
        all_data = load_all_data(FINANCE_TEST, FINANCE_CORPUS, COVID_TEST, COVID_CORPUS)
    except FileNotFoundError as e:
        print(f"  DATA ERROR: {e}")
        print("  Ensure test/corpus CSV files exist. "
              "See data/raw/finance/ and data/raw/health/.")
        return

    all_test = {d: all_data[d]["test"] for d in all_data}
    all_corpus = {d: all_data[d]["corpus"] for d in all_data}

    # ── Run experiments ─────────────────────────────────────────
    print("\nEXPERIMENTS")
    all_results: dict[str, Any] = {}
    for domain in ["finance", "healthcare"]:
        items = all_test[domain]
        corpus = all_corpus[domain]
        print(f"\n{'=' * 65}")
        print(f"{domain.upper()} - {len(items)} items, {len(corpus)} corpus")
        print(f"{'=' * 65}")
        domain_results: dict[str, Any] = {}

        # TF-IDF Baseline
        print("\n  TF-IDF Baseline...")
        sys.stdout.flush()
        t0 = time.time()
        tfidf_r = run_tfidf_baseline(items, corpus)
        domain_results["tfidf_baseline"] = evaluate_architecture(
            items, tfidf_r, "tfidf_baseline", 0
        )
        domain_results["tfidf_baseline"]["elapsed_s"] = time.time() - t0
        print_metrics(domain_results["tfidf_baseline"], "TF-IDF Baseline")

        # Single-Shot RAG OFF / ON
        for rag_flag, rag_label in [(False, "RAG OFF"), (True, "RAG ON")]:
            arch_label = f"single_shot_{rag_label.lower().replace(' ', '_')}"
            print(f"\n  Single-Shot {rag_label}...")
            sys.stdout.flush()
            t0 = time.time()
            ss_r = run_single_shot(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            domain_results[arch_label] = evaluate_architecture(
                items, ss_r, arch_label, len(items)
            )
            domain_results[arch_label]["elapsed_s"] = time.time() - t0
            domain_results[arch_label]["_cached_results"] = ss_r
            save_raw_output(domain, arch_label, items, ss_r, OUTPUT_DIR)
            print_metrics(domain_results[arch_label], f"SS {rag_label}")

        # Voting (7 outputs → N=1, 3, 5, 7)
        for rag_flag, rag_label in [(False, "RAG OFF"), (True, "RAG ON")]:
            rl = rag_label.lower().replace(" ", "_")
            print(f"\n  Voting {rag_label} (7 outputs per item)...")
            sys.stdout.flush()
            t0 = time.time()
            voting_map = run_voting_all_n(
                items, rag_on=rag_flag, corpus=corpus if rag_flag else None, n_total=7
            )
            v_elapsed = time.time() - t0
            api_calls = len(items) * 7
            for N in [1, 3, 5, 7]:
                agg, per_voter = voting_map[N]
                arch_label = f"voting_n{N}_{rl}"
                domain_results[arch_label] = evaluate_architecture(
                    items, agg, arch_label, api_calls
                )
                domain_results[arch_label]["elapsed_s"] = round(v_elapsed, 1)
                domain_results[arch_label]["_cached_results"] = agg
                domain_results[arch_label]["voter_agreement"] = analyze_voter_agreement(
                    items, per_voter, N
                )
                save_raw_output(domain, arch_label, items, agg, OUTPUT_DIR)
                print_metrics(domain_results[arch_label], f"Voting N={N} {rag_label}")

        # MoA RAG OFF / ON
        for rag_flag, rag_label in [(False, "RAG OFF"), (True, "RAG ON")]:
            arch_label = f"moa_{rag_label.lower().replace(' ', '_')}"
            print(f"\n  MoA {rag_label}...")
            sys.stdout.flush()
            t0 = time.time()
            moa_r = run_moa(items, rag_on=rag_flag, corpus=corpus if rag_flag else None)
            domain_results[arch_label] = evaluate_architecture(
                items, moa_r, arch_label, len(items) * 3
            )
            domain_results[arch_label]["elapsed_s"] = time.time() - t0
            domain_results[arch_label]["_cached_results"] = moa_r
            save_raw_output(domain, arch_label, items, moa_r, OUTPUT_DIR)
            print_metrics(domain_results[arch_label], f"MoA {rag_label}")

        all_results[domain] = domain_results

    # ── Print results ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VOTING SIZE SENSITIVITY (RAG OFF)")
    print(f"{'=' * 70}")
    for domain in ["finance", "healthcare"]:
        print_voting_sensitivity(all_results[domain], domain)

    print(f"\n{'=' * 70}")
    print("FINAL ARCHITECTURE COMPARISON")
    print(f"{'=' * 70}")
    for domain in ["finance", "healthcare"]:
        print_final_table(all_results[domain], domain)
    print_cross_domain_summary(all_results)
    print_pairwise_comparisons(all_results)

    # ── Generate figures ────────────────────────────────────────
    print("\nGENERATING FIGURES")
    generate_figures(all_results, OUTPUT_DIR)

    # ── Summary ─────────────────────────────────────────────────
    total_runtime = time.time() - global_start
    total_calls = sum(
        all_results[d].get("voting_n7_rag_off", {}).get("total_api_calls", 0)
        + all_results[d].get("voting_n7_rag_on", {}).get("total_api_calls", 0)
        + all_results[d].get("single_shot_rag_off", {}).get("total_api_calls", 0)
        + all_results[d].get("single_shot_rag_on", {}).get("total_api_calls", 0)
        + all_results[d].get("moa_rag_off", {}).get("total_api_calls", 0)
        + all_results[d].get("moa_rag_on", {}).get("total_api_calls", 0)
        for d in ["finance", "healthcare"]
    )

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total runtime: {total_runtime:.0f}s ({total_runtime / 60:.1f} min)")
    print(f"Estimated total API calls: ~{total_calls}")
    print(f"Estimated cost: ~${total_calls * COST_PER_CALL:.4f}")
    print(f"Results: {OUTPUT_DIR}/")
    sys.stdout.flush()

    # ── Save report ─────────────────────────────────────────────
    report = {
        "metadata": {
            "experiment": "final_1000_validation",
            "model": MODEL,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "total_runtime_s": total_runtime,
        },
        "data_summary": {
            d: {
                "test_size": len(all_test[d]),
                "corpus_size": len(all_corpus[d]),
            }
            for d in ["finance", "healthcare"]
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
