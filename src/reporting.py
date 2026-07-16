"""Result persistence and console reporting."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any

import numpy as np

from src.config import MODEL, OUTPUT_DIR
from src.schemas import VerificationItem, VerificationResult


# ── Raw output persistence ──────────────────────────────────────
def save_raw_output(
    domain: str,
    arch_name: str,
    items: list[VerificationItem],
    results: list[VerificationResult],
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Write a JSON file with per-item verdicts for one architecture run."""
    path = os.path.join(output_dir, "raw_outputs", f"{domain}_{arch_name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "architecture": arch_name,
                "domain": domain,
                "model": MODEL,
                "results": [
                    {
                        "item_id": r.item_id,
                        "verdict": r.verdict.name,
                        "confidence": r.confidence,
                        "latency_s": r.latency_s,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            default=str,
        )
    return path


# ── Console helpers ─────────────────────────────────────────────
def print_metrics(section: dict[str, Any], label: str = "") -> None:
    """Print a single line of metrics for one architecture."""
    m = section.get("metrics", {})
    ci = section.get("confidence_intervals", {})
    ci_str = (
        f"  F1 CI=({ci['f1_95'][0]:.3f},{ci['f1_95'][1]:.3f})"
        if ci.get("f1_95")
        else ""
    )
    print(
        f"    {label:30s} "
        f"F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  "
        f"R={m.get('recall', 0):.4f}  FPR={m.get('fpr', 0):.4f}  "
        f"FNR={m.get('fnr', 0):.4f}  Acc={m.get('accuracy', 0):.4f}  "
        f"ESC={section.get('escalate_rate', 0):.2%}  "
        f"ECE={section.get('ece', 0):.4f}  "
        f"Lat={section.get('latency', {}).get('mean', 0):.1f}s{ci_str}"
    )
    sys.stdout.flush()


# ── Voting sensitivity table ────────────────────────────────────
def extract_voting_table(
    domain_results: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    table: dict[int, dict[str, Any]] = {}
    for N in [1, 3, 5, 7]:
        for rag in ["rag_off", "rag_on"]:
            key = f"voting_n{N}_{rag}"
            if key in domain_results:
                if N not in table:
                    table[N] = {}
                s = domain_results[key]
                table[N][rag] = {
                    "metrics": s.get("metrics", {}),
                    "latency": s.get("latency", {}),
                    "escalate_rate": s.get("escalate_rate", 0),
                    "ece": s.get("ece", 0),
                    "mean_disagreement": s.get("mean_disagreement", 0),
                    "agreement": s.get("voter_agreement", {}),
                    "confidence_intervals": s.get("confidence_intervals", {}),
                }
    return table


def print_voting_sensitivity(
    domain_results: dict[str, Any], domain: str
) -> dict[int, dict[str, Any]]:
    table = extract_voting_table(domain_results)
    rag_key = "rag_off"
    print(f"\n  VOTING SENSITIVITY - {domain.upper()} (RAG OFF)")
    print(
        f"  {'N':6s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} {'FNR':8s} "
        f"{'Acc':8s} {'ESC':7s} {'ECE':7s} {'Unanim':7s} {'PairAgr':7s} {'Lat':6s}"
    )
    print(f"  {'-' * 100}")
    for N in [1, 3, 5, 7]:
        if N in table and rag_key in table[N]:
            s = table[N][rag_key]
            m = s["metrics"]
            lt = s["latency"]
            a = s.get("agreement", {})
            print(
                f"  {N:6d} {m.get('f1', 0):.4f}  {m.get('precision', 0):.4f}  "
                f"{m.get('recall', 0):.4f}  {m.get('fpr', 0):.4f}  "
                f"{m.get('fnr', 0):.4f}  {m.get('accuracy', 0):.4f}  "
                f"{s.get('escalate_rate', 0):.2%}  {s.get('ece', 0):.4f}  "
                f"{a.get('unanimous', 0) / max(a.get('total_items', 1), 1):.2%}  "
                f"{a.get('pairwise_agreement', 0):.4f}  {lt.get('mean', 0):.1f}s"
            )

    print(f"\n  Paired bootstrap 95% CI vs N=1 (RAG OFF):")
    for N in [3, 5, 7]:
        if N in table and rag_key in table[N]:
            ci = table[N][rag_key].get("confidence_intervals", {}).get("f1_95", [])
            f1_n = table[N][rag_key]["metrics"]["f1"]
            f1_1 = table[1][rag_key]["metrics"]["f1"]
            sig = "NS" if (ci[0] <= 0 <= ci[1]) else "SIG"
            print(f"    N={N}: DF1={f1_n - f1_1:+.4f}  CI=({ci[0]:.3f},{ci[1]:.3f})  {sig}")

    return table


# ── Final comparison table ──────────────────────────────────────
def print_final_table(
    domain_results: dict[str, Any], domain: str
) -> None:
    cells = [
        ("tfidf_baseline", "TF-IDF Baseline"),
        ("single_shot_rag_off", "SS RAG OFF"),
        ("single_shot_rag_on", "SS RAG ON"),
        ("voting_n1_rag_off", "Vot N=1 OFF"),
        ("voting_n1_rag_on", "Vot N=1 ON"),
        ("voting_n3_rag_off", "Vot N=3 OFF"),
        ("voting_n3_rag_on", "Vot N=3 ON"),
        ("voting_n5_rag_off", "Vot N=5 OFF"),
        ("voting_n5_rag_on", "Vot N=5 ON"),
        ("voting_n7_rag_off", "Vot N=7 OFF"),
        ("voting_n7_rag_on", "Vot N=7 ON"),
        ("moa_rag_off", "MoA RAG OFF"),
        ("moa_rag_on", "MoA RAG ON"),
    ]
    print(f"\n  FINAL ARCHITECTURE COMPARISON - {domain.upper()}")
    h = (
        f"  {'Architecture':22s} {'F1':8s} {'P':8s} {'R':8s} {'FPR':8s} "
        f"{'FNR':8s} {'Acc':8s} {'ESC':7s} {'ECE':7s} {'Lat':6s} {'Calls':6s}"
    )
    print(h)
    print(f"  {'-' * len(h.strip())}")
    for key, label in cells:
        s = domain_results.get(key, {})
        if not s or not s.get("metrics"):
            continue
        m = s["metrics"]
        lt = s["latency"]
        print(
            f"  {label:22s} {m.get('f1', 0):.4f}  {m.get('precision', 0):.4f}  "
            f"{m.get('recall', 0):.4f}  {m.get('fpr', 0):.4f}  "
            f"{m.get('fnr', 0):.4f}  {m.get('accuracy', 0):.4f}  "
            f"{s.get('escalate_rate', 0):.2%}  {s.get('ece', 0):.4f}  "
            f"{lt.get('mean', 0):.1f}s  {s.get('total_api_calls', 0):5d}"
        )

    print(f"\n  95% Bootstrap CIs:")
    for key, label in cells:
        s = domain_results.get(key, {})
        ci = s.get("confidence_intervals", {}).get("f1_95", [])
        if ci:
            print(
                f"    {label:22s} F1={s['metrics']['f1']:.4f}  "
                f"CI=({ci[0]:.3f},{ci[1]:.3f})"
            )


# ── Cross-domain summary ───────────────────────────────────────
def print_cross_domain_summary(all_results: dict[str, Any]) -> None:
    print(f"\n  CROSS-DOMAIN MEAN F1")
    cells = [
        ("tfidf_baseline", "TF-IDF"),
        ("single_shot_rag_off", "SS OFF"),
        ("single_shot_rag_on", "SS ON"),
        ("voting_n1_rag_off", "V1 OFF"),
        ("voting_n1_rag_on", "V1 ON"),
        ("voting_n3_rag_off", "V3 OFF"),
        ("voting_n3_rag_on", "V3 ON"),
        ("voting_n5_rag_off", "V5 OFF"),
        ("voting_n5_rag_on", "V5 ON"),
        ("voting_n7_rag_off", "V7 OFF"),
        ("voting_n7_rag_on", "V7 ON"),
        ("moa_rag_off", "MoA OFF"),
        ("moa_rag_on", "MoA ON"),
    ]
    print(f"  {'Architecture':12s} {'Fin F1':8s} {'Health F1':10s} {'Mean':8s} {'Lat':8s}")
    print(f"  {'-' * 46}")
    for key, label in cells:
        f1s, lats = [], []
        for domain in ["finance", "healthcare"]:
            s = all_results.get(domain, {}).get(key, {})
            if s and s.get("metrics"):
                f1s.append(s["metrics"]["f1"])
                lats.append(s.get("latency", {}).get("mean", 0))
        if f1s:
            print(
                f"  {label:12s} {f1s[0]:.4f}     {f1s[1]:.4f}     "
                f"{float(np.mean(f1s)):.4f}  {float(np.mean(lats)):.1f}s"
            )


# ── Pairwise bootstrap comparisons ──────────────────────────────
def print_pairwise_comparisons(all_results: dict[str, Any]) -> None:
    print(f"\n  PAIRWISE COMPARISONS (Bootstrap DF1)")
    comparisons = [
        ("RAG ON vs OFF (SS)", "single_shot_rag_on", "single_shot_rag_off"),
        ("RAG ON vs OFF (Vot N=3)", "voting_n3_rag_on", "voting_n3_rag_off"),
        ("RAG ON vs OFF (MoA)", "moa_rag_on", "moa_rag_off"),
        ("MoA vs SS (RAG OFF)", "moa_rag_off", "single_shot_rag_off"),
        ("MoA vs SS (RAG ON)", "moa_rag_on", "single_shot_rag_on"),
        ("Vot N=3 vs N=1 (RAG OFF)", "voting_n3_rag_off", "voting_n1_rag_off"),
        ("Vot N=3 vs N=1 (RAG ON)", "voting_n3_rag_on", "voting_n1_rag_on"),
        ("SS vs TF-IDF", "single_shot_rag_off", "tfidf_baseline"),
    ]
    for comp_name, key_a, key_b in comparisons:
        print(f"\n  {comp_name}:")
        for domain in ["finance", "healthcare"]:
            sa = all_results.get(domain, {}).get(key_a, {})
            sb = all_results.get(domain, {}).get(key_b, {})
            fa = sa.get("metrics", {}).get("f1", 0)
            fb = sb.get("metrics", {}).get("f1", 0)
            print(f"    {domain}: A={fa:.4f} vs B={fb:.4f}  D={fa - fb:+.4f}")
