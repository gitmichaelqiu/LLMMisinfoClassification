"""Result persistence and console reporting."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from src.config import MODEL, OUTPUT_DIR
from src.schemas import VerificationItem, VerificationResult


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


def print_final_table(
    domain_results: dict[str, Any], domain: str
) -> None:
    cells = [
        ("single_shot_rag_off", "SS RAG OFF"),
        ("single_shot_rag_on", "SS RAG ON"),
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

    print("\n  95% Bootstrap CIs:")
    for key, label in cells:
        s = domain_results.get(key, {})
        ci = s.get("confidence_intervals", {}).get("f1_95", [])
        if ci:
            print(
                f"    {label:22s} F1={s['metrics']['f1']:.4f}  "
                f"CI=({ci[0]:.3f},{ci[1]:.3f})"
            )
