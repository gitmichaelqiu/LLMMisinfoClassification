"""Figure generation for the verification experiment."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def generate_figures(all_results: dict[str, Any], output_dir: str) -> None:
    """Generate comparison figures from experiment results."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Fig 1: F1 vs N
    plt.figure(figsize=(8, 5))
    for domain in ["finance", "healthcare"]:
        ns, f1s = [], []
        for N in [1, 3, 5, 7]:
            s = all_results[domain].get(f"voting_n{N}_rag_off", {})
            if s.get("metrics"):
                ns.append(N)
                f1s.append(s["metrics"]["f1"])
        plt.plot(ns, f1s, "o-", label=domain.capitalize())
    plt.xlabel("N voters")
    plt.ylabel("F1")
    plt.title("F1 vs Voting Size (RAG OFF)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig1_f1_vs_n.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Architecture comparison
    plt.figure(figsize=(12, 5))
    arch_keys = [
        "single_shot_rag_off",
        "single_shot_rag_on",
        "voting_n3_rag_off",
        "voting_n3_rag_on",
        "moa_rag_off",
        "moa_rag_on",
    ]
    arch_labels = ["SS\nOFF", "SS\nON", "V3\nOFF", "V3\nON", "MoA\nOFF", "MoA\nON"]
    x = np.arange(len(arch_labels))
    for i, domain in enumerate(["finance", "healthcare"]):
        f1s = [
            all_results[domain].get(k, {}).get("metrics", {}).get("f1", 0)
            for k in arch_keys
        ]
        plt.bar(x + i * 0.25 - 0.25, f1s, 0.25, label=domain.capitalize())
    plt.xlabel("Architecture")
    plt.ylabel("F1")
    plt.title("Architecture Comparison")
    plt.xticks(x, arch_labels)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "fig2_architecture_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Fig 3: RAG effect
    plt.figure(figsize=(8, 5))
    effects, labels = [], []
    for arch, label in [
        ("single_shot", "SS"),
        ("voting_n3", "V3"),
        ("voting_n5", "V5"),
        ("voting_n7", "V7"),
        ("moa", "MoA"),
    ]:
        for domain in ["finance", "healthcare"]:
            soff = (
                all_results[domain]
                .get(f"{arch}_rag_off", {})
                .get("metrics", {})
                .get("f1", 0)
            )
            son = (
                all_results[domain]
                .get(f"{arch}_rag_on", {})
                .get("metrics", {})
                .get("f1", 0)
            )
            if soff or son:
                effects.append(son - soff)
                labels.append(f"{label}\n{domain[:4]}")
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in effects]
    plt.bar(range(len(effects)), effects, color=colors)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.xticks(range(len(effects)), labels, fontsize=9)
    plt.ylabel("DF1 (RAG ON - RAG OFF)")
    plt.title("RAG Effect on F1")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "fig3_rag_effect.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(f"  Figures saved to {fig_dir}/")
