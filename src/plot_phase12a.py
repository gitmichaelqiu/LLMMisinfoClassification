"""Generate updated Phase 11 + Phase 12a plots for meeting evidence map.

Reads from output/repair_rerun/report.json and output/phase12a_rag_factorial/report.json
Outputs PNGs to plots/ directory.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

DOMAINS = ["finance", "healthcare", "political"]
BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
BR_LABELS = ["0.1%", "1%", "5%", "10%", "25%", "50%"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Colors: SS=blue, Voting=green, MoA=orange
# Solid = no RAG, Dashed/faded = +RAG
COLORS = {
    "ss": "#4C72B0", "voting": "#55A868", "moa": "#DD8452",
    "ss_rag": "#7BA0D5", "voting_rag": "#7FBF8F", "moa_rag": "#E8B07D",
}
LABELS = {
    "ss": "SS", "ss_rag": "SS+RAG",
    "voting": "Voting", "voting_rag": "Voting+RAG",
    "moa": "MoA", "moa_rag": "MoA+RAG",
}

ARCH_KEY_P11 = {
    "ss": "single_shot", "voting": "repaired_voting_n3",
    "moa": "repaired_moa", "ss_rag": "repaired_rag",
}
ARCH_KEY_P12 = {"voting_rag": "voting_n3_rag", "moa_rag": "moa_rag"}


def load_data():
    with open("output/repair_rerun/report.json") as f:
        p11 = json.load(f)
    p12a = None
    p12_path = "output/phase12a_rag_factorial/report.json"
    if os.path.exists(p12_path):
        with open(p12_path) as f:
            p12a = json.load(f)
    return p11, p12a


def get_metric(p11, p12a, domain, arch):
    if arch in ARCH_KEY_P11:
        key = ARCH_KEY_P11[arch]
        d = p11.get("domains", {}).get(domain, {}).get(key, {})
        return d.get("metrics", {}) or d
    elif arch in ARCH_KEY_P12 and p12a:
        key = ARCH_KEY_P12[arch]
        return p12a.get("domains", {}).get(domain, {}).get(key, {}).get("metrics", {})
    return {}


def bayesian_ppv(sens, fpr, prevalence):
    denom = sens * prevalence + fpr * (1 - prevalence)
    return (sens * prevalence) / denom if denom > 0 else 0.0


# ── Plot 1: 2x3 Factorial F1 Matrix ──────────────────────────────

def plot_factorial_matrix(p11, p12a):
    archs = ["ss", "ss_rag", "voting", "voting_rag", "moa", "moa_rag"]
    f1_matrix = np.zeros((3, 6))
    for i, d in enumerate(DOMAINS):
        for j, a in enumerate(archs):
            m = get_metric(p11, p12a, d, a)
            f1_matrix[i, j] = m.get("f1", 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(6)
    width = 0.25
    for i, d in enumerate(DOMAINS):
        bars = ax.bar(x + i * width, f1_matrix[i], width,
                      label=d.title(), alpha=0.85)
        for bar, v in zip(bars, f1_matrix[i]):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["SS\n(no RAG)", "SS\n+RAG", "Voting\n(no RAG)", "Voting\n+RAG", "MoA\n(no RAG)", "MoA\n+RAG"])
    ax.set_ylabel("F1 Score")
    ax.set_title("2x3 Factorial Matrix: F1 by Architecture and Evidence Augmentation")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.axhline(y=0.0, color="grey", linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase12a_factorial_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Plot 2: Comparison bars — Voting vs Voting+RAG, MoA vs MoA+RAG ──

def plot_rag_comparison(p11, p12a):
    pairs = [("voting", "voting_rag"), ("moa", "moa_rag")]
    pair_labels = [("Voting", "Voting+RAG"), ("MoA", "MoA+RAG")]
    metrics_to_plot = ["f1", "precision", "recall"]
    metric_labels = ["F1", "Precision", "Recall"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, (a_no, a_yes) in enumerate(pairs):
        for col, (d) in enumerate(DOMAINS):
            ax = axes[row, col]
            m_no = get_metric(p11, p12a, d, a_no)
            m_yes = get_metric(p11, p12a, d, a_yes)

            vals_no = [m_no.get(met, 0) for met in metrics_to_plot]
            vals_yes = [m_yes.get(met, 0) for met in metrics_to_plot]

            x = np.arange(3)
            w = 0.3
            ax.bar(x - w/2, vals_no, w, label=pair_labels[row][0], color=COLORS[a_no], alpha=0.8)
            ax.bar(x + w/2, vals_yes, w, label=pair_labels[row][1], color=COLORS[a_yes], alpha=0.8)

            # Add delta labels
            for xi, vn, vy in zip(x, vals_no, vals_yes):
                if vn != vy:
                    delta = vy - vn
                    ax.annotate(f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}",
                                (xi + w/2, vy + 0.03), ha="center", fontsize=7,
                                color="green" if delta > 0 else "red", weight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_title(f"{d.title()} - {pair_labels[row][0]}", fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.2)

    fig.suptitle("Evidence Augmentation Impact: No RAG vs +RAG", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "phase12a_rag_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Plot 3: PPV curves for the complete 2x3 matrix ───────────────

def plot_ppv_2x3(p11, p12a):
    archs = ["ss", "ss_rag", "voting", "voting_rag", "moa", "moa_rag"]
    styles = ["-", "--", "-", "--", "-", "--"]
    fig, ax = plt.subplots(figsize=(8, 5))
    br_arr = np.array(BASE_RATES)

    for i, a in enumerate(archs):
        sens_list, fpr_list = [], []
        for d in DOMAINS:
            m = get_metric(p11, p12a, d, a)
            if m:
                sens_list.append(m.get("recall", 0))
                fpr_list.append(m.get("fpr", 0))
        if not sens_list:
            continue
        sens = np.mean(sens_list)
        fpr_val = np.mean(fpr_list)
        ppv = [bayesian_ppv(sens, fpr_val, br) for br in br_arr]
        ax.semilogx(br_arr, ppv, linestyle=styles[i], marker="o",
                     label=LABELS[a], color=COLORS[a], linewidth=2, markersize=5)

    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(br_arr[-1]*1.8, 0.5, "PPV=0.5", fontsize=8, color="grey", va="center")
    ax.set_xlabel("Base Rate P(Fake)")
    ax.set_ylabel("PPV")
    ax.set_title("PPV Across Base Rates: 2x3 Matrix (Cross-Domain Mean)")
    ax.set_xlim(br_arr[0]*0.5, br_arr[-1]*4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(br_arr)
    ax.set_xticklabels(BR_LABELS)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase12a_ppv_2x3.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── main ──────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    p11, p12a = load_data()
    if p12a is None:
        print("WARNING: Phase 12a report not found — some plots will be incomplete")

    print("Generating plots...")
    plot_factorial_matrix(p11, p12a)
    plot_rag_comparison(p11, p12a)
    plot_ppv_2x3(p11, p12a)
    print(f"All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
