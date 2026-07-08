"""Generate Phase 11 plots for meeting evidence map.

Reads from output/repair_rerun/report.json and results/phase11/
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
ARCHS = ["single_shot", "voting_n3", "moa", "rag"]
ARCH_LABELS = ["Single-Shot", "Voting N=3", "MoA", "SS+RAG†"]
ARCH_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
BR_LABELS = ["0.1%", "1%", "5%", "10%", "25%", "50%"]

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    with open("output/repair_rerun/report.json") as f:
        report = json.load(f)
    with open("results/phase11/base_rate_routing_summary.json") as f:
        summary = json.load(f)
    return report, summary


def extract_metrics(report):
    """Return {domain: {arch: {f1, prec, rec, fpr, lat, ece}}}"""
    arch_key = {
        "single_shot": "single_shot",
        "voting_n3": "repaired_voting_n3",
        "moa": "repaired_moa",
        "rag": "repaired_rag",
    }
    data = {}
    for d in DOMAINS:
        data[d] = {}
        for a in ARCHS:
            k = arch_key[a]
            ad = report.get("domains", {}).get(d, {}).get(k, {})
            if ad.get("skipped", False):
                continue
            m = ad.get("metrics", {})
            lat = ad.get("latency", {})
            data[d][a] = {
                "f1": m.get("f1", 0),
                "prec": m.get("precision", 0),
                "rec": m.get("recall", 0),
                "fpr": m.get("fpr", 0),
                "lat": lat.get("mean", 0),
                "ece": ad.get("ece", 0),
            }
    return data


def compute_cross(data):
    """Compute cross-domain means."""
    cross = {}
    for a in ARCHS:
        vals = [data[d][a] for d in DOMAINS if a in data[d]]
        if not vals:
            continue
        cross[a] = {k: np.mean([v[k] for v in vals]) for k in vals[0]}
    return cross


# ── Plot 1: F1 bar chart ───────────────────────────────────────────

def plot_f1_comparison(data, cross):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(DOMAINS) + 1)  # 3 domains + cross
    width = 0.2

    for i, a in enumerate(ARCHS):
        vals = [data[d].get(a, {}).get("f1", 0) for d in DOMAINS]
        vals.append(cross.get(a, {}).get("f1", 0))
        bars = ax.bar(x + i * width, vals, width, label=ARCH_LABELS[i], color=ARCH_COLORS[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Finance", "Healthcare", "Political", "Cross-Domain"])
    ax.set_ylabel("F1 Score")
    ax.set_title("Architecture F1 Score by Domain")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.0, color="grey", linewidth=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase11_f1_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── Plot 2: PPV curve ─────────────────────────────────────────────

def plot_ppv_curves(cross):
    fig, ax = plt.subplots(figsize=(7, 5))
    br_arr = np.array(BASE_RATES)

    for i, a in enumerate(ARCHS):
        if a not in cross:
            continue
        c = cross[a]
        sens, fpr = c["rec"], c["fpr"]
        ppv = sens * br_arr / (sens * br_arr + fpr * (1 - br_arr))
        ax.semilogx(br_arr, ppv, marker="o", label=ARCH_LABELS[i],
                     color=ARCH_COLORS[i], linewidth=2, markersize=5)

    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(br_arr[-1] * 1.8, 0.5, "PPV=0.5 (operational threshold)", fontsize=8,
            color="grey", va="center")

    ax.set_xlabel("Base Rate P(Fake)")
    ax.set_ylabel("Positive Predictive Value (PPV)")
    ax.set_title("PPV Across Base Rates (Cross-Domain Mean)")
    ax.set_xlim(br_arr[0] * 0.5, br_arr[-1] * 3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(br_arr)
    ax.set_xticklabels(BR_LABELS)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase11_ppv_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── Plot 3: Expected cost at FP:FN = 1:10 ──────────────────────────

def plot_expected_cost(cross):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    cost_ratios = [(1, 1), (1, 5), (1, 10), (1, 25)]
    ratio_labels = ["FP:FN = 1:1", "FP:FN = 1:5", "FP:FN = 1:10", "FP:FN = 1:25"]

    for idx, (ax, (cfp, cfn), rlabel) in enumerate(zip(axes, cost_ratios, ratio_labels)):
        br_arr = np.array(BASE_RATES)
        for i, a in enumerate(ARCHS):
            if a not in cross:
                continue
            c = cross[a]
            ec = (1 - br_arr) * c["fpr"] * cfp + br_arr * (1 - c["rec"]) * cfn
            ax.semilogx(br_arr, ec, marker="o", label=ARCH_LABELS[i],
                         color=ARCH_COLORS[i], linewidth=2, markersize=4)

        ax.set_title(rlabel, fontsize=10)
        ax.set_xlabel("Base Rate")
        ax.set_xticks(br_arr)
        ax.set_xticklabels(BR_LABELS, fontsize=7)
        if idx == 0:
            ax.set_ylabel("Expected Per-Item Cost")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.suptitle("Expected Cost Across Cost Ratios (Cross-Domain Mean)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "phase11_expected_cost.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── Plot 4: Latency vs F1 scatter ──────────────────────────────────

def plot_latency_f1(data, cross):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Per-domain points (small circles, lighter)
    for d in DOMAINS:
        for i, a in enumerate(ARCHS):
            if a not in data.get(d, {}):
                continue
            lat = data[d][a]["lat"]
            f1v = data[d][a]["f1"]
            ax.scatter(lat, f1v, s=60, c=[ARCH_COLORS[i]], marker="o",
                       alpha=0.35, zorder=3)

    # Cross-domain means (large diamonds, full opacity)
    cross_legend = []
    for i, a in enumerate(ARCHS):
        if a not in cross:
            continue
        lat = cross[a]["lat"]
        f1v = cross[a]["f1"]
        h = ax.scatter(lat, f1v, s=200, c=[ARCH_COLORS[i]], marker="D",
                       alpha=0.9, zorder=5, label=ARCH_LABELS[i])
        cross_legend.append(h)

    from matplotlib.lines import Line2D
    pd_handle = Line2D([0], [0], marker="o", color="gray",
                       linestyle="", markersize=6, alpha=0.5,
                       label="Per-domain (N=10)")

    # Latency budget zones (no labels in legend — use text annotations)
    ax.axvspan(0, 5, alpha=0.06, color="green")
    ax.axvspan(5, 15, alpha=0.06, color="orange")
    ax.axvspan(15, 30, alpha=0.06, color="red")
    ax.text(2.5, 0.03, "<5s HFT window", ha="center", fontsize=8, color="green", alpha=0.7)
    ax.text(10, 0.03, "<15s swing trading", ha="center", fontsize=8, color="orange", alpha=0.7)
    ax.text(22, 0.03, "<30s fundamental", ha="center", fontsize=8, color="red", alpha=0.7)

    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Latency vs F1: Architecture Positioning")
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 1.1)

    lgnd1 = ax.legend(handles=cross_legend, loc="upper left",
                      framealpha=0.9, fontsize=9, title="Cross-domain mean")
    ax.add_artist(lgnd1)
    ax.legend(handles=[pd_handle], loc="center right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase11_latency_f1.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── Plot 5: Regime win pie ─────────────────────────────────────────

def plot_regime_wins(summary):
    fig, ax = plt.subplots(figsize=(5, 5))
    wins = summary.get("regime_win_counts", {})
    labels = []
    sizes = []
    colors = []
    for a in ARCHS:
        cnt = wins.get(a, 0)
        if cnt > 0:
            labels.append(f"{ARCH_LABELS[ARCHS.index(a)]}\n({cnt})")
            sizes.append(cnt)
            colors.append(ARCH_COLORS[ARCHS.index(a)])

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.6,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title("Regime Wins by Architecture\n(72 configurations)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase11_regime_wins.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── Plot 6: Calibration / ECE comparison ───────────────────────────

def plot_ece_comparison(data, cross):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(DOMAINS) + 1)
    width = 0.2

    for i, a in enumerate(ARCHS):
        vals = [data[d].get(a, {}).get("ece", 0) for d in DOMAINS]
        vals.append(cross.get(a, {}).get("ece", 0))
        bars = ax.bar(x + i * width, vals, width, label=ARCH_LABELS[i], color=ARCH_COLORS[i])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Finance", "Healthcare", "Political", "Cross-Domain"])
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("Calibration Error by Architecture (lower is better)")
    ax.set_ylim(0, 0.5)
    ax.axhline(y=0.0, color="grey", linewidth=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "phase11_ece_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ── main ───────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    report, summary = load_data()
    data = extract_metrics(report)
    cross = compute_cross(data)

    print("Generating plots...")
    plot_f1_comparison(data, cross)
    plot_ppv_curves(cross)
    plot_expected_cost(cross)
    plot_latency_f1(data, cross)
    plot_regime_wins(summary)
    plot_ece_comparison(data, cross)
    print(f"All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
