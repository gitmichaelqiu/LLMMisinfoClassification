"""Generate comparative MoA vs Single-Shot robustness plots.

Loads results from both the single-shot and MoA adversarial stress tests,
then produces a side-by-side comparison showing Precision/Recall/P&L Saved
at each bot intensity level.

Output: output/moa_robustness_plots/comparison.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    """Load adversarial results JSON."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def generate_comparison_plot(
    singleshot_path="./output/adversarial_results_singleshot.json",
    moa_path="./output/adversarial_results_moa.json",
    output_dir="./output/moa_robustness_plots",
):
    """Generate MoA vs Single-Shot comparison plot."""
    ss_results = load_results(singleshot_path)
    moa_results = load_results(moa_path)

    if not ss_results or not moa_results:
        print(f"[MoAPlot] Missing results: ss={len(ss_results)}, moa={len(moa_results)}")
        return

    os.makedirs(output_dir, exist_ok=True)

    bot_levels = [r["bot_pct"] for r in ss_results]
    x_labels = [f"{int(b*100)}%" for b in bot_levels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Precision comparison
    ax = axes[0]
    ss_prec = [r["precision"] for r in ss_results]
    moa_prec = [r["precision"] for r in moa_results]
    x = np.arange(len(bot_levels))
    w = 0.35
    ax.bar(x - w/2, ss_prec, w, label="Single-Shot", color="steelblue", alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, moa_prec, w, label="MoA Debate", color="orange", alpha=0.85, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Bot Intensity")
    ax.set_ylabel("Precision")
    ax.set_title("Precision: MoA vs Single-Shot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for i, (ss, moa) in enumerate(zip(ss_prec, moa_prec)):
        ax.text(i - w/2, ss + 0.02, f"{ss:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + w/2, moa + 0.02, f"{moa:.2f}", ha="center", fontsize=8, fontweight="bold")

    # 2. Recall comparison
    ax = axes[1]
    ss_rec = [r["recall"] for r in ss_results]
    moa_rec = [r["recall"] for r in moa_results]
    ax.bar(x - w/2, ss_rec, w, label="Single-Shot", color="steelblue", alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, moa_rec, w, label="MoA Debate", color="orange", alpha=0.85, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Bot Intensity")
    ax.set_ylabel("Recall")
    ax.set_title("Recall: MoA vs Single-Shot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    for i, (ss, moa) in enumerate(zip(ss_rec, moa_rec)):
        ax.text(i - w/2, ss + 0.02, f"{ss:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + w/2, moa + 0.02, f"{moa:.3f}", ha="center", fontsize=8, fontweight="bold")

    # 3. P&L Saved comparison
    ax = axes[2]
    ss_pnl = [r["total_pnl_saved"] / 1000 for r in ss_results]  # scale to $K
    moa_pnl = [r["total_pnl_saved"] / 1000 for r in moa_results]
    ax.bar(x - w/2, ss_pnl, w, label="Single-Shot", color="steelblue", alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, moa_pnl, w, label="MoA Debate", color="orange", alpha=0.85, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Bot Intensity")
    ax.set_ylabel("P&L Saved ($K)")
    ax.set_title("Economic Impact: MoA vs Single-Shot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for i, (ss, moa) in enumerate(zip(ss_pnl, moa_pnl)):
        y_pos = ss + (1 if ss >= 0 else -3)
        ax.text(i - w/2, y_pos, f"${ss:.1f}k", ha="center", fontsize=8, fontweight="bold")
        y_pos = moa + (1 if moa >= 0 else -3)
        ax.text(i + w/2, y_pos, f"${moa:.1f}k", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Phase 9: MoA Debate vs Single-Shot Robustness Comparison",
                 fontsize=15, fontweight="bold", y=1.02)

    plot_path = os.path.join(output_dir, "moa_vs_singleshot_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MoAPlot] Comparison plot saved to {plot_path}")

    # Also print summary table
    print(f"\n{'='*80}")
    print(f"  MoA vs Single-Shot Robustness Comparison")
    print(f"{'='*80}")
    print(f"  {'Intensity':>10s} | {'Method':>12s} | {'Prec':>6s} | {'Rec':>6s} | {'P&L Saved':>10s}")
    print(f"  {'-'*10} | {'-'*12} | {'-'*6} | {'-'*6} | {'-'*10}")
    for i, bl in enumerate(bot_levels):
        print(f"  {f'{bl:.0%}':>10s} | {'Single-Shot':>12s} | {ss_prec[i]:6.3f} | {ss_rec[i]:6.3f} | ${ss_pnl[i]:>8,.1f}k")
        print(f"  {'':>10s} | {'MoA Debate':>12s} | {moa_prec[i]:6.3f} | {moa_rec[i]:6.3f} | ${moa_pnl[i]:>8,.1f}k")
        imp_prec = moa_prec[i] - ss_prec[i]
        imp_rec = moa_rec[i] - ss_rec[i]
        imp_pnl = moa_pnl[i] - ss_pnl[i]
        arrow_p = "▲" if imp_prec > 0 else ("▼" if imp_prec < 0 else "─")
        arrow_r = "▲" if imp_rec > 0 else ("▼" if imp_rec < 0 else "─")
        arrow_pnl = "▲" if imp_pnl > 0 else ("▼" if imp_pnl < 0 else "─")
        print(f"  {'':>10s} | {'Delta':>12s} | {arrow_p}{abs(imp_prec):.3f} | {arrow_r}{abs(imp_rec):.3f} | {arrow_pnl}${abs(imp_pnl):>7,.1f}k")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MoA Comparison Plot")
    parser.add_argument("--singleshot", default="./output/adversarial_results_singleshot.json")
    parser.add_argument("--moa", default="./output/adversarial_results_moa.json")
    parser.add_argument("--output", default="./output/moa_robustness_plots")
    args = parser.parse_args()

    generate_comparison_plot(
        singleshot_path=args.singleshot,
        moa_path=args.moa,
        output_dir=args.output,
    )
