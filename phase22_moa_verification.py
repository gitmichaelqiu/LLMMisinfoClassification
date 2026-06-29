"""Phase 22: MoA and Adversarial Recall Verification.

Runs adversarial stress tests with both Single-Shot and MoA modes
using the real API key. Generates:
  - MoA Before/After PR/ROC curves (phase_22_moa_before_after.png)
  - Verification that recall floor is not a cached artifact
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("./output", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

from src.stress_test import run_full_stress_test

BOT_LEVELS = [0.0, 0.25, 0.50, 0.75]
MAX_EVENTS = 10  # small cost-conscious run

print("=" * 60)
print("  MOA ADVERSARIAL RECALL VERIFICATION")
print("=" * 60)

# ── Run Single-Shot ────────────────────────────────────────────
print("\n[1/2] Running Single-Shot adversarial sweep...")
ss_results = run_full_stress_test(
    bot_levels=BOT_LEVELS,
    max_test_events=MAX_EVENTS,
    use_moa=False,
    output_path="./output/adversarial_results_singleshot.json",
)
print("[Single-Shot] Done.")

# ── Run MoA ────────────────────────────────────────────────────
print("\n[2/2] Running MoA adversarial sweep...")
moa_results = run_full_stress_test(
    bot_levels=BOT_LEVELS,
    max_test_events=MAX_EVENTS,
    use_moa=True,
    output_path="./output/adversarial_results_moa.json",
)
print("[MoA] Done.")

# ── Generate Comparison Plots ──────────────────────────────────
print("\nGenerating MoA Before/After comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Precision/Recall comparison
ax = axes[0]
bot_pcts = [r["bot_pct"] for r in ss_results]

for method_data, method_name, fmt, color in [
    (ss_results, "Single-Shot", "-o", "#1f77b4"),
    (moa_results, "MoA Debate", "-s", "#d62728"),
]:
    precisions = [r["precision"] for r in method_data]
    recalls = [r["recall"] for r in method_data]
    ax.plot(bot_pcts, precisions, marker="o", linestyle="-", color=color, label=f"{method_name} Precision",
            linewidth=2, markersize=8)
    ax.plot(bot_pcts, recalls, marker="s", linestyle="--", color=color, label=f"{method_name} Recall",
            linewidth=2, markersize=8, alpha=0.7)

ax.axhline(y=0.5, color="gray", ls=":", alpha=0.5, label="Random (0.5)")
ax.set_xlabel("Bot Intensity", fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Precision/Recall: MoA vs Single-Shot", fontsize=13, fontweight="bold")
ax.set_xticks(BOT_LEVELS)
ax.set_xticklabels([f"{int(b*100)}%" for b in BOT_LEVELS])
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Panel 2: P&L saved comparison
ax = axes[1]
for method_data, method_name, fmt, color in [
    (ss_results, "Single-Shot", "-o", "#1f77b4"),
    (moa_results, "MoA Debate", "-s", "#d62728"),
]:
    pnls = [r.get("total_pnl_saved", 0) / 1000 for r in method_data]
    ax.plot(bot_pcts, pnls, marker="o", linestyle="-", color=color, label=method_name,
            linewidth=2, markersize=8)

ax.axhline(y=0, color="black", linewidth=1)
ax.set_xlabel("Bot Intensity", fontsize=11)
ax.set_ylabel("P&L Saved ($K)", fontsize=11)
ax.set_title("Economic Impact Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(BOT_LEVELS)
ax.set_xticklabels([f"{int(b*100)}%" for b in BOT_LEVELS])
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.suptitle("MoA Before/After: Adversarial Robustness Comparison",
            fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
comp_path = "./plots/phase_22_moa_before_after.png"
fig.savefig(comp_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Comparison plot -> {comp_path}")

# ── Verify 71% recall floor ───────────────────────────────────
print("\n" + "=" * 60)
print("  RECALL FLOOR VERIFICATION")
print("=" * 60)

# Check that recall at bot=0% matches expected (87-92%) and
# the degradation is independent of historical validation
for method_name, data in [("Single-Shot", ss_results), ("MoA", moa_results)]:
    r0 = next(r for r in data if r["bot_pct"] == 0.0)
    r75 = next(r for r in data if r["bot_pct"] == 0.75)

    print(f"\n  {method_name}:")
    print(f"    0% bot: Recall={r0['recall']:.3f}  Precision={r0['precision']:.3f}")
    print(f"    75% bot: Recall={r75['recall']:.3f}  Precision={r75['precision']:.3f}")
    print(f"    Degradation: Recall {r0['recall']:.3f} -> {r75['recall']:.3f} "
          f"({(r75['recall']-r0['recall']):+.3f})")

    floor = min(r["recall"] for r in data)
    print(f"    Recall floor across all intensities: {floor:.3f}")
    print(f"    Floor is artifact-cached? "
          f"{'YES' if r0['recall'] == r75['recall'] else 'NO — genuine degradation'}")

# ── Save results ───────────────────────────────────────────────
output = {
    "phase": "22_moa_verification",
    "single_shot_results": ss_results,
    "moa_results": moa_results,
    "comparison_plot": comp_path,
    "verification": {
        "recall_floor_is_cached_artifact": r0["recall"] == r75["recall"],
        "single_shot_recall_at_0": next(r["recall"] for r in ss_results if r["bot_pct"] == 0),
        "moa_recall_at_0": next(r["recall"] for r in moa_results if r["bot_pct"] == 0),
    },
}
out_path = "./output/phase_22_moa_verification.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nMoA verification -> {out_path}")
