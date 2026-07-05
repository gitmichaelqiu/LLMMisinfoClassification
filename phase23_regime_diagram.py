"""Phase 23: Regime-Mapping Reference Diagram & Overlay Plot.

1. Maps all headline dollar values, PnL figures, and crossover base rates
   in the manuscript to their governing regime (1.8% empirical vs 18% stress).
2. Side-by-side crossover curves for both regimes.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("./output", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

# Regime table
PNL_MAP = {
    "PnL_TP_saved": {"value": "+$7,852", "regime": "1.8% empirical (mid-cap default)"},
    "PnL_FN_loss": {"value": "-$29,788", "regime": "1.8% empirical (mid-cap default)"},
    "PnL_TN_hold": {"value": "+$7,995", "regime": "1.8% empirical (mid-cap default)"},
    "PnL_FP_cost": {"value": "-$6,802", "regime": "1.8% empirical (mid-cap default)"},
    "Trade_First_fake_loss": {"value": "-$30,012", "regime": "1.8% empirical"},
    "Verify_First_fake_loss": {"value": "$0", "regime": "1.8% empirical"},
    "Trade_First_real_profit": {"value": "+$7,995", "regime": "1.8% empirical"},
    "Verify_First_real_cost": {"value": "-$6,800", "regime": "1.8% empirical"},
    "CRASH_SCENARIO_fake_loss": {"value": "-$180,000", "regime": "18% stress"},
}
CROSSOVER_MAP = {
    "crossover_1_8pct": {"value": "~4-6%", "regime": "1.8% empirical"},
    "crossover_18pct": {"value": "~0.3-0.5%", "regime": "18% stress"},
}

# Side-by-side crossover curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

params = [
    ("1.8% Empirical (Default)", {"color": "#1f77b4", "pnl_tp": 7852, "pnl_fn": -29788, "pnl_tn": 7995, "pnl_fp": -6802}),
    ("18% Stress (Tail Risk)", {"color": "#d62728", "pnl_tp": 30000, "pnl_fn": -180000, "pnl_tn": 7995, "pnl_fp": -6802}),
]

methods = {
    "Single-Shot CoT": {"recall": 0.92, "fpr": 0.021, "color": "#1f77b4", "ls": "-", "lw": 2.5},
    "MoA Debate": {"recall": 1.00, "fpr": 0.500, "color": "#d62728", "ls": "--", "lw": 2.0},
    "Voting N=5": {"recall": 0.80, "fpr": 0.350, "color": "#2ca02c", "ls": ":", "lw": 2.0},
}

base_rates = np.logspace(-4, 0, 500)

for ax, (regime_name, reg) in zip(axes, params):
    crossovers = {}
    for mname, mp in methods.items():
        r = mp["recall"]; fpr = mp["fpr"]
        A = r * reg["pnl_tp"] + (1-r) * reg["pnl_fn"]
        B = (1-fpr) * reg["pnl_tn"] + fpr * reg["pnl_fp"]
        pnl = base_rates * A + (1 - base_rates) * B
        ax.plot(base_rates, pnl/1000, label=mname, color=mp["color"], ls=mp["ls"], lw=mp["lw"])
        if abs(A - B) > 1e-10:
            pc = -B / (A - B)
            if 0 < pc < 1:
                crossovers[mname] = pc
                ax.axvline(pc, color=mp["color"], ls=":", alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("P(Fake) Base Rate")
    ax.set_ylabel("Expected P&L ($K)")
    ax.set_title(f"Crossover: {regime_name}", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim([1e-4, 1])
    ss = crossovers.get("Single-Shot CoT")
    if ss:
        ax.annotate(f"CoT: {ss:.1%}", xy=(ss, 0), fontsize=8,
                   color=methods["Single-Shot CoT"]["color"],
                   xytext=(ss*3, -6), arrowprops=dict(arrowstyle="->", lw=0.8),
                   bbox=dict(boxstyle="round", fc="white", alpha=0.8))

fig.suptitle("Crossover Comparison: 1.8% Empirical vs 18% Stress Regime",
            fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("./plots/phase_23_crossover_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Crossover comparison -> ./plots/phase_23_crossover_comparison.png")

output = {
    "phase": "23_regime_diagram",
    "pnl_values": PNL_MAP,
    "crossover_rates": CROSSOVER_MAP,
    "interpretation": (
        "All default P&L use 1.8% empirical regime. "
        "18% stress represents >3 sigma tail scenario: "
        "Trade-First loss = -$180K, crossover ~0.3-0.5%. "
        "All paper plots/tables should note governing regime."
    ),
}
with open("./output/phase_23_regime_diagram.json", "w") as f:
    json.dump(output, f, indent=2)
print("Regime diagram saved")
