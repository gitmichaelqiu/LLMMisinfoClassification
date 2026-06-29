"""Phase 22: Calibrated Microstructure Reconciliation.

1. Fill-ratio trace for the 124x interaction factor (Low-Cap profile)
2. Carry-Cost Sensitivity Surface 2D heatmap
3. Events-per-year and entry price disclosure
"""

import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("./output", exist_ok=True)
os.makedirs("./plots", exist_ok=True)

# ═════════════════════════════════════════════════════════════════
#  1. FILL-RATIO & 124x INTERACTION FACTOR TRACE (LOW-CAP)
# ═════════════════════════════════════════════════════════════════
print("=" * 60)
print("  124x INTERACTION FACTOR TRACE (LOW-CAP)")
print("=" * 60)

# Low-Cap profile: normal_bid_depth=500, min_bid_depth=20
# At T1=5s during panic, depth decays to near min_bid_depth
# Exponential decay: depth(t=5) = 20 + (500-20)*exp(-5/0.8) ≈ 20
V_t1 = 20  # bid depth at T1 for Low-Cap during panic
Q_full = 1000

fill_ratio = Q_full / V_t1
# Original formula: reflexivity = half_spread * min(5.0, fill_ratio * 2.0)
# = half_spread * min(5.0, 50 * 2.0) = half_spread * 5.0
original_reflex_mult = min(5.0, fill_ratio * 2.0)

# With dynamic sizing: Q_cap = min(Q_full, 0.5 * V_t1) = min(1000, 10) = 10
Q_cap = min(Q_full, int(0.5 * V_t1))
fill_ratio_cap = Q_cap / V_t1
capped_reflex_mult = min(5.0, fill_ratio_cap * 2.0)

interaction_factor = original_reflex_mult / max(capped_reflex_mult, 0.01)

print(f"  Low-Cap profile at T1 panic:")
print(f"    Bid depth at T1: V={V_t1}")
print(f"    Full reversal: Q={Q_full} -> fill_ratio={fill_ratio:.1f} -> reflex_mult={original_reflex_mult:.2f}")
print(f"    Dynamic sizing: Q_cap={Q_cap} -> fill_ratio={fill_ratio_cap:.2f} -> reflex_mult={capped_reflex_mult:.2f}")
print(f"    Interaction factor (reflex_mult ratio): {interaction_factor:.2f}x")
print(f"    124x decomposition:")
print(f"      42.4x (execution cost) × {interaction_factor:.2f}x (interaction) = {42.4 * interaction_factor:.0f}x")

trace_result = {
    "profile": "low_cap",
    "bid_depth_at_T1": V_t1,
    "full_reversal_fill_ratio": round(fill_ratio, 2),
    "capped_reversal_fill_ratio": round(fill_ratio_cap, 2),
    "original_reflexivity_multiplier": original_reflex_mult,
    "capped_reflexivity_multiplier": capped_reflex_mult,
    "interaction_factor": round(interaction_factor, 2),
    "execution_cost_improvement": 42.4,
    "resultant_124x": round(42.4 * interaction_factor, 0),
}

# ═════════════════════════════════════════════════════════════════
#  2. CARRY-COST SENSITIVITY SURFACE
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CARRY-COST SENSITIVITY SURFACE")
print("=" * 60)

# Parameters (disclosed for reproducibility)
S0 = 100.0            # entry price per share (mid-cap base)
Q = 1000              # position size
PREMIUM_PCT = 0.005   # option premium: 0.5% of S0
EVENTS_PER_DAY = 50   # average daily events
TRADING_DAYS = 250    # annual trading days
EVENTS_PER_YEAR = EVENTS_PER_DAY * TRADING_DAYS

premium_per_event = S0 * PREMIUM_PCT * Q

print(f"  Disclosed parameters:")
print(f"    Entry price (S0): ${S0}")
print(f"    Position size (Q): {Q} shares")
print(f"    Premium: {PREMIUM_PCT:.1%} of S0 = ${premium_per_event:.2f}/event")
print(f"    Events/day: {EVENTS_PER_DAY}")
print(f"    Trading days/year: {TRADING_DAYS}")
print(f"    Events/year: {EVENTS_PER_YEAR}")

# 2D surface: escalation_frequency x premium_pct -> annual carry cost
frequencies = np.linspace(0.01, 0.50, 25)
premiums = np.linspace(0.001, 0.020, 20)
F, P = np.meshgrid(frequencies, premiums)
C = F * EVENTS_PER_YEAR * S0 * P * Q  # annual carry cost

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.contourf(F * 100, P * 100, C / 1e6, levels=15, cmap="RdYlGn_r")
ax.contour(F * 100, P * 100, C / 1e6, levels=8, colors="black", linewidths=0.5, alpha=0.5)

# Annotate key points
methods = [
    ("CoT (12%)", 0.12, 0.5),
    ("MoA (25%)", 0.25, 0.5),
    ("Voting (18%)", 0.18, 0.5),
]
for label, freq, prem in methods:
    cost = freq * EVENTS_PER_YEAR * S0 * (prem / 100) * Q
    ax.plot(freq * 100, prem, "ko", markersize=8)
    ax.annotate(f"{label}\n${cost/1e6:.2f}M/yr",
               xy=(freq * 100, prem),
               xytext=(freq * 100 + 3, prem + 1),
               fontsize=9, fontweight="bold",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
               arrowprops=dict(arrowstyle="->", lw=0.8))

cbar = plt.colorbar(im, ax=ax, label="Annual Carry Cost ($M)")
ax.set_xlabel("Escalation Frequency (%)", fontsize=11)
ax.set_ylabel("Option Premium (% of S₀)", fontsize=11)
ax.set_title("Carry-Cost Sensitivity Surface: Escalation × Premium → Annual Cost",
            fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)

surface_path = "./plots/phase_22_carry_surface.png"
fig.tight_layout()
fig.savefig(surface_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n  Carry-cost surface -> {surface_path}")

# ═════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════
output = {
    "phase": "22_carry_surface",
    "disclosed_params": {
        "entry_price_S0": S0,
        "position_size_Q": Q,
        "option_premium_pct": PREMIUM_PCT,
        "events_per_day": EVENTS_PER_DAY,
        "trading_days_per_year": TRADING_DAYS,
        "events_per_year": EVENTS_PER_YEAR,
    },
    "fill_ratio_trace": trace_result,
    "carry_surface_params": {
        "frequency_range": [0.01, 0.50],
        "premium_range": [0.001, 0.02],
        "n_frequency_points": 25,
        "n_premium_points": 20,
    },
}

out_path = "./output/phase_22_carry_surface.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nCarry surface -> {out_path}")
