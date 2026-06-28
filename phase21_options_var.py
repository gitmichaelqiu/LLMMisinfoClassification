"""Phase 21: Options Cost-of-Carry & VaR Precision.

Refines:
1. Expected P&L model incorporating amortized option premium cost-of-carry,
   scaled by the frequency of events landing in the escalation band.
2. VaR circuit breaker trigger defined by sector membership and 30-day
   historical correlation (rho > 0.5).

Output:
  - output/phase_21_options_var_analysis.json
"""

import os, json, math
import numpy as np

os.makedirs("./output", exist_ok=True)

# ═════════════════════════════════════════════════════════════════
#  1. OPTIONS COST-OF-CARRY
# ═════════════════════════════════════════════════════════════════
print("=" * 60)
print("  OPTIONS COST-OF-CARRY ANALYSIS")
print("=" * 60)

# Parameters
PREMIUM_PCT = 0.005       # 0.5% of S0
S0 = 100.0                 # entry price
Q = 1000                   # position size
T1 = 5.0                   # verification window in seconds
T2 = 300.0                 # human verification in seconds

# Expected escalation frequency: proportion of events where confidence
# falls in the HEDGE zone [0.35, 0.65]
# From Phase 7a empirical: ~15% of events land in escalation band
# (this varies by verifier method)
ESCALATION_FREQUENCIES = {
    "single_shot_cot": 0.12,   # 12% of events
    "moa_debate": 0.25,        # 25% (MoA is more uncertain)
    "voting_n5": 0.18,         # 18%
    "average": 0.15,           # 15% blended
}

# Premium cost per event
premium_per_event = S0 * PREMIUM_PCT * Q
print(f"  Premium per hedged event: ${premium_per_event:,.2f}")
print(f"  (Premium = {PREMIUM_PCT:.1%} × ${S0} × {Q} shares)")

# Annualized carry
# Assuming N events/day, ~250 trading days/year
N_EVENTS_PER_DAY = 50
TRADING_DAYS = 250
total_events_year = N_EVENTS_PER_DAY * TRADING_DAYS
escalation_events_year = total_events_year * ESCALATION_FREQUENCIES["average"]
annual_premium_cost = escalation_events_year * premium_per_event

print(f"\n  Estimated escalation events/year: {escalation_events_year:.0f}")
print(f"  Annual premium cost: ${annual_premium_cost:,.2f}")

# Cost-of-carry analysis for different escalation frequencies
carry_results = {}
for method, freq in ESCALATION_FREQUENCIES.items():
    n_hedged = total_events_year * freq
    cost = n_hedged * premium_per_event
    carry_results[method] = {
        "escalation_frequency": freq,
        "n_hedged_events_year": round(n_hedged),
        "annual_premium_cost": round(cost, 2),
        "premium_per_event": round(premium_per_event, 2),
    }
    print(f"  {method:20s}: freq={freq:.0%}  annual_premium=${cost:>9,.2f}")

# Sensitivity: premium cost vs escalation frequency
frequencies = np.linspace(0.01, 0.50, 20)
carry_sensitivity = []
for f in frequencies:
    carry_sensitivity.append({
        "escalation_frequency": round(f, 4),
        "n_events_hedged": round(total_events_year * f),
        "annual_cost": round(total_events_year * f * premium_per_event, 2),
    })

# ═════════════════════════════════════════════════════════════════
#  2. VAR CIRCUIT BREAKER PRECISION
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  VaR CIRCUIT BREAKER PRECISION")
print("=" * 60)

# Sector correlation matrix (simplified)
# Correlation threshold for breaker trigger: rho > 0.5
CORR_THRESHOLD = 0.50

SECTOR_MEMBERSHIP = {
    "Tech": {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "ORCL", "IBM", "CRM", "ADBE"},
    "Finance": {"JPM", "GS", "BAC", "C", "WFC", "MS", "SCHW", "BLK"},
    "Energy": {"XOM", "CVX", "COP", "EOG", "SLB"},
    "Healthcare": {"PFE", "MRNA", "JNJ", "ABBV", "MRK", "UNH"},
    "Retail": {"WMT", "HD", "NKE", "TGT", "COST"},
    "Crypto": {"BTC", "ETH", "SOL", "USDT"},
}

# Estimated 30-day rolling correlations between sectors (simplified)
SECTOR_CORRELATIONS = {
    ("Tech", "Finance"): 0.45,
    ("Tech", "Energy"): 0.30,
    ("Tech", "Healthcare"): 0.35,
    ("Tech", "Retail"): 0.55,     # > rho threshold -> correlated
    ("Finance", "Energy"): 0.40,
    ("Finance", "Healthcare"): 0.35,
    ("Finance", "Retail"): 0.50,   # at threshold
    ("Energy", "Healthcare"): 0.25,
    ("Energy", "Retail"): 0.30,
    ("Healthcare", "Retail"): 0.30,
    ("Crypto", "Tech"): 0.20,
    ("Crypto", "Finance"): 0.15,
}

def check_correlation_trigger(asset_a, asset_b):
    """Check if two assets trigger the VaR circuit breaker (rho > 0.5 + same sector)."""
    # Find sectors for each asset
    sector_a = next((s for s, members in SECTOR_MEMBERSHIP.items() if asset_a in members), None)
    sector_b = next((s for s, members in SECTOR_MEMBERSHIP.items() if asset_b in members), None)

    if not sector_a or not sector_b:
        return {"triggered": False, "reason": "unknown_sector"}

    # Same sector -> always correlated
    if sector_a == sector_b:
        return {
            "triggered": True,
            "reason": f"same_sector({sector_a})",
            "correlation": 1.0,
            "sector_a": sector_a,
            "sector_b": sector_b,
        }

    # Different sectors -> check 30-day correlation
    corr = SECTOR_CORRELATIONS.get((sector_a, sector_b),
           SECTOR_CORRELATIONS.get((sector_b, sector_a), 0.0))

    if corr > CORR_THRESHOLD:
        return {
            "triggered": True,
            "reason": f"correlation={corr:.2f}>{CORR_THRESHOLD}",
            "correlation": corr,
            "sector_a": sector_a,
            "sector_b": sector_b,
        }

    return {
        "triggered": False,
        "reason": f"correlation={corr:.2f}<={CORR_THRESHOLD}",
        "correlation": corr,
        "sector_a": sector_a,
        "sector_b": sector_b,
    }


# Test asset pairs
test_pairs = [
    ("AAPL", "MSFT"),    # same sector (Tech) -> triggered
    ("JPM", "GS"),       # same sector (Finance) -> triggered
    ("AAPL", "WMT"),     # different sector, high corr -> triggered
    ("AAPL", "JPM"),     # different sector, low corr -> not triggered
    ("BTC", "AAPL"),     # crypto vs tech, very low corr -> not triggered
    ("PFE", "MRK"),      # same sector (Healthcare) -> triggered
]

var_results = []
for a, b in test_pairs:
    result = check_correlation_trigger(a, b)
    var_results.append({"asset_a": a, "asset_b": b, **result})
    status = "✅ TRIGGERED" if result["triggered"] else "❌ NOT triggered"
    corr = result.get("correlation", "N/A")
    print(f"  {a:5s} vs {b:5s}: {status} (corr={corr}, reason={result['reason']})")

# ═════════════════════════════════════════════════════════════════
#  SAVE
# ═════════════════════════════════════════════════════════════════
output = {
    "phase": "21_options_var",
    "options_cost_of_carry": {
        "premium_pct": PREMIUM_PCT,
        "premium_per_event": round(premium_per_event, 2),
        "annual_premium_by_method": carry_results,
        "annual_premium_average": round(
            total_events_year * ESCALATION_FREQUENCIES["average"] * premium_per_event, 2
        ),
        "carry_sensitivity": carry_sensitivity,
        "assumptions": {
            "events_per_day": N_EVENTS_PER_DAY,
            "trading_days": TRADING_DAYS,
            "position_size": Q,
            "entry_price": S0,
        },
    },
    "var_circuit_breaker": {
        "correlation_threshold": CORR_THRESHOLD,
        "asset_pair_tests": var_results,
        "estimated_sector_correlations": {
            f"{s1}-{s2}": c for (s1, s2), c in SECTOR_CORRELATIONS.items()
        },
    },
}

output_path = "./output/phase_21_options_var_analysis.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nOptions & VaR analysis -> {output_path}")
print("Done.")
