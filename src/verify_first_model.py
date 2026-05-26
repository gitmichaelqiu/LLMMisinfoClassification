"""Verify-First vs. Trade-First Tradeoff Analysis.

This script models the exact economic tradeoff of entering trades late (5 seconds)
on REAL news to verify authenticity ("Verify-First") vs. entering instantly on sentiment
and reversing later if FAKE ("Trade-First"), under realistic microstructure constraints.
It includes a base rate sweep and a head-to-head liquidity sensitivity sweep.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.l2_data_generator import FlashCrashL2Config, compute_realized_pnl, compute_ideal_pnl
from src.sensitivity_analysis import latin_hypercube_sampling


def run_verify_first_tradeoff(output_dir="./output", plots_dir="./plots"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Cost parameters calibrated from Phase 7a normal regime
    position_size = 1000
    base_price = 190.0
    trough_price = 150.0
    
    # Trade-First Constants
    c_fn_tf = -position_size * (base_price - trough_price)  # -$40,000 full crash loss
    s_tp_tf = 5791.62     # realized savings on correct intervention
    pnl_tp_tf = c_fn_tf + s_tp_tf  # -$34,208.38 net realized P&L on TP
    pnl_fp_tf = -16874.00  # realized cost on false positive (opportunity + spread)
    pnl_tn_tf = 10000.00   # profit captured on instant real news entry
    
    # Verify-First Constants
    pnl_tp_vf = 0.00       # correct non-entry on fake news ($0 exposure)
    pnl_fp_vf = 0.00       # correct non-entry on real news (missed profit, $0)
    # Late entry price at 5s during linear crash recovery: 
    # At 5s, price is $161.43. Shorting late at $161.43 and covering at $190.00
    # leads to a loss of -$28,570.
    pnl_fn_vf = -position_size * (base_price - 161.43)
    pnl_tn_vf = 5000.00    # late real news entry captures only 50% of the $10,000 profit

    tpr = 1.0              # Verifier recall (TPR) from Phase 7a

    # Range of fake news base rates (log space from 1 in 100,000 to 1 in 10)
    base_rates = np.logspace(-5, -1, 100)
    
    # False Positive Rates to sweep
    fpr_values = [0.0952, 0.05, 0.021, 0.01, 0.001]
    
    results = {}

    for fpr in fpr_values:
        pnl_tf_list = []
        pnl_vf_list = []
        crossover_p = None
        
        for p in base_rates:
            # Expected P&L for Trade-First
            exp_pnl_tf = p * (tpr * pnl_tp_tf + (1.0 - tpr) * c_fn_tf) + (1.0 - p) * (fpr * pnl_fp_tf + (1.0 - fpr) * pnl_tn_tf)
            pnl_tf_list.append(exp_pnl_tf)
            
            # Expected P&L for Verify-First
            exp_pnl_vf = p * (tpr * pnl_tp_vf + (1.0 - tpr) * pnl_fn_vf) + (1.0 - p) * (fpr * pnl_fp_vf + (1.0 - fpr) * pnl_tn_vf)
            pnl_vf_list.append(exp_pnl_vf)
            
            # Crossover: first base rate where Verify-First outperforms Trade-First
            if crossover_p is None and exp_pnl_vf > exp_pnl_tf:
                crossover_p = p
                
        results[str(fpr)] = {
            "base_rates": base_rates.tolist(),
            "pnl_tf": pnl_tf_list,
            "pnl_vf": pnl_vf_list,
            "crossover_base_rate": crossover_p,
            "crossover_headlines_ratio": f"1 in {int(1/crossover_p):,}" if crossover_p else "N/A"
        }

    # Save to JSON
    output_path = os.path.join(output_dir, "verify_first_tradeoff.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Verify-First tradeoff JSON saved to {output_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot expected P&L of Trade-First for different FPRs
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    
    for idx, fpr in enumerate(fpr_values):
        data = results[str(fpr)]
        label_tf = f"Trade-First (FPR={fpr*100:.1f}%)"
        label_vf = f"Verify-First (FPR={fpr*100:.1f}%)"
        
        plt.plot(base_rates, data["pnl_tf"], label=label_tf, color=colors[idx], linestyle='-', linewidth=2)
        plt.plot(base_rates, data["pnl_vf"], label=label_vf, color=colors[idx], linestyle='--', linewidth=2)
        
        cross_p = data["crossover_base_rate"]
        if cross_p:
            cross_idx = np.argmin(np.abs(np.array(data["base_rates"]) - cross_p))
            cross_y = data["pnl_vf"][cross_idx]
            plt.scatter([cross_p], [cross_y], color=colors[idx], s=70, zorder=5)
            plt.annotate(
                f"Crossover: 1 in {int(1/cross_p):,}\n({cross_p*100:.2f}%)",
                xy=(cross_p, cross_y),
                xytext=(cross_p * 1.5, cross_y - 2000 - 3000 * idx),
                arrowprops=dict(arrowstyle="->", color=colors[idx], alpha=0.6),
                fontsize=8,
                fontweight='bold',
                color=colors[idx]
            )

    plt.xscale('log')
    plt.xlabel("Base Rate of Fake News Headlines (Log Scale)", fontsize=11)
    plt.ylabel("Expected P&L per Headline ($)", fontsize=11)
    plt.title("Trade-First vs. Verify-First: Architectural Tradeoff under L2 Stress", fontsize=12, fontweight='bold')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "verify_first_tradeoff.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Verify-First tradeoff plot saved to {plot_path}")


def run_liquidity_tradeoff_sweep(output_dir="./output", plots_dir="./plots", n_samples=30):
    """Head-to-head P&L sweep comparing Trade-First vs. Verify-First across Phase 8 liquidity parameters."""
    print("\n[Phase 8] Running head-to-head Trade-First vs. Verify-First liquidity tradeoff sweep...")
    
    # Load detection results from Phase 7 (without System 0 filter to get clean evaluations)
    p7_path = os.path.join(output_dir, "phase7_detection_results.json")
    if not os.path.exists(p7_path):
        print(f"  Warning: {p7_path} not found. Sourcing mock detection results.")
        # Fallback synthetic detection results for testing
        results = []
        for i in range(50):
            actual = 1 if i < 25 else 0
            results.append({
                "actual": actual,
                "verdict": actual,
                "confidence": 0.85 if actual == 1 else 0.15,
                "intervention_time_ms": 1200.0 if actual == 1 else -1.0
            })
    else:
        with open(p7_path, "r") as f:
            data = json.load(f)
            results = data["results"]

    # Parameter Bounds for LHS Sweep
    bounds = {
        "depth_decay_ms": (200.0, 5000.0),
        "normal_bid_depth": (100.0, 10000.0),
        "min_fill_prob": (0.01, 0.5),
        "position_size": (100.0, 10000.0),
    }
    log_dims = {"normal_bid_depth", "position_size"}

    samples = latin_hypercube_sampling(n_samples, bounds, log_scale_dims=log_dims, seed=42)

    sweep_results = []
    crossover_point = None

    for i, params in enumerate(samples):
        # Format parameters
        params["normal_bid_depth"] = int(round(params["normal_bid_depth"]))
        params["position_size"] = int(round(params["position_size"]))
        
        config = FlashCrashL2Config(
            base_price=190.0,
            depth_decay_ms=params["depth_decay_ms"],
            normal_bid_depth=params["normal_bid_depth"],
            min_fill_prob=params["min_fill_prob"]
        )

        pnl_tf_total = 0.0
        pnl_vf_total = 0.0

        for r in results:
            actual = r["actual"]
            verdict = r["verdict"]
            confidence = r["confidence"]
            inter_time = r["intervention_time_ms"]
            
            intervened = (verdict == 1 and confidence >= 0.5)
            ps = params["position_size"]

            # 1. Trade-First Architecture
            if actual == 1 and intervened:
                # Sized reversal trade P&L
                reversal_ps = min(ps, config.bid_depth_at(inter_time))
                realized = compute_realized_pnl(inter_time, reversal_ps, config, seed=42+i)
                pnl_tf_total += realized["pnl"]
            elif actual == 0 and intervened:
                # Opportunity cost on sized false alarm
                reversal_ps = min(ps, config.bid_depth_at(inter_time))
                p_intervention = config.price_at(inter_time)
                pnl_tf_total -= reversal_ps * max(0.0, 190.0 - p_intervention) * 0.5
            elif actual == 1 and not intervened:
                # FN: Ride full crash
                pnl_tf_total -= ps * (190.0 - 150.0)
            else:
                # TN: Capture normal profit on instant entry
                pnl_tf_total += ps * (190.0 - 150.0) * 0.25

            # 2. Verify-First Architecture
            if actual == 1 and intervened:
                # Correct non-entry on fake news ($0 crash exposure)
                pnl_vf_total += 0.0
            elif actual == 0 and intervened:
                # FP: AI says it's fake, we do NOT enter (missed profit)
                pnl_vf_total += 0.0
            elif actual == 1 and not intervened:
                # FN: Ride full crash
                pnl_vf_total -= ps * (190.0 - 150.0)
            else:
                # TN: Enter 5 seconds late, suffering severe slippage
                late_realized = compute_realized_pnl(5000.0, ps, config, seed=42+i)
                pnl_vf_total += late_realized["pnl"]

        capacity_ratio = params["position_size"] / params["normal_bid_depth"]
        
        sweep_results.append({
            "sample": i,
            "capacity_ratio": round(capacity_ratio, 4),
            "pnl_tf": round(pnl_tf_total, 2),
            "pnl_vf": round(pnl_vf_total, 2),
            "pnl_diff": round(pnl_vf_total - pnl_tf_total, 2),
            **params
        })

    # Sort results by capacity ratio to find crossover
    sweep_results.sort(key=lambda x: x["capacity_ratio"])
    
    # Identify crossover capacity ratio
    for r in sweep_results:
        if crossover_point is None and r["pnl_vf"] > r["pnl_tf"]:
            crossover_point = r["capacity_ratio"]

    # Save to JSON
    output_path = os.path.join(output_dir, "verify_first_liquidity_sweep.json")
    with open(output_path, "w") as f:
        json.dump({
            "crossover_capacity_ratio": crossover_point or "N/A",
            "results": sweep_results
        }, f, indent=2)
    print(f"Verify-First liquidity sweep JSON saved to {output_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(sweep_results)
    
    plt.plot(df["capacity_ratio"], df["pnl_tf"], label="Trade-First (Instant Entry)", color="steelblue", marker="o", linewidth=2)
    plt.plot(df["capacity_ratio"], df["pnl_vf"], label="Verify-First (5s Delay)", color="crimson", marker="x", linestyle="--", linewidth=2)
    
    if crossover_point is not None:
        plt.axvline(crossover_point, color="purple", linestyle=":", label=f"Crossover point: {crossover_point:.2f}x")
        # Find corresponding y coordinate for annotation
        cross_idx = df[df["capacity_ratio"] >= crossover_point].index[0]
        plt.annotate(
            f"Liquidity Trap Zone\n(Verify-First Dominates)",
            xy=(crossover_point * 1.2, df["pnl_vf"].iloc[cross_idx]),
            xytext=(crossover_point * 2.0, df["pnl_vf"].iloc[cross_idx] - 100000),
            arrowprops=dict(arrowstyle="->", color="purple"),
            fontweight="bold",
            color="purple"
        )
        
    plt.xscale("log")
    plt.xlabel("Capacity Ratio (Position Size / Normal Bid Depth)", fontsize=11)
    plt.ylabel("Total Realized P&L ($)", fontsize=11)
    plt.title("Trade-First vs. Verify-First: Liquidity Capacity Crossover", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "verify_first_liquidity_sweep.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Verify-First liquidity sweep plot saved to {plot_path}")


if __name__ == "__main__":
    run_verify_first_tradeoff()
    run_liquidity_tradeoff_sweep()
