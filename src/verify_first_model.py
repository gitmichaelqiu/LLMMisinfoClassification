"""Verify-First vs. Trade-First Tradeoff Analysis.

This script models the exact economic tradeoff of entering trades late (5 seconds)
on REAL news to verify authenticity ("Verify-First") vs. entering instantly on sentiment
and reversing later if FAKE ("Trade-First"), under realistic microstructure constraints.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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
            # E[P&L_TF] = p * [TPR * pnl_tp_tf + (1-TPR) * c_fn_tf] + (1-p) * [FPR * pnl_fp_tf + (1-FPR) * pnl_tn_tf]
            exp_pnl_tf = p * (tpr * pnl_tp_tf + (1.0 - tpr) * c_fn_tf) + (1.0 - p) * (fpr * pnl_fp_tf + (1.0 - fpr) * pnl_tn_tf)
            pnl_tf_list.append(exp_pnl_tf)
            
            # Expected P&L for Verify-First
            # E[P&L_VF] = p * [TPR * pnl_tp_vf + (1-TPR) * pnl_fn_vf] + (1-p) * [FPR * pnl_fp_vf + (1-FPR) * pnl_tn_vf]
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
    
    # Plot Verify-First Expected P&L (which is the same for all FPRs if we do not trade on FPs)
    # E[P&L_VF] = (1-p) * (1-FPR) * pnl_tn_vf, which is slightly FPR-dependent but mostly flat
    for idx, fpr in enumerate(fpr_values):
        data = results[str(fpr)]
        label_tf = f"Trade-First (FPR={fpr*100:.1f}%)"
        label_vf = f"Verify-First (FPR={fpr*100:.1f}%)"
        
        plt.plot(base_rates, data["pnl_tf"], label=label_tf, color=colors[idx], linestyle='-', linewidth=2)
        plt.plot(base_rates, data["pnl_vf"], label=label_vf, color=colors[idx], linestyle='--', linewidth=2)
        
        cross_p = data["crossover_base_rate"]
        if cross_p:
            # Find index of crossover point to get the y value
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

if __name__ == "__main__":
    run_verify_first_tradeoff()
