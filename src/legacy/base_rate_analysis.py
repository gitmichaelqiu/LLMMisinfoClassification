"""Base Rate Fallacy Analysis for Dual-System HFT Verifier.

This script analyzes the impact of the Base Rate Fallacy on the verifier's P&L.
In real markets, the base rate of fake news is extremely low (e.g. 1 in 10,000 headlines).
Even with a high F1 score, if the False Positive Rate is non-zero, the verifier will
trigger many false alarms on legitimate news. This script quantifies the Bayesian precision
(Positive Predictive Value) and expected P&L of the system across varying base rates
and False Positive Rates.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def run_base_rate_analysis(output_dir="./output", plots_dir="./plots"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Parameters calibrated from Phase 7a normal regime
    base_price = 190.0
    trough_price = 150.0
    position_size = 1000  # shares

    # Realized values from hft_backtest normal regime ct=0.5
    # TP realized savings: average savings from intervening before trough
    # FP realized cost: average opportunity cost of wrong intervention
    # FN realized cost: full crash loss (riding from base to trough)
    s_tp = 5791.62     # $167,957.10 / 29 TPs
    c_fp = 16874.00    # $33,748.00 / 2 FPs
    c_fn = 40000.00    # position_size * (base_price - trough_price) = $40,000
    tpr = 1.0          # Recall from Phase 7a

    # Range of base rates (from 1 in 1,000,000 to 1 in 10)
    base_rates = np.logspace(-6, -1, 100)

    # False Positive Rates to sweep
    # 0.10: Phase 7a normal regime (FPR = 9.5%)
    # 0.05: Moderate verifier
    # 0.021: FinFakeBERT (2025) reported FPR
    # 0.01: Highly optimized verifier
    # 0.001: Near-perfect verifier
    fpr_values = [0.10, 0.05, 0.021, 0.01, 0.001]

    results = {}

    for fpr in fpr_values:
        ppv_list = []
        net_pnl_list = []
        crossover_p = None

        for p in base_rates:
            # 1. Bayesian Precision (PPV)
            # PPV = P(Fake | Alert) = P(Alert | Fake) * P(Fake) / P(Alert)
            # P(Alert) = P(Alert | Fake) * P(Fake) + P(Alert | Real) * P(Real)
            # PPV = TPR * p / (TPR * p + FPR * (1 - p))
            denom = (tpr * p) + (fpr * (1 - p))
            ppv = (tpr * p) / denom if denom > 0 else 0.0
            ppv_list.append(ppv)

            # 2. Expected P&L added by verifier per headline
            # E[P&L_verifier] = p * TPR * S_TP - (1-p) * FPR * C_FP - p * (1-TPR) * C_FN
            # E[P&L_baseline] = - p * C_FN
            # Net P&L added = E[P&L_verifier] - E[P&L_baseline]
            #               = p * TPR * (S_TP + C_FN) - (1-p) * FPR * C_FP
            net_pnl = p * tpr * (s_tp + c_fn) - (1.0 - p) * fpr * c_fp
            # Scale to per 10,000 headlines for readability
            net_pnl_10k = net_pnl * 10000
            net_pnl_list.append(net_pnl_10k)

            # Crossover point: first base rate where Net P&L becomes positive
            if crossover_p is None and net_pnl_10k > 0:
                crossover_p = p

        results[str(fpr)] = {
            "base_rates": base_rates.tolist(),
            "ppv": ppv_list,
            "net_pnl_10k": net_pnl_list,
            "crossover_base_rate": crossover_p,
            "crossover_headlines_ratio": f"1 in {int(1/crossover_p):,}" if crossover_p else "N/A"
        }

    # Save results to JSON
    output_path = os.path.join(output_dir, "base_rate_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Base rate analysis JSON saved to {output_path}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Color palette
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

    for idx, fpr in enumerate(fpr_values):
        fpr_data = results[str(fpr)]
        label = f"FPR = {fpr*100:.1f}%"
        if fpr == 0.021:
            label += " (FinFakeBERT)"
        elif fpr == 0.10:
            label += " (Our Phase 7a)"

        ax1.plot(base_rates, fpr_data["ppv"], label=label, color=colors[idx], linewidth=2)
        ax2.plot(base_rates, fpr_data["net_pnl_10k"], label=label, color=colors[idx], linewidth=2)

        # Plot crossover markers
        cross_p = fpr_data["crossover_base_rate"]
        if cross_p:
            ax2.scatter([cross_p], [0.0], color=colors[idx], s=50, zorder=5)
            # Label crossover point
            ax2.annotate(
                f"1 in {int(1/cross_p):,}",
                xy=(cross_p, 0.0),
                xytext=(cross_p * 1.5, 0.05 * (max(ax2.get_ylim()) or 1) + 20000 * (idx - 2)),
                arrowprops=dict(arrowstyle="->", color=colors[idx], alpha=0.6),
                fontsize=8,
                color=colors[idx],
                fontweight='bold'
            )

    # Subplot 1: Bayesian Precision
    ax1.set_xscale('log')
    ax1.set_xlabel("Base Rate of Fake News Headlines (Log Scale)", fontsize=11)
    ax1.set_ylabel("Bayesian Precision / PPV\n(P(Fake | verifier Alert))", fontsize=11)
    ax1.set_title("Bayesian Precision vs. Fake News Base Rate", fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.set_ylim(-0.05, 1.05)

    # Subplot 2: Net P&L
    ax2.set_xscale('log')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel("Base Rate of Fake News Headlines (Log Scale)", fontsize=11)
    ax2.set_ylabel("Expected Net P&L Added per 10,000 Headlines ($)", fontsize=11)
    ax2.set_title("Verifier Expected Net P&L Added vs. Base Rate", fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    plt.suptitle("The Base Rate Fallacy in HFT Fake News Detection (1,000 Share Position)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "base_rate_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Base rate analysis plot saved to {plot_path}")

if __name__ == "__main__":
    run_base_rate_analysis()
