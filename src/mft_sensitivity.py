"""MFT Sensitivity Analysis using Latin Hypercube Sampling (LHS).

Quantifies which parameters most influence the P&L variance in the
MFT verification arbitrage framework.

Parameters swept:
- T₁ (Intervention Latency): 2s to 30s
- T₂ (Human Verification Delay): 60s to 600s
- FP Cost Multiplier: 0.5 to 5.0

Method:
1. Generate N samples across the parameter space using LHS.
2. For each sample, replay pipeline results through a simulator
   configured with those parameter values.
3. Compute partial rank correlation coefficients (PRCC) to identify
   dominant drivers of P&L variance.
4. Generate tornado charts and heatmaps.

Reference: Phase 7 of CLAUDE.md
"""

import os
import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class SensitivitySample:
    """Single LHS sample with computed metrics."""
    t1: float          # Intervention latency (s)
    t2: float          # Human verification delay (s)
    fp_cost_mult: float  # FP cost multiplier
    total_pnl: float = 0.0
    total_pnl_saved: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0


def latin_hypercube_sample(n_samples, bounds, seed=42):
    """Generate LHS samples in the [0,1] hypercube, then scale to parameter bounds.

    Args:
        n_samples: Number of samples to generate
        bounds: dict of {param_name: (lower, upper)}
        seed: Random seed

    Returns:
        pd.DataFrame with columns for each parameter
    """
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    samples = sampler.random(n=n_samples)

    param_names = list(bounds.keys())
    scaled = qmc.scale(samples, [bounds[p][0] for p in param_names],
                       [bounds[p][1] for p in param_names])

    return pd.DataFrame(scaled, columns=param_names)


def _evaluate_sample(params, pipeline_results):
    """Evaluate P&L for a single sample of (t1, t2, fp_cost_mult).

    Uses the default simulator for price/liquidity curves but varies
    the intervention and verification timing.

    Args:
        params: dict with keys t1, t2, fp_cost_mult
        pipeline_results: list of dicts from MFTPipeline

    Returns:
        SensitivitySample
    """
    from src.mft_simulator import MFTMarketSimulator

    t1 = params["t1"]
    t2 = params["t2"]
    fp_cost_mult = params["fp_cost_mult"]
    pos = 1000

    # Simulator with DEFAULT timeline (T1=5, T2=300 for market dynamics).
    # We query price_at(t) at arbitrary t to get prices for custom timelines.
    sim = MFTMarketSimulator(base_price=100.0, position_size=pos)

    tp = fp = tn = fn = 0
    total_pnl = 0.0
    total_pnl_saved = 0.0

    for row in pipeline_results:
        is_fake = row.get("is_fake_event")
        llm_verdict = row["llm_verdict"]
        llm_intervened = (llm_verdict == 1)
        llm_escalated = (llm_verdict == 2)

        if is_fake is None:
            continue

        entry_price = 100.0

        # Hold P&L: exit at t2 + small delay
        exit_time = t2 + 2.0
        mid_exit = sim.price_at(exit_time, is_fake=is_fake)
        spread_bps = sim.spread_at(exit_time, is_fake=is_fake)
        half_spread = mid_exit * spread_bps / 10000.0
        if is_fake:
            slippage_multiple = 3.0
        else:
            slippage_multiple = 1.0
        slippage_cost = half_spread * slippage_multiple
        hold_exit_price = mid_exit - slippage_cost
        hold_pnl = (hold_exit_price - entry_price) * pos

        # Intervene P&L: reverse at t1
        mid_t1 = sim.price_at(t1, is_fake=is_fake)
        spread_bps_t1 = sim.spread_at(t1, is_fake=is_fake)
        half_spread_t1 = mid_t1 * spread_bps_t1 / 10000.0
        slippage_cost_t1 = half_spread_t1
        bid_depth = sim.bid_depth_at(t1, is_fake=is_fake)
        if bid_depth > 0:
            fill_ratio = pos / bid_depth
            reflexivity_penalty = half_spread_t1 * min(5.0, fill_ratio * 2.0)
        else:
            fill_ratio = float("inf")
            reflexivity_penalty = half_spread_t1 * 5.0
        total_cost = slippage_cost_t1 + reflexivity_penalty
        intervene_exit_price = mid_t1 - total_cost
        intervene_pnl = (intervene_exit_price - entry_price) * pos

        if llm_intervened and is_fake:
            actual_pnl = intervene_pnl
            pnl_saved = intervene_pnl - hold_pnl
            tp += 1
        elif llm_intervened and not is_fake:
            real_price_t1 = sim.price_at(t1, is_fake=False)
            real_price_t2 = sim.price_at(t2, is_fake=False)
            missed = (real_price_t2 - real_price_t1) * pos * fp_cost_mult
            actual_pnl = intervene_pnl
            pnl_saved = intervene_pnl - hold_pnl
            fp += 1
        elif not llm_intervened and is_fake:
            actual_pnl = hold_pnl
            pnl_saved = 0.0
            fn += 1
        else:
            actual_pnl = hold_pnl
            pnl_saved = 0.0
            tn += 1

        total_pnl += actual_pnl
        total_pnl_saved += pnl_saved

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return SensitivitySample(
        t1=round(t1, 2),
        t2=round(t2, 2),
        fp_cost_mult=round(fp_cost_mult, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_saved=round(total_pnl_saved, 2),
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
    )


def run_lhs_sensitivity(
    pipeline_results_path=None,
    pipeline_results=None,
    n_samples=200,
    seed=42,
    output_path="./output/mft_sensitivity_results.json",
):
    """Run full LHS sensitivity analysis.

    Args:
        pipeline_results_path: Path to cached pipeline results JSON (optional)
        pipeline_results: Direct list of pipeline result dicts (optional)
        n_samples: Number of LHS samples
        seed: Random seed
        output_path: Output JSON path

    Returns:
        list of SensitivitySample
    """
    # Load pipeline results
    if pipeline_results is None and pipeline_results_path:
        import json
        with open(pipeline_results_path) as f:
            data = json.load(f)
            pipeline_results = data.get("per_sample", data.get("results", []))
    elif pipeline_results is None:
        # Build from a fresh run
        raise ValueError("Must provide either pipeline_results or pipeline_results_path")

    print(f"Loaded {len(pipeline_results)} pipeline results")

    # Define parameter bounds
    bounds = {
        "t1": (2.0, 30.0),
        "t2": (60.0, 600.0),
        "fp_cost_mult": (0.5, 5.0),
    }

    # Generate LHS samples
    samples_df = latin_hypercube_sample(n_samples, bounds, seed=seed)
    print(f"Generated {len(samples_df)} LHS samples")

    # Evaluate each sample
    results = []
    for i, (_, sample) in enumerate(samples_df.iterrows()):
        sr = _evaluate_sample(sample.to_dict(), pipeline_results)
        results.append(sr)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n_samples}] t1={sr.t1:.1f}s, t2={sr.t2:.0f}s, "
                  f"fp_mult={sr.fp_cost_mult:.1f} → "
                  f"P&L=${sr.total_pnl:,.0f}, Saved=${sr.total_pnl_saved:,.0f}")

    # Build results dict
    results_dict = {
        "phase": "7_mft_sensitivity_analysis",
        "timestamp": time.ctime(),
        "method": "latin_hypercube_sampling",
        "n_samples": n_samples,
        "parameter_bounds": bounds,
        "parameters_swept": list(bounds.keys()),
        "samples": [asdict(r) for r in results],
    }

    # Add sensitivity summary
    results_dict["summary"] = _compute_sensitivity_summary(results)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSensitivity results saved to {output_path}")

    # Generate plots
    _generate_sensitivity_plots(results)

    return results


def _compute_sensitivity_summary(results):
    """Compute PRCC-like sensitivity indices for each parameter."""
    df = pd.DataFrame([asdict(r) for r in results])
    params = ["t1", "t2", "fp_cost_mult"]
    targets = ["total_pnl", "total_pnl_saved", "precision", "recall"]

    summary = {}
    for target in targets:
        target_corrs = {}
        for param in params:
            corr = df[param].corr(df[target])
            target_corrs[param] = round(corr, 4)
        summary[target] = target_corrs

    # Topline stats
    best_sample = max(results, key=lambda r: r.total_pnl_saved)
    worst_sample = min(results, key=lambda r: r.total_pnl_saved)

    summary["best_case"] = asdict(best_sample)
    summary["worst_case"] = asdict(worst_sample)
    summary["mean_pnl_saved"] = round(float(df["total_pnl_saved"].mean()), 2)
    summary["std_pnl_saved"] = round(float(df["total_pnl_saved"].std()), 2)

    return summary


def _generate_sensitivity_plots(results):
    """Generate tornado charts and heatmaps for sensitivity analysis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Sensitivity] Matplotlib not available, skipping plots.")
        return

    df = pd.DataFrame([asdict(r) for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Tornado chart: parameter correlation with P&L saved
    ax1 = axes[0, 0]
    params = ["t1", "t2", "fp_cost_mult"]
    labels = ["T₁ Latency", "T₂ Human Delay", "FP Cost Mult"]
    target = "total_pnl_saved"
    corrs = [df[p].corr(df[target]) for p in params]
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in corrs]

    bars = ax1.barh(labels, corrs, color=colors, alpha=0.8, edgecolor="white")
    ax1.axvline(x=0, color="black", linewidth=1)
    for bar, c in zip(bars, corrs):
        ax1.text(bar.get_width() + (0.01 if c >= 0 else -0.08),
                 bar.get_y() + bar.get_height() / 2,
                 f"{c:.3f}", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Correlation with P&L Saved")
    ax1.set_title("Parameter Importance (PRCC-like)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    # 2. Scatter: T1 vs P&L Saved
    ax2 = axes[0, 1]
    ax2.scatter(df["t1"], df["total_pnl_saved"] / 1000, alpha=0.5, s=20, c="steelblue")
    z = np.polyfit(df["t1"], df["total_pnl_saved"], 1)
    p = np.poly1d(z)
    t1_sorted = np.sort(df["t1"])
    ax2.plot(t1_sorted, p(t1_sorted) / 1000, "r--", linewidth=2, label=f"Trend (slope={z[0]:.1f})")
    ax2.set_xlabel("T₁ Intervention Latency (s)")
    ax2.set_ylabel("P&L Saved ($K)")
    ax2.set_title("T₁ vs P&L Saved", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: T2 vs P&L Saved
    ax3 = axes[1, 0]
    ax3.scatter(df["t2"], df["total_pnl_saved"] / 1000, alpha=0.5, s=20, c="steelblue")
    z = np.polyfit(df["t2"], df["total_pnl_saved"], 1)
    p = np.poly1d(z)
    t2_sorted = np.sort(df["t2"])
    ax3.plot(t2_sorted, p(t2_sorted) / 1000, "r--", linewidth=2, label=f"Trend (slope={z[0]:.1f})")
    ax3.set_xlabel("T₂ Human Verification Delay (s)")
    ax3.set_ylabel("P&L Saved ($K)")
    ax3.set_title("T₂ vs P&L Saved", fontsize=13, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap-style: 2D bin of (T1, T2) → P&L Saved
    ax4 = axes[1, 1]
    t1_bins = np.linspace(df["t1"].min(), df["t1"].max(), 8)
    t2_bins = np.linspace(df["t2"].min(), df["t2"].max(), 8)
    heatmap_data, _, _ = np.histogram2d(
        df["t1"], df["t2"], bins=[t1_bins, t2_bins],
        weights=df["total_pnl_saved"] / 1000,
    )
    counts, _, _ = np.histogram2d(
        df["t1"], df["t2"], bins=[t1_bins, t2_bins],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        heatmap_mean = np.divide(heatmap_data, counts, where=counts > 0)
        heatmap_mean[counts == 0] = np.nan

    im = ax4.imshow(heatmap_mean.T, origin="lower", aspect="auto",
                    cmap="RdYlGn", interpolation="nearest")
    ax4.set_xticks(range(len(t1_bins) - 1))
    ax4.set_yticks(range(len(t2_bins) - 1))
    ax4.set_xticklabels([f"{t1_bins[i]:.0f}-{t1_bins[i+1]:.0f}s" for i in range(len(t1_bins) - 1)],
                        rotation=45, fontsize=8)
    ax4.set_yticklabels([f"{t2_bins[i]:.0f}-{t2_bins[i+1]:.0f}s" for i in range(len(t2_bins) - 1)],
                        fontsize=8)
    ax4.set_xlabel("T₁ (s)")
    ax4.set_ylabel("T₂ (s)")
    ax4.set_title("Mean P&L Saved ($K) by (T₁, T₂)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    plt.tight_layout()
    plot_path = "./plots/mft_sensitivity_heatmaps.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Sensitivity] Sensitivity plots saved to {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MFT Sensitivity Analysis")
    parser.add_argument("--samples", type=int, default=200, help="LHS sample count")
    parser.add_argument("--results-path", default=None,
                        help="Path to cached pipeline results JSON")
    parser.add_argument("--output", default="./output/mft_sensitivity_results.json")
    args = parser.parse_args()

    run_lhs_sensitivity(
        n_samples=args.samples,
        pipeline_results_path=args.results_path,
        output_path=args.output,
    )
