"""Phase 8: Liquidity sensitivity sweep.

Latin Hypercube Sampling over 4-dim liquidity parameter space:
1. depth_decay_ms (200-5000ms): Speed of bid depth evaporation.
2. normal_bid_depth (100-10000 shares, log scale): Baseline order book depth.
3. min_fill_prob (0.01-0.5): Minimum fill probability during panic.
4. position_size (100-10000 shares, log scale): Order size.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.sensitivity_analysis import latin_hypercube_sampling
from src.l2_data_generator import FlashCrashL2Config
from src.hft_backtest import MicrostructureSimulator


class LiquiditySensitivityAnalyzer:
    """Sweep liquidity and trade size parameters, evaluate P&L, Sharpe, and Capacity limits."""

    PARAM_BOUNDS = {
        "depth_decay_ms": (200.0, 5000.0),
        "normal_bid_depth": (100.0, 10000.0),
        "min_fill_prob": (0.01, 0.5),
        "position_size": (100.0, 10000.0),
    }
    LOG_SCALE_DIMS = {"normal_bid_depth", "position_size"}

    def __init__(self, base_results, base_price=190.0):
        """
        Args:
            base_results: list of dicts with keys:
                actual, verdict, confidence, intervention_time_ms
            base_price: reference stock price before crash
        """
        self.results = base_results
        self.base_price = base_price

    def evaluate_params(self, params, seed=42):
        """Evaluate a single liquidity parameter set on Phase 7 detection results."""
        config = FlashCrashL2Config(
            base_price=self.base_price,
            depth_decay_ms=params["depth_decay_ms"],
            normal_bid_depth=int(params["normal_bid_depth"]),
            min_fill_prob=params["min_fill_prob"],
        )

        simulator = MicrostructureSimulator(
            results=self.results,
            base_price=self.base_price,
            position_size=int(params["position_size"]),
            config=config,
            seed=seed,
        )

        # Run under standard normal regime conditions
        metrics = simulator.run(confidence_threshold=0.5, regime="normal")

        # Annualized Sharpe ratio on realized P&L stream
        realized_pnls = [d["realized_pnl"] for d in metrics["details"]]
        std = np.std(realized_pnls)
        mean = np.mean(realized_pnls)
        sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0.0

        # Capacity Ratio: position size relative to normal bid depth
        capacity_ratio = params["position_size"] / params["normal_bid_depth"]

        return {
            "total_ideal_pnl": metrics["total_ideal_pnl"],
            "total_realized_pnl": metrics["total_realized_pnl"],
            "pnl_gap": metrics["pnl_gap"],
            "pnl_gap_pct": metrics["pnl_gap_pct"],
            "mean_fill_rate": metrics["mean_fill_rate"],
            "total_spread_cost": metrics["total_spread_cost"],
            "total_adverse_selection": metrics["total_adverse_selection"],
            "mean_slippage_pct": metrics["mean_slippage_pct"],
            "sharpe_ratio": round(sharpe, 4),
            "capacity_ratio": round(capacity_ratio, 4),
            **params,
        }

    def run_sweep(self, n_samples=30, seed=42):
        """Generate LHS samples and run sweep. Returns list of result dicts."""
        samples = latin_hypercube_sampling(
            n_samples, self.PARAM_BOUNDS,
            log_scale_dims=self.LOG_SCALE_DIMS, seed=seed
        )

        results = []
        print(f"\n[Phase 8] Running {n_samples} LHS liquidity sensitivity samples...")
        for i, params in enumerate(samples):
            # Enforce integers for counts
            params["normal_bid_depth"] = int(round(params["normal_bid_depth"]))
            params["position_size"] = int(round(params["position_size"]))

            metrics = self.evaluate_params(params, seed=seed + i)
            results.append(metrics)

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  LHS [{i+1}/{n_samples}] "
                      f"Pos={metrics['position_size']:>5d} | "
                      f"Depth={metrics['normal_bid_depth']:>5d} | "
                      f"Realized P&L=${metrics['total_realized_pnl']:>10,.0f} | "
                      f"Sharpe={metrics['sharpe_ratio']:.3f}")

        return results

    def analyze_thresholds(self, sweep_results):
        """Identify critical liquidity thresholds: Sign Flip and Capacity Limit."""
        df = pd.DataFrame(sweep_results)
        
        # 1. Sign Flip Point: where realized P&L crosses below 0
        negatives = df[df["total_realized_pnl"] < 0]
        positives = df[df["total_realized_pnl"] >= 0]
        
        crossover_capacity_ratio = "N/A"
        max_scalable_position = "N/A"
        
        if not negatives.empty and not positives.empty:
            # Capacity Ratio crossover point (average capacity ratio for negative runs vs positive runs)
            crossover_capacity_ratio = round(float(negatives["capacity_ratio"].min()), 4)
            # Maximum position size that still allowed positive realized P&L
            max_scalable_position = int(positives["position_size"].max())
        elif negatives.empty:
            crossover_capacity_ratio = "No Sign Flip (Always Positive)"
            max_scalable_position = int(df["position_size"].max())
        else:
            crossover_capacity_ratio = "Always Negative (Liquidity Insufficient)"
            max_scalable_position = 0

        # 2. Rank parameters by impact on Realized P&L
        param_cols = list(self.PARAM_BOUNDS.keys())
        X = df[param_cols].copy()
        for col in param_cols:
            if col in self.LOG_SCALE_DIMS:
                X[col] = np.log(X[col])
        X = (X - X.mean()) / X.std()
        y = df["total_realized_pnl"].values
        
        coeffs, _, _, _ = np.linalg.lstsq(X.values, y, rcond=None)
        ranking = sorted(
            [(param_cols[i], coeffs[i]) for i in range(len(param_cols))],
            key=lambda x: abs(x[1]), reverse=True
        )

        importance_ranking = [
            {
                "param": r[0],
                "coefficient": round(float(r[1]), 2),
                "direction": "positive" if r[1] > 0 else "negative"
            }
            for r in ranking
        ]

        return {
            "crossover_capacity_ratio": crossover_capacity_ratio,
            "max_scalable_position_shares": max_scalable_position,
            "importance_ranking": importance_ranking,
        }

    def plot_heatmaps(self, sweep_results, save_path="./plots/phase8_liquidity_heatmaps.png"):
        """Plot a 2x3 grid of P&L sensitivity across parameter pairs."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(sweep_results)

        friendly = {
            "depth_decay_ms": "Depth Decay (ms)",
            "normal_bid_depth": "Normal Bid Depth (shares)",
            "min_fill_prob": "Min Fill Prob",
            "position_size": "Position Size (shares)",
        }

        pairs = [
            ("depth_decay_ms", "normal_bid_depth"),
            ("position_size", "normal_bid_depth"),
            ("min_fill_prob", "depth_decay_ms"),
            ("position_size", "depth_decay_ms"),
            ("position_size", "min_fill_prob"),
            ("normal_bid_depth", "min_fill_prob"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        axes = axes.flatten()

        for idx, (px, py) in enumerate(pairs):
            ax = axes[idx]
            
            # Use log scale on axes where appropriate for visual clarity
            if px in self.LOG_SCALE_DIMS:
                ax.set_xscale("log")
            if py in self.LOG_SCALE_DIMS:
                ax.set_yscale("log")
                
            scatter = ax.scatter(
                df[px], df[py], c=df["total_realized_pnl"], cmap="RdYlGn",
                s=70, alpha=0.8, edgecolors="k", linewidth=0.3
            )
            ax.set_xlabel(friendly[px], fontsize=9)
            ax.set_ylabel(friendly[py], fontsize=9)
            
            # For the capacity plot, show the 1:1 capacity boundary line
            if px == "position_size" and py == "normal_bid_depth":
                xlims = ax.get_xlim()
                ax.plot(xlims, xlims, color="blue", linestyle="--", alpha=0.5, label="Capacity Limit (1:1)")
                ax.legend(fontsize=8)
                
            plt.colorbar(scatter, ax=ax, label="Realized P&L ($)")

        fig.suptitle("Phase 8: Liquidity & Capacity Sensitivity Analysis (Realized P&L)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Liquidity sensitivity heatmaps saved to {save_path}")
