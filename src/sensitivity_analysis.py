"""Phase 5: Market impact and sensitivity analysis.

Latin Hypercube Sampling over 5-dim parameter space:
1. Drop duration (500-10000ms)
2. Trough price ($130-$180)
3. Recovery duration (1000-20000ms)
4. Position size (100-100000 shares, log scale)
5. Confidence threshold (0.1-0.95)

Two market regimes: normal, stress.
Summary metric: Sharpe ratio.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


def latin_hypercube_sampling(n_samples, param_bounds, log_scale_dims=None, seed=42):
    """Generate LHS samples scaled to parameter bounds.

    Args:
        n_samples: Number of points to sample
        param_bounds: dict of {name: (lo, hi)}
        log_scale_dims: set of parameter names to sample in log space
        seed: Random seed

    Returns:
        list of dicts [{param_name: value, ...}, ...]
    """
    if log_scale_dims is None:
        log_scale_dims = set()

    rng = np.random.default_rng(seed)
    names = list(param_bounds.keys())
    n_dims = len(names)

    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        # Stratify [0, 1] into n intervals, pick one point per interval
        perm = rng.permutation(n_samples)
        samples[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples
        lo, hi = param_bounds[names[d]]
        if names[d] in log_scale_dims:
            samples[:, d] = np.exp(np.log(lo) + samples[:, d] * (np.log(hi) - np.log(lo)))
        else:
            samples[:, d] = lo + samples[:, d] * (hi - lo)

    return [{names[d]: samples[i, d] for d in range(n_dims)} for i in range(n_samples)]


def flash_crash_price(t_ms, base_price=190.0, trough_price=150.0,
                      drop_duration_ms=3000, recovery_duration_ms=7000):
    """Price at t_ms during a configurable flash crash.

    Drop phase: [0, drop_duration_ms] — linear from base_price → trough_price
    Recovery: [drop_duration_ms, drop_duration_ms + recovery_duration_ms]
    After recovery: base_price

    With defaults, matches PnLCalculator.intervention_price() exactly.
    """
    if t_ms is None or t_ms < 0:
        return base_price

    drop_magnitude = base_price - trough_price

    if t_ms <= drop_duration_ms:
        return base_price - (t_ms / drop_duration_ms) * drop_magnitude
    elif t_ms <= drop_duration_ms + recovery_duration_ms:
        recovery_t = t_ms - drop_duration_ms
        return trough_price + (recovery_t / recovery_duration_ms) * drop_magnitude
    return base_price


class SensitivityAnalyzer:
    """Sweep flash crash parameters over LHS samples, compute P&L + Sharpe.

    Takes pre-collected per-sample pipeline results (verdict, confidence,
    intervention_time_ms, actual labels) and recomputes net P&L under
    different crash parameters and position sizes — without re-running
    the LLM pipeline.
    """

    PARAM_BOUNDS = {
        "drop_duration_ms": (500, 10000),
        "trough_price": (130, 180),
        "recovery_duration_ms": (1000, 20000),
        "position_size": (100, 100000),
        "confidence_threshold": (0.1, 0.95),
    }
    LOG_SCALE_DIMS = {"position_size"}

    NORMAL_REGIME = {
        "drop_duration_ms": 3000,
        "trough_price": 150,
        "recovery_duration_ms": 7000,
        "position_size": 1000,
        "confidence_threshold": 0.5,
    }
    STRESS_REGIME = {
        "drop_duration_ms": 1000,
        "trough_price": 130,
        "recovery_duration_ms": 15000,
        "position_size": 10000,
        "confidence_threshold": 0.7,
    }

    def __init__(self, base_results, base_price=190.0, fp_cost_factor=0.5):
        """Args:
            base_results: list of dicts with keys:
                actual, verdict, confidence, intervention_time_ms
            base_price: starting price before crash
            fp_cost_factor: P(price_moves_against) for FP opportunity cost
        """
        self.results = pd.DataFrame(base_results)
        self.base_price = base_price
        self.fp_cost_factor = fp_cost_factor

        required = {"actual", "verdict", "confidence", "intervention_time_ms"}
        missing = required - set(self.results.columns)
        if missing:
            raise ValueError(f"base_results missing columns: {missing}")

    def compute_pnl(self, params):
        """Per-sample net P&L for given parameter set.

        Returns numpy array (one P&L value per sample):
          TP (label=1, intervened) → +savings from intervening before trough
          FP (label=0, intervened) → -opportunity cost of wrong intervention
          FN (label=1, not intervened) → -full loss from riding crash
          TN (label=0, not intervened) → 0
        """
        dd = params["drop_duration_ms"]
        tp = params["trough_price"]
        rd = params["recovery_duration_ms"]
        ps = params["position_size"]
        ct = params["confidence_threshold"]

        pnl = np.zeros(len(self.results))

        for i, (_, row) in enumerate(self.results.iterrows()):
            actual = int(row["actual"])
            verdict = int(row["verdict"])
            confidence = float(row["confidence"])
            inter_time = row["intervention_time_ms"]

            intervened = (verdict == 1 and confidence >= ct)

            if actual == 1 and intervened:
                # TP: correct intervention before trough
                p_intervention = flash_crash_price(
                    inter_time, self.base_price, tp, dd, rd)
                pnl[i] = ps * max(0.0, p_intervention - tp)

            elif actual == 0 and intervened:
                # FP: unnecessarily reversed a correct trade
                p_intervention = flash_crash_price(
                    inter_time, self.base_price, tp, dd, rd)
                pnl[i] = -ps * (self.base_price - p_intervention) * self.fp_cost_factor

            elif actual == 1 and not intervened:
                # FN: missed detection, rode full crash
                pnl[i] = -ps * (self.base_price - tp)

            # TN: correct non-intervention, zero P&L

        return pnl

    def compute_sharpe(self, pnl_values):
        """Annualized Sharpe ratio from per-sample P&L stream."""
        if len(pnl_values) < 2:
            return 0.0
        std = float(np.std(pnl_values))
        if std == 0:
            return 0.0
        return float(np.mean(pnl_values) / std * np.sqrt(252))

    def evaluate_params(self, params):
        """Aggregate metrics for a single parameter set."""
        pnl = self.compute_pnl(params)
        ct = params["confidence_threshold"]

        n_tp = int(((self.results["actual"] == 1) &
                    (self.results["verdict"] == 1) &
                    (self.results["confidence"] >= ct)).sum())
        n_fp = int(((self.results["actual"] == 0) &
                    (self.results["verdict"] == 1) &
                    (self.results["confidence"] >= ct)).sum())
        n_actual_fake = int((self.results["actual"] == 1).sum())
        n_fn = n_actual_fake - n_tp

        return {
            "total_pnl": round(float(pnl.sum()), 2),
            "sharpe_ratio": round(self.compute_sharpe(pnl), 4),
            "max_drawdown": round(float(np.min(pnl)), 2),
            "mean_pnl": round(float(np.mean(pnl)), 2),
            "std_pnl": round(float(np.std(pnl)), 2),
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_fn": n_fn,
            "n_interventions": n_tp + n_fp,
            **params,
        }

    def evaluate_regime(self, regime_params):
        """Evaluate a market regime, return (metrics_dict, pnl_array)."""
        pnl = self.compute_pnl(regime_params)
        metrics = self.evaluate_params(regime_params)
        return metrics, pnl

    def run_sweep(self, n_samples=30, seed=42):
        """Generate LHS samples and evaluate each. Returns list of metrics dicts."""
        samples = latin_hypercube_sampling(
            n_samples, self.PARAM_BOUNDS,
            log_scale_dims=self.LOG_SCALE_DIMS, seed=seed)

        results = []
        for i, params in enumerate(samples):
            params["position_size"] = int(round(params["position_size"]))
            metrics = self.evaluate_params(params)
            results.append(metrics)

            if (i + 1) % 5 == 0:
                print(f"  LHS [{i+1}/{n_samples}] "
                      f"P&L=${metrics['total_pnl']:>10,.0f} "
                      f"Sharpe={metrics['sharpe_ratio']:.3f}")

        return results

    def sensitivity_ranking(self, sweep_results):
        """Rank parameters by impact on total P&L via linear regression."""
        df = pd.DataFrame(sweep_results)
        param_cols = list(self.PARAM_BOUNDS.keys())

        X = df[param_cols].copy()
        for col in param_cols:
            if col in self.LOG_SCALE_DIMS:
                X[col] = np.log(X[col])
        # Z-score normalize
        X = (X - X.mean()) / X.std()

        y = df["total_pnl"].values
        coeffs, _residuals, _rank, _s = np.linalg.lstsq(X.values, y, rcond=None)

        ranking = sorted(
            [(param_cols[i], abs(coeffs[i]), coeffs[i])
             for i in range(len(param_cols))],
            key=lambda x: x[1], reverse=True)

        return [{"param": r[0],
                 "importance": round(float(r[1]), 2),
                 "direction": "positive" if r[2] > 0 else "negative"}
                for r in ranking]

    def plot_heatmaps(self, sweep_results, save_path="./plots/sensitivity_heatmaps.png"):
        """2D sensitivity heatmaps: 2x3 grid of P&L across parameter pairs."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df = pd.DataFrame(sweep_results)
        param_cols = list(self.PARAM_BOUNDS.keys())
        friendly = {
            "drop_duration_ms": "Drop Duration (ms)",
            "trough_price": "Trough Price ($)",
            "recovery_duration_ms": "Recovery Duration (ms)",
            "position_size": "Position Size (shares)",
            "confidence_threshold": "Conf. Threshold",
        }

        pairs = [
            ("drop_duration_ms", "trough_price"),
            ("drop_duration_ms", "position_size"),
            ("trough_price", "confidence_threshold"),
            ("position_size", "confidence_threshold"),
            ("drop_duration_ms", "recovery_duration_ms"),
            ("trough_price", "recovery_duration_ms"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        axes = axes.flatten()

        for idx, (px, py) in enumerate(pairs):
            ax = axes[idx]
            scatter = ax.scatter(
                df[px], df[py], c=df["total_pnl"], cmap="RdYlGn",
                s=60, alpha=0.7, edgecolors="k", linewidth=0.3)
            ax.set_xlabel(friendly[px], fontsize=9)
            ax.set_ylabel(friendly[py], fontsize=9)
            plt.colorbar(scatter, ax=ax, label="Total P&L ($)")

        fig.suptitle("Sensitivity Analysis: P&L Across Parameter Space",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Sensitivity heatmaps saved to {save_path}")

    def plot_pnl_distribution(self, normal_pnl, stress_pnl,
                              save_path="./plots/pnl_distribution.png"):
        """Per-sample P&L histogram comparing normal vs stress regime."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.figure(figsize=(12, 6))
        bins = 30
        plt.hist(normal_pnl, bins=bins, alpha=0.55, label="Normal Regime",
                 color="steelblue", edgecolor="white")
        plt.hist(stress_pnl, bins=bins, alpha=0.55, label="Stress Regime",
                 color="crimson", edgecolor="white")

        mean_n = np.mean(normal_pnl)
        mean_s = np.mean(stress_pnl)
        plt.axvline(mean_n, color="steelblue", linestyle="--", linewidth=2,
                    label=f"Normal mean: ${mean_n:,.0f}")
        plt.axvline(mean_s, color="crimson", linestyle="--", linewidth=2,
                    label=f"Stress mean: ${mean_s:,.0f}")

        plt.xlabel("Per-Sample P&L ($)")
        plt.ylabel("Frequency")
        plt.title("P&L Distribution: Normal vs Stress Regime", fontsize=13,
                  fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"P&L distribution saved to {save_path}")
