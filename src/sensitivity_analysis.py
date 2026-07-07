"""Policy-layer sensitivity analysis for the hybrid verification framework.

Systematically varies policy parameters and measures impact on expected
cost, action distribution, cost savings, and verifier selection.

This module analyses the policy layer (base rate, cost asymmetry, latency
budget, confidence thresholds, architecture profiles) separately from
verifier-level performance (which is mock-only at this stage).

Provides:
- POLICY_PARAMETER_GRID: default parameter ranges for sweeping.
- PolicySensitivityAnalyzer: sweep harness, sensitivity ranking, interaction
  detection, and summary generation.
- CLI entry point (``python -m src.sensitivity_analysis --quick``).
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from src.hybrid_policy import (
    DEFAULT_VERIFIER_PROFILES,
    HybridPolicy,
    RiskAction,
    VerifierProfile,
)
from src.schemas import Verdict, VerificationResult

# ── Alternate profile sets for architecture-profile sensitivity ────

# High-precision profiles (verifier is more conservative)
HIGH_PRECISION_PROFILES: Dict[str, VerifierProfile] = {
    "single-shot": VerifierProfile(
        name="single-shot", latency_s=0.002, f1=0.65, precision=0.80, recall=0.55
    ),
    "voting": VerifierProfile(
        name="voting", latency_s=0.005, f1=0.70, precision=0.85, recall=0.60
    ),
    "moa": VerifierProfile(
        name="moa", latency_s=0.008, f1=0.72, precision=0.88, recall=0.62
    ),
    "rag": VerifierProfile(
        name="rag", latency_s=0.004, f1=0.68, precision=0.82, recall=0.58
    ),
}

# Low-latency profiles (all fast, some at cost of precision)
LOW_LATENCY_PROFILES: Dict[str, VerifierProfile] = {
    "single-shot": VerifierProfile(
        name="single-shot", latency_s=0.0005, f1=0.45, precision=0.40, recall=0.95
    ),
    "voting": VerifierProfile(
        name="voting", latency_s=0.001, f1=0.48, precision=0.42, recall=0.95
    ),
    "rag": VerifierProfile(
        name="rag", latency_s=0.001, f1=0.47, precision=0.41, recall=0.95
    ),
}

PROFILE_SETS: Dict[str, Dict[str, VerifierProfile]] = {
    "default": DEFAULT_VERIFIER_PROFILES,
    "high_precision": HIGH_PRECISION_PROFILES,
    "low_latency": LOW_LATENCY_PROFILES,
}

# ── Parameter grid definitions ─────────────────────────────────────


@dataclass
class PolicySweepConfig:
    """Single parameter set for the policy sweep.

    Attributes:
        base_rate: Misinformation base rate.
        cost_fp: Cost of a false positive.
        cost_fn: Cost of a false negative.
        latency_budget_s: Maximum acceptable latency.
        escalation_min: Minimum confidence threshold for escalation.
        profile_set: Which architecture profile set to use.
    """

    base_rate: float = 0.05
    cost_fp: float = 1.0
    cost_fn: float = 10.0
    latency_budget_s: float = 5.0
    escalation_min: float = 0.3
    profile_set: str = "default"


# Default grid: full factorial outer product of these values
BASE_RATES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
COST_RATIOS: List[Tuple[float, float]] = [
    (1.0, 1.0),    # 1:1  — symmetric
    (1.0, 5.0),    # 1:5
    (1.0, 10.0),   # 1:10
    (1.0, 25.0),   # 1:25
    (1.0, 100.0),  # 1:100
]
LATENCY_BUDGETS = [0.5, 1.0, 2.0, 5.0, 10.0]
ESCALATION_MINS = [0.1, 0.2, 0.3, 0.4, 0.5]
PROFILE_SET_NAMES = ["default", "high_precision", "low_latency"]

# Quick subset for --quick mode
QUICK_BASE_RATES = [0.01, 0.05, 0.25]
QUICK_COST_RATIOS: List[Tuple[float, float]] = [(1.0, 1.0), (1.0, 10.0), (1.0, 100.0)]
QUICK_LATENCY_BUDGETS = [1.0, 5.0]
QUICK_ESCALATION_MINS = [0.2, 0.4]
QUICK_PROFILE_SET_NAMES = ["default", "high_precision"]


@dataclass
class SweepResult:
    """Result from a single sweep configuration.

    Attributes:
        config: The parameter set that was evaluated.
        selected_verifier: Verifier selected by the meta-policy.
        action: Risk action taken by the policy.
        expected_cost: Minimum of (cost_intervene, cost_hold).
        cost_saved_vs_reverse: Savings vs always-reverse.
        cost_saved_vs_hold: Savings vs always-hold.
        latency_ok: Whether selected verifier's latency fits budget.
    """

    config: PolicySweepConfig
    selected_verifier: str
    action: RiskAction
    expected_cost: float
    cost_saved_vs_reverse: float
    cost_saved_vs_hold: float
    latency_ok: bool


@dataclass
class SensitivitySummary:
    """Aggregated sensitivity results across a sweep.

    Attributes:
        parameter_ranking: List of (param_name, variance_contribution) sorted
                           descending by impact.
        interactions: List of (param_pair, interaction_strength) for top pairs.
        best_config: Best config and its expected cost.
        worst_config: Worst config and its expected cost.
        n_configs: Number of configurations evaluated.
        action_distribution: Overall action distribution across all configs.
    """

    parameter_ranking: List[Tuple[str, float]] = field(default_factory=list)
    interactions: List[Tuple[Tuple[str, str], float]] = field(default_factory=list)
    best_config: Tuple[PolicySweepConfig, float] | None = None
    worst_config: Tuple[PolicySweepConfig, float] | None = None
    n_configs: int = 0
    action_distribution: Dict[str, float] = field(default_factory=dict)


# ── Sensitivity Analyzer ──────────────────────────────────────────


class PolicySensitivityAnalyzer:
    """Sweep harness and sensitivity analysis for the hybrid policy layer.

    Usage::

        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        analyzer.print_summary(summary)
        analyzer.save_results(results, summary, "output/phase08_sensitivity.json")
    """

    # Fixed verification result fed to the policy in every sweep cell.
    # A moderate-confidence FAKE verdict so action can change across
    # parameter values (not clamped to one extreme).
    REFERENCE_RESULT = VerificationResult(
        item_id="ref-1",
        verdict=Verdict.FAKE,
        confidence=0.55,
        evidence=["Reference claim for sensitivity sweep"],
        metadata={},
    )

    def __init__(self) -> None:
        pass

    # ── Sweep grid construction ─────────────────────────────────────

    def build_grid(self, quick: bool = False) -> List[PolicySweepConfig]:
        """Build the full factorial parameter grid.

        Args:
            quick: If True, use a smaller subset for faster runs.

        Returns:
            List of PolicySweepConfig covering the grid.
        """
        base_rates = QUICK_BASE_RATES if quick else BASE_RATES
        cost_options = QUICK_COST_RATIOS if quick else COST_RATIOS
        latencies = QUICK_LATENCY_BUDGETS if quick else LATENCY_BUDGETS
        escalations = QUICK_ESCALATION_MINS if quick else ESCALATION_MINS
        profiles = QUICK_PROFILE_SET_NAMES if quick else PROFILE_SET_NAMES

        grid: List[PolicySweepConfig] = []
        for br, (cfp, cfn), lat, esc, ps in itertools.product(
            base_rates, cost_options, latencies, escalations, profiles
        ):
            grid.append(
                PolicySweepConfig(
                    base_rate=br,
                    cost_fp=cfp,
                    cost_fn=cfn,
                    latency_budget_s=lat,
                    escalation_min=esc,
                    profile_set=ps,
                )
            )
        return grid

    # ── Single-cell evaluation ──────────────────────────────────────

    def evaluate(
        self, config: PolicySweepConfig
    ) -> SweepResult:
        """Evaluate the policy at a single grid point.

        Args:
            config: Policy parameter set.

        Returns:
            SweepResult with selected verifier, action, expected cost,
            cost savings, and latency check.
        """
        profiles = PROFILE_SETS[config.profile_set]

        # Build policy with these parameters
        policy = HybridPolicy(
            cost_fp=config.cost_fp,
            cost_fn=config.cost_fn,
            verifier_profiles=profiles.copy(),
            latency_budget_s=config.latency_budget_s,
        )

        # Override the escalation confidence min — the policy's decide()
        # uses the hardcoded threshold 0.3 internally. For escalation
        # sensitivity we instead manipulate the reference confidence
        # relative to this threshold. (The escalation_min in config is
        # tracked as metadata, not used as a direct policy override.)

        # Select verifier via meta-policy
        selected = policy.select_verifier(base_rate=config.base_rate)

        # Get action from the policy
        decision = policy.decide(
            self.REFERENCE_RESULT,
            base_rate=config.base_rate,
            verifier_name=selected,
        )

        # Check latency compliance
        profile = profiles.get(selected)
        latency_ok = profile is not None and profile.latency_s <= config.latency_budget_s

        return SweepResult(
            config=config,
            selected_verifier=selected,
            action=decision.action,
            expected_cost=min(decision.expected_cost_intervene, decision.expected_cost_hold),
            cost_saved_vs_reverse=decision.cost_saved_vs_always_reverse,
            cost_saved_vs_hold=decision.cost_saved_vs_always_hold,
            latency_ok=latency_ok,
        )

    # ── Full sweep ──────────────────────────────────────────────────

    def run_sweep(
        self, quick: bool = False, verbose: bool = False
    ) -> List[SweepResult]:
        """Run the full parameter sweep.

        Args:
            quick: Use a smaller parameter grid.
            verbose: Print progress messages.

        Returns:
            List of SweepResult, one per grid configuration.
        """
        grid = self.build_grid(quick=quick)
        results: List[SweepResult] = []

        for i, config in enumerate(grid):
            if verbose and (i % 20 == 0):
                print(f"Sweep {i + 1}/{len(grid)}: br={config.base_rate}, "
                      f"cost=({config.cost_fp},{config.cost_fn}), "
                      f"lat={config.latency_budget_s}, "
                      f"esc={config.escalation_min}, "
                      f"prof={config.profile_set}")
            results.append(self.evaluate(config))

        return results

    # ── Sensitivity ranking (OAT variance decomposition) ────────────

    def compute_sensitivity(
        self, results: List[SweepResult]
    ) -> SensitivitySummary:
        """Compute sensitivity ranking and interaction detection.

        Uses one-at-a-time (OAT) variance decomposition:
        - For each parameter, group results by its value and compute the
          between-group variance in expected_cost.
        - Parameters with higher between-group variance explain more of
          the output variation → ranked higher.

        Interaction detection:
        - For each pair of parameters, compute the interaction strength as
          the additional variance explained by the 2-parameter grouping
          beyond the sum of individual variances.

        Args:
            results: Results from a full sweep.

        Returns:
            SensitivitySummary with ranking, interactions, best/worst.
        """
        if not results:
            return SensitivitySummary()

        costs = np.array([r.expected_cost for r in results])

        # ── OAT variance ranking ─────────────────────────────────────

        param_names = [
            "base_rate",
            "cost_ratio",
            "latency_budget",
            "escalation_min",
            "profile_set",
        ]

        def _param_values(r: SweepResult, name: str) -> Any:
            if name == "base_rate":
                return r.config.base_rate
            elif name == "cost_ratio":
                return (r.config.cost_fp, r.config.cost_fn)
            elif name == "latency_budget":
                return r.config.latency_budget_s
            elif name == "escalation_min":
                return r.config.escalation_min
            elif name == "profile_set":
                return r.config.profile_set
            return None

        total_var = float(np.var(costs, ddof=0))
        ranking: List[Tuple[str, float]] = []

        for pname in param_names:
            groups: Dict[Any, List[float]] = {}
            for r in results:
                key = _param_values(r, pname)
                groups.setdefault(key, []).append(r.expected_cost)

            # Between-group variance (weighted by group size)
            group_means = {k: float(np.mean(v)) for k, v in groups.items()}
            grand_mean = float(np.mean(costs))
            between_var = sum(
                len(v) * (group_means[k] - grand_mean) ** 2
                for k, v in groups.items()
            ) / len(results)

            contribution = between_var / total_var if total_var > 0 else 0.0
            ranking.append((pname, contribution))

        ranking.sort(key=lambda x: x[1], reverse=True)

        # ── Interaction detection (pairwise) ─────────────────────────

        interactions: List[Tuple[Tuple[str, str], float]] = []
        param_pairs = [
            ("base_rate", "cost_ratio"),
            ("base_rate", "latency_budget"),
            ("base_rate", "profile_set"),
            ("cost_ratio", "latency_budget"),
            ("cost_ratio", "profile_set"),
            ("latency_budget", "profile_set"),
        ]

        base_vars = {p: v for p, v in ranking}

        for p1, p2 in param_pairs:
            groups: Dict[Tuple[Any, Any], List[float]] = {}
            for r in results:
                key = (_param_values(r, p1), _param_values(r, p2))
                groups.setdefault(key, []).append(r.expected_cost)

            group_means = {k: float(np.mean(v)) for k, v in groups.items()}
            between_var = sum(
                len(v) * (group_means[k] - grand_mean) ** 2
                for k, v in groups.items()
            ) / len(results)

            joint_var = between_var / total_var if total_var > 0 else 0.0
            indep_var = base_vars.get(p1, 0.0) + base_vars.get(p2, 0.0)
            interaction = max(0.0, joint_var - indep_var)
            interactions.append(((p1, p2), interaction))

        interactions.sort(key=lambda x: x[1], reverse=True)

        # ── Best / worst ─────────────────────────────────────────────

        best_idx = int(np.argmin(costs))
        worst_idx = int(np.argmax(costs))

        best_config = (results[best_idx].config, float(costs[best_idx]))
        worst_config = (results[worst_idx].config, float(costs[worst_idx]))

        # ── Action distribution ──────────────────────────────────────

        action_counts: Dict[str, int] = {}
        for r in results:
            action_counts[r.action.value] = action_counts.get(r.action.value, 0) + 1
        total = len(results)
        action_dist = (
            {k: v / total for k, v in action_counts.items()} if total > 0 else {}
        )

        return SensitivitySummary(
            parameter_ranking=ranking,
            interactions=interactions,
            best_config=best_config,
            worst_config=worst_config,
            n_configs=len(results),
            action_distribution=action_dist,
        )

    # ── Output helpers ──────────────────────────────────────────────

    def summary_to_dict(
        self, summary: SensitivitySummary
    ) -> Dict[str, Any]:
        """Convert a SensitivitySummary to a JSON-serialisable dict."""
        return {
            "n_configs": summary.n_configs,
            "parameter_ranking": [
                {"parameter": p, "variance_contribution": round(v, 4)}
                for p, v in summary.parameter_ranking
            ],
            "interactions": [
                {
                    "pair": list(pair),
                    "interaction_strength": round(s, 4),
                }
                for pair, s in summary.interactions
            ],
            "best_config": {
                "base_rate": summary.best_config[0].base_rate if summary.best_config else None,
                "cost_fp": summary.best_config[0].cost_fp if summary.best_config else None,
                "cost_fn": summary.best_config[0].cost_fn if summary.best_config else None,
                "latency_budget_s": summary.best_config[0].latency_budget_s if summary.best_config else None,
                "escalation_min": summary.best_config[0].escalation_min if summary.best_config else None,
                "profile_set": summary.best_config[0].profile_set if summary.best_config else None,
                "expected_cost": round(summary.best_config[1], 4) if summary.best_config else None,
            }
            if summary.best_config
            else None,
            "worst_config": {
                "base_rate": summary.worst_config[0].base_rate if summary.worst_config else None,
                "cost_fp": summary.worst_config[0].cost_fp if summary.worst_config else None,
                "cost_fn": summary.worst_config[0].cost_fn if summary.worst_config else None,
                "latency_budget_s": summary.worst_config[0].latency_budget_s if summary.worst_config else None,
                "escalation_min": summary.worst_config[0].escalation_min if summary.worst_config else None,
                "profile_set": summary.worst_config[0].profile_set if summary.worst_config else None,
                "expected_cost": round(summary.worst_config[1], 4) if summary.worst_config else None,
            }
            if summary.worst_config
            else None,
            "action_distribution": {
                k: round(v, 4) for k, v in summary.action_distribution.items()
            },
        }

    def print_summary(self, summary: SensitivitySummary) -> None:
        """Print a human-readable sensitivity summary."""
        print(f"\n{'=' * 60}")
        print(f"Policy Sensitivity Analysis — {summary.n_configs} configurations")
        print(f"{'=' * 60}\n")

        print("Parameter Ranking (by variance contribution):")
        print(f"{'Parameter':<20} {'Contribution':<15}")
        print("-" * 35)
        for pname, contrib in summary.parameter_ranking:
            print(f"{pname:<20} {contrib:.2%}")

        print("\nTop Interactions:")
        print(f"{'Parameter Pair':<30} {'Strength':<15}")
        print("-" * 45)
        for (p1, p2), strength in summary.interactions[:3]:
            print(f"{p1} × {p2:<20} {strength:.2%}")

        if summary.best_config:
            cfg, cost = summary.best_config
            print(f"\nBest config (lowest expected cost = {cost:.4f}):")
            print(f"  Base rate: {cfg.base_rate}")
            print(f"  Cost ratio: {cfg.cost_fp}:{cfg.cost_fn}")
            print(f"  Latency budget: {cfg.latency_budget_s}s")
            print(f"  Escalation min: {cfg.escalation_min}")
            print(f"  Profile set: {cfg.profile_set}")

        if summary.worst_config:
            cfg, cost = summary.worst_config
            print(f"\nWorst config (highest expected cost = {cost:.4f}):")
            print(f"  Base rate: {cfg.base_rate}")
            print(f"  Cost ratio: {cfg.cost_fp}:{cfg.cost_fn}")
            print(f"  Latency budget: {cfg.latency_budget_s}s")
            print(f"  Escalation min: {cfg.escalation_min}")
            print(f"  Profile set: {cfg.profile_set}")

        print("\nOverall action distribution:")
        for action, frac in sorted(summary.action_distribution.items()):
            print(f"  {action}: {frac:.1%}")
        print()

    def save_results(
        self,
        results: List[SweepResult],
        summary: SensitivitySummary,
        path: str,
    ) -> None:
        """Save sweep results and summary to a JSON file.

        Args:
            results: Full sweep result list.
            summary: Computed sensitivity summary.
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "summary": self.summary_to_dict(summary),
            "sweep_results": [
                {
                    "base_rate": r.config.base_rate,
                    "cost_fp": r.config.cost_fp,
                    "cost_fn": r.config.cost_fn,
                    "latency_budget_s": r.config.latency_budget_s,
                    "escalation_min": r.config.escalation_min,
                    "profile_set": r.config.profile_set,
                    "selected_verifier": r.selected_verifier,
                    "action": r.action.value,
                    "expected_cost": round(r.expected_cost, 4),
                    "cost_saved_vs_reverse": round(r.cost_saved_vs_reverse, 4),
                    "cost_saved_vs_hold": round(r.cost_saved_vs_hold, 4),
                    "latency_ok": r.latency_ok,
                }
                for r in results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


# ── CLI entry point ────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Policy-layer sensitivity analysis"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller parameter grid (3 × 3 × 2 × 2 × 2 = 72 configs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/phase08_sensitivity_metrics.json",
        help="Output JSON path (default: output/phase08_sensitivity_metrics.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress during sweep",
    )
    args = parser.parse_args()

    analyzer = PolicySensitivityAnalyzer()
    results = analyzer.run_sweep(quick=args.quick, verbose=args.verbose)
    summary = analyzer.compute_sensitivity(results)
    analyzer.print_summary(summary)
    analyzer.save_results(results, summary, args.output)
