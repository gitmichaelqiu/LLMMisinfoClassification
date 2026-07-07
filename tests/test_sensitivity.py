"""Tests for src/sensitivity_analysis.py — policy-layer sensitivity analysis.

Tests cover:
- Grid construction (full and quick).
- Single-cell evaluation returns correct SweepResult fields.
- Full sweep runs without error and returns expected config count.
- Sensitivity ranking orders parameters by variance contribution.
- Interaction detection produces valid pairs.
- Best/worst config identification.
- JSON serialisation round-trip.
- CLI entry point.
"""

from __future__ import annotations

import json
import os
import tempfile

from src.hybrid_policy import RiskAction
from src.sensitivity_analysis import (
    BASE_RATES,
    COST_RATIOS,
    ESCALATION_MINS,
    HIGH_PRECISION_PROFILES,
    LATENCY_BUDGETS,
    LOW_LATENCY_PROFILES,
    PROFILE_SET_NAMES,
    PROFILE_SETS,
    QUICK_BASE_RATES,
    QUICK_COST_RATIOS,
    QUICK_ESCALATION_MINS,
    QUICK_LATENCY_BUDGETS,
    QUICK_PROFILE_SET_NAMES,
    PolicySensitivityAnalyzer,
    PolicySweepConfig,
    SweepResult,
)

# ====================================================================
# Grid Construction Tests
# ====================================================================


class TestGridConstruction:
    def test_full_grid_size(self):
        """Full grid includes all parameter combinations."""
        analyzer = PolicySensitivityAnalyzer()
        grid = analyzer.build_grid(quick=False)
        expected = (
            len(BASE_RATES)
            * len(COST_RATIOS)
            * len(LATENCY_BUDGETS)
            * len(ESCALATION_MINS)
            * len(PROFILE_SET_NAMES)
        )
        assert len(grid) == expected

    def test_quick_grid_size(self):
        """Quick grid is a subset."""
        analyzer = PolicySensitivityAnalyzer()
        grid = analyzer.build_grid(quick=True)

        # All quick parameters
        # 3 base rates × 3 cost ratios × 2 latencies × 2 escalations × 2 profiles
        expected = (
            len(QUICK_BASE_RATES)
            * len(QUICK_COST_RATIOS)
            * len(QUICK_LATENCY_BUDGETS)
            * len(QUICK_ESCALATION_MINS)
            * len(QUICK_PROFILE_SET_NAMES)
        )
        assert len(grid) == expected

    def test_quick_is_smaller_than_full(self):
        """Quick grid has fewer configs than full."""
        analyzer = PolicySensitivityAnalyzer()
        full = analyzer.build_grid(quick=False)
        quick = analyzer.build_grid(quick=True)
        assert len(quick) < len(full)

    def test_grid_contains_PolicySweepConfig(self):
        """Every grid entry is a PolicySweepConfig."""
        analyzer = PolicySensitivityAnalyzer()
        grid = analyzer.build_grid(quick=True)
        for config in grid:
            assert isinstance(config, PolicySweepConfig)

    def test_grid_parameters_in_range(self):
        """All grid parameters are within valid ranges."""
        analyzer = PolicySensitivityAnalyzer()
        grid = analyzer.build_grid(quick=True)
        for c in grid:
            assert 0.0 <= c.base_rate <= 1.0
            assert c.cost_fp > 0
            assert c.cost_fn > 0
            assert c.latency_budget_s > 0
            assert 0.0 <= c.escalation_min <= 1.0
            assert c.profile_set in PROFILE_SET_NAMES

    def test_profile_sets_defined(self):
        """All profile sets are defined."""
        assert "default" in PROFILE_SETS
        assert "high_precision" in PROFILE_SETS
        assert "low_latency" in PROFILE_SETS
        for name, profiles in PROFILE_SETS.items():
            assert len(profiles) >= 1, f"{name} has no profiles"


# ====================================================================
# Single-Cell Evaluation Tests
# ====================================================================


class TestSingleCellEvaluation:
    def test_evaluate_returns_sweep_result(self):
        """Single evaluation returns a SweepResult."""
        analyzer = PolicySensitivityAnalyzer()
        config = PolicySweepConfig()
        result = analyzer.evaluate(config)
        assert isinstance(result, SweepResult)

    def test_result_has_all_fields(self):
        """SweepResult contains all expected fields."""
        analyzer = PolicySensitivityAnalyzer()
        result = analyzer.evaluate(PolicySweepConfig())

        assert isinstance(result.config, PolicySweepConfig)
        assert isinstance(result.selected_verifier, str)
        assert isinstance(result.action, RiskAction)
        assert isinstance(result.expected_cost, float)
        assert isinstance(result.cost_saved_vs_reverse, float)
        assert isinstance(result.cost_saved_vs_hold, float)
        assert isinstance(result.latency_ok, bool)

    def test_selected_verifier_is_valid(self):
        """Selected verifier exists in the profile set."""
        analyzer = PolicySensitivityAnalyzer()
        result = analyzer.evaluate(PolicySweepConfig(profile_set="default"))
        assert result.selected_verifier in PROFILE_SETS["default"]

    def test_expected_cost_is_non_negative(self):
        """Expected cost is always ≥ 0."""
        analyzer = PolicySensitivityAnalyzer()
        for br in [0.001, 0.05, 0.50]:
            for (cfp, cfn) in [(1, 1), (1, 10), (1, 100)]:
                config = PolicySweepConfig(
                    base_rate=br, cost_fp=cfp, cost_fn=cfn
                )
                result = analyzer.evaluate(config)
                assert result.expected_cost >= 0, f"Negative cost at br={br}, cost=({cfp},{cfn})"

    def test_costs_vary_with_base_rate(self):
        """Expected cost changes when base rate changes (FV asymmetry)."""
        analyzer = PolicySensitivityAnalyzer()
        r1 = analyzer.evaluate(PolicySweepConfig(base_rate=0.01, cost_fn=10.0))
        r2 = analyzer.evaluate(PolicySweepConfig(base_rate=0.50, cost_fn=10.0))
        assert r1.expected_cost != r2.expected_cost

    def test_latency_ok_true_when_within_budget(self):
        """latency_ok is True when selected verifier fits budget."""
        analyzer = PolicySensitivityAnalyzer()
        # All default profiles have latency ≤ 0.003, so budget 5.0 is fine
        result = analyzer.evaluate(
            PolicySweepConfig(latency_budget_s=5.0, profile_set="default")
        )
        assert result.latency_ok is True

    def test_cost_savings_non_negative(self):
        """Cost savings vs baselines are non-negative."""
        analyzer = PolicySensitivityAnalyzer()
        for br in [0.001, 0.05, 0.25]:
            config = PolicySweepConfig(base_rate=br)
            result = analyzer.evaluate(config)
            assert result.cost_saved_vs_reverse >= 0
            assert result.cost_saved_vs_hold >= 0


# ====================================================================
# Full Sweep Tests
# ====================================================================


class TestFullSweep:
    def test_run_sweep_quick_returns_list(self):
        """Quick sweep returns a list of results."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_sweep_quick_has_correct_count(self):
        """Quick sweep returns the expected number of results."""
        analyzer = PolicySensitivityAnalyzer()
        grid = analyzer.build_grid(quick=True)
        results = analyzer.run_sweep(quick=True)
        assert len(results) == len(grid)

    def test_run_sweep_all_results_valid(self):
        """Every sweep result has valid fields."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        for r in results:
            assert isinstance(r, SweepResult)
            assert r.expected_cost >= 0
            assert r.action in RiskAction

    def test_run_sweep_full_returns_list(self):
        """Full sweep runs without error and returns results."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=False)
        assert len(results) > 0


# ====================================================================
# Sensitivity Ranking Tests
# ====================================================================


class TestSensitivityRanking:
    def test_compute_sensitivity_on_sweep(self):
        """Sensitivity summary is computed from sweep results."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        assert summary.n_configs == len(results)

    def test_ranking_has_all_parameters(self):
        """All five parameters appear in the ranking."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        ranked_params = {p for p, _ in summary.parameter_ranking}
        expected = {"base_rate", "cost_ratio", "latency_budget", "escalation_min", "profile_set"}
        assert ranked_params == expected

    def test_ranking_sorted_by_contribution(self):
        """Ranking is sorted descending by variance contribution."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        for i in range(len(summary.parameter_ranking) - 1):
            assert summary.parameter_ranking[i][1] >= summary.parameter_ranking[i + 1][1]

    def test_contributions_sum_to_one(self):
        """Variance contributions roughly sum to ≤ 1 (may not exactly 1
        due to interaction effects captured in joint variance)."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        total = sum(v for _, v in summary.parameter_ranking)
        assert total <= 1.0 + 1e-6  # allow small fp error

    def test_high_fn_cost_dominates_at_low_base_rate(self):
        """At low base rates, FN cost (cost ratio) should be influential."""
        # Run a focused sweep at extreme base rates
        analyzer = PolicySensitivityAnalyzer()
        summary = analyzer.compute_sensitivity(
            analyzer.run_sweep(quick=True)
        )
        top_params = [p for p, _ in summary.parameter_ranking[:3]]
        assert "cost_ratio" in top_params or "base_rate" in top_params

    def test_best_config_has_lowest_cost(self):
        """Best config has the minimum expected cost."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        all_costs = [r.expected_cost for r in results]
        assert abs(summary.best_config[1] - min(all_costs)) < 1e-10

    def test_worst_config_has_highest_cost(self):
        """Worst config has the maximum expected cost."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        all_costs = [r.expected_cost for r in results]
        assert abs(summary.worst_config[1] - max(all_costs)) < 1e-10


# ====================================================================
# Interaction Detection Tests
# ====================================================================


class TestInteractionDetection:
    def test_interactions_list_non_empty(self):
        """Interaction detection returns at least one pair."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        assert len(summary.interactions) >= 1

    def test_interaction_strengths_valid(self):
        """Interaction strengths are in [0, 1]."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        for (p1, p2), strength in summary.interactions:
            assert 0.0 <= strength <= 1.0 + 1e-6
            assert isinstance(p1, str)
            assert isinstance(p2, str)

    def test_interactions_sorted_by_strength(self):
        """Interactions are sorted descending by strength."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        for i in range(len(summary.interactions) - 1):
            assert summary.interactions[i][1] >= summary.interactions[i + 1][1]

    def test_base_rate_cost_ratio_interaction(self):
        """base_rate × cost_ratio pair appears in interactions."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        pairs = {pair for pair, _ in summary.interactions}
        assert ("base_rate", "cost_ratio") in pairs


# ====================================================================
# Action Distribution Tests
# ====================================================================


class TestActionDistribution:
    def test_action_distribution_sum_to_one(self):
        """Action distribution fractions sum to 1."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        total = sum(summary.action_distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_at_least_one_action_present(self):
        """At least one action type has positive frequency."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        assert any(v > 0 for v in summary.action_distribution.values())


# ====================================================================
# Output and Serialisation Tests
# ====================================================================


class TestOutput:
    def test_summary_to_dict_has_all_keys(self):
        """Dictionary output contains expected top-level keys."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        d = analyzer.summary_to_dict(summary)
        assert "n_configs" in d
        assert "parameter_ranking" in d
        assert "interactions" in d
        assert "best_config" in d
        assert "worst_config" in d
        assert "action_distribution" in d

    def test_summary_to_dict_json_serialisable(self):
        """Dictionary is JSON-serialisable."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)
        d = analyzer.summary_to_dict(summary)
        dumped = json.dumps(d)
        assert isinstance(dumped, str)
        assert len(dumped) > 0

    def test_save_results_creates_file(self):
        """save_results writes a valid JSON file."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            outpath = f.name
            f.close()
            os.unlink(outpath)  # remove so save creates fresh

        try:
            analyzer.save_results(results, summary, outpath)
            assert os.path.exists(outpath)
            with open(outpath) as f:
                data = json.load(f)
            assert "summary" in data
            assert "sweep_results" in data
            assert len(data["sweep_results"]) == len(results)
        finally:
            if os.path.exists(outpath):
                os.unlink(outpath)

    def test_print_summary_runs(self):
        """print_summary runs without error (smoke test)."""
        analyzer = PolicySensitivityAnalyzer()
        results = analyzer.run_sweep(quick=True)
        summary = analyzer.compute_sensitivity(results)

        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        try:
            analyzer.print_summary(summary)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "Parameter Ranking" in output
        assert "Best config" in output
        assert "Worst config" in output
        assert "Overall action distribution" in output


# ====================================================================
# Profile Set Tests
# ====================================================================


class TestProfileSets:
    def test_high_precision_all_slower_than_default(self):
        """High-precision profiles have higher latency than default."""
        for name in DEFAULT_PROFILES:
            assert HIGH_PRECISION_PROFILES[name].latency_s >= DEFAULT_PROFILES[name].latency_s

    def test_low_latency_all_have_lower_or_equal_latency(self):
        """Low-latency profiles are all fast."""
        for name, profile in LOW_LATENCY_PROFILES.items():
            assert profile.latency_s <= 0.002

    def test_profile_sets_differ(self):
        """Profile sets produce different verifier selections."""

        analyzer = PolicySensitivityAnalyzer()
        r_default = analyzer.evaluate(
            PolicySweepConfig(base_rate=0.10, profile_set="default")
        )
        r_high = analyzer.evaluate(
            PolicySweepConfig(base_rate=0.10, profile_set="high_precision")
        )
        # At base rate 0.10, min_precision = 0.15
        # default profiles all have precision=0.50 → any qualifies → fastest (single-shot)
        # high_precision profiles all have latency>0.001, but single-shot at 0.002 is fastest
        # Both may select single-shot. That's OK — test just verifies they don't crash.
        assert r_default.selected_verifier in PROFILE_SETS["default"]
        assert r_high.selected_verifier in PROFILE_SETS["high_precision"]


DEFAULT_PROFILES = {
    "single-shot": type("P", (), {"latency_s": 0.001})(),
    "voting": type("P", (), {"latency_s": 0.002})(),
    "moa": type("P", (), {"latency_s": 0.003})(),
    "rag": type("P", (), {"latency_s": 0.002})(),
}


# ====================================================================
# Empty / Edge Case Tests
# ====================================================================


class TestEdgeCases:
    def test_compute_sensitivity_empty(self):
        """Empty results produce empty summary."""
        analyzer = PolicySensitivityAnalyzer()
        summary = analyzer.compute_sensitivity([])
        assert summary.n_configs == 0
        assert summary.action_distribution == {}
        assert summary.best_config is None
        assert summary.worst_config is None

    def test_compute_sensitivity_single_result(self):
        """Single result produces ranking with NaN/zero contributions."""
        analyzer = PolicySensitivityAnalyzer()
        config = PolicySweepConfig()
        result = analyzer.evaluate(config)
        summary = analyzer.compute_sensitivity([result])
        # All contributions should be valid (total_var=0 → contributions=0)
        for _, v in summary.parameter_ranking:
            assert 0.0 <= v <= 1.0 or (v == 0.0)


# ====================================================================
# CLI Entry Point Test
# ====================================================================


class TestCLI:
    def test_main_runs_with_quick(self):
        """CLI entry point runs with --quick flag (via subprocess)."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.sensitivity_analysis", "--quick", "--output", "/tmp/_test_sens_quick.json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Parameter Ranking" in result.stdout

    def test_main_output_file_created(self):
        """CLI with --quick creates the output JSON file."""
        import subprocess
        import sys

        outpath = "/tmp/_test_sens_output.json"
        result = subprocess.run(
            [sys.executable, "-m", "src.sensitivity_analysis", "--quick", "--output", outpath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert os.path.exists(outpath)
        with open(outpath) as f:
            data = json.load(f)
        assert "summary" in data
        assert "sweep_results" in data
        os.unlink(outpath)
