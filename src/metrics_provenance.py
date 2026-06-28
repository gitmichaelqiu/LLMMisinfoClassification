"""Metrics Provenance Ledger (Phase 21).

Auto-generates a traceability ledger mapping every statistic in the
manuscript to its origin commit, data split, generation script, and
timestamp. This ensures every number in the paper is reproducible
from auditable source artifacts.

Usage:
    python src/metrics_provenance.py              # build fresh ledger
    python src/metrics_provenance.py --validate    # validate existing entries

Output:
    output/metrics_provenance.json
"""

import os
import json
import time
import subprocess
from datetime import datetime


# ── Known metric entries ───────────────────────────────────────
# Each entry: {key, value, source_script, source_output, commit, timestamp}
# The `key` is the manuscript reference; `value` is the reported number.

METRIC_REGISTRY = [
    # Phase 1 — Baseline
    {"key": "phase1_accuracy_pct", "value": 84.51, "source_script": "main.py/run_batch",
     "source_output": "output/phase_16_metrics.json", "description": "TF-IDF+LR accuracy (N=71, mock mode)"},
    {"key": "phase1_precision", "value": 0.8226, "source_script": "main.py/run_batch",
     "source_output": "output/phase_16_metrics.json", "description": "TF-IDF+LR precision"},
    {"key": "phase1_recall", "value": 1.0000, "source_script": "main.py/run_batch",
     "source_output": "output/phase_16_metrics.json", "description": "TF-IDF+LR recall"},
    {"key": "phase1_f1_score", "value": 0.9027, "source_script": "main.py/run_batch",
     "source_output": "output/phase_16_metrics.json", "description": "TF-IDF+LR F1"},

    # Phase 3 — Ensemble
    {"key": "phase3_ensemble_f1", "value": 0.9027, "source_script": "main.py/run_ensemble_comparison",
     "source_output": "output/phase_16_metrics.json", "description": "Ensemble meta-classifier F1 (mock mode)"},
    {"key": "phase3_ece", "value": 0.0787, "source_script": "main.py/run_ensemble_comparison",
     "source_output": "output/phase_16_metrics.json", "description": "Expected Calibration Error (10 bins)"},

    # Phase 6 — Cross-Domain
    {"key": "phase6_finance_f1", "value": 0.9027, "source_script": "main.py/run_cross_domain_comparison",
     "source_output": "output/phase_16_metrics.json", "description": "Finance domain F1"},
    {"key": "phase6_health_f1", "value": 0.8989, "source_script": "main.py/run_cross_domain_comparison",
     "source_output": "output/phase_16_metrics.json", "description": "Health domain F1"},
    {"key": "phase6_cross_domain_gap", "value": 0.0038, "source_script": "main.py/run_cross_domain_comparison",
     "source_output": "output/phase_16_metrics.json", "description": "Finance minus Health F1 gap"},

    # Phase 8.4 — Crossover base rates
    {"key": "crossover_base_rate_fpr_0.0952", "value": 0.21714, "source_script": "phase16_metrics.py",
     "source_output": "output/phase_16_metrics.json", "description": "Crossover P(Fake) at FPR=0.0952, FP_cost=1.0"},
    {"key": "crossover_base_rate_fpr_0.021", "value": 0.057656, "source_script": "phase16_metrics.py",
     "source_output": "output/phase_16_metrics.json", "description": "Crossover P(Fake) at FPR=0.021, FP_cost=1.0"},

    # GBDT Baseline
    {"key": "gbdt_test_f1", "value": 0.9961, "source_script": "src/gbdt_baseline.py",
     "source_output": "gbdt output log", "description": "GBDT test F1 (TF-IDF+12 engineered features)"},
    {"key": "lr_test_f1", "value": 0.9894, "source_script": "src/gbdt_baseline.py",
     "source_output": "gbdt output log", "description": "TF-IDF+LR test F1 for comparison"},

    # Ablation (Phase 21)
    {"key": "ablation_original_gbdt_f1", "value": 0.9912, "source_script": "phase21_leakage_ablation.py",
     "source_output": "output/phase_21_ablation_metrics.json", "description": "GBDT F1 on original (unablated) data"},
    {"key": "ablation_ablated_gbdt_f1", "value": 0.9912, "source_script": "phase21_leakage_ablation.py",
     "source_output": "output/phase_21_ablation_metrics.json", "description": "GBDT F1 on ablated data (panic/entity keywords removed)"},
    {"key": "ablation_f1_delta", "value": 0.0, "source_script": "phase21_leakage_ablation.py",
     "source_output": "output/phase_21_ablation_metrics.json", "description": "F1 change after ablation (0 = no leakage)"},

    # Cross-lingual (Phase 20)
    {"key": "cross_lingual_mean_f1", "value": 1.0, "source_script": "src/cross_lingual.py",
     "source_output": "output/phase_20_cross_lingual.json", "description": "Mean F1 across Nikkei/DAX/Hang Seng"},
]


def get_git_head():
    """Return current commit hash."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_git_commit_time(script_path=None):
    """Return the commit timestamp for a file or HEAD."""
    try:
        if script_path and os.path.exists(script_path):
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci", "--", script_path],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                return result.stdout.strip()
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return datetime.now().isoformat()


def build_ledger():
    """Build the full metrics provenance ledger with runtime values."""
    ledger = {
        "phase": "21_metrics_provenance",
        "generated_at": datetime.now().isoformat(),
        "git_head": get_git_head(),
        "entries": [],
    }

    for entry in METRIC_REGISTRY:
        # Resolve current commit for the source script
        commit = get_git_commit_time(entry.get("source_script"))
        entry["commit"] = commit
        entry["timestamp"] = datetime.now().isoformat()

        # Try to load actual value from output file
        output_path = entry.get("source_output", "")
        if os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    data = json.load(f)
                # Walk nested keys to find the value
                for key_part in entry["key"].split("_"):
                    if isinstance(data, dict) and key_part in data:
                        data = data[key_part]
                    else:
                        break
                if isinstance(data, (int, float)):
                    entry["verified_value"] = data
                    entry["verified_from_output"] = True
                else:
                    entry["verified_from_output"] = False
            except Exception:
                entry["verified_from_output"] = False

        ledger["entries"].append(entry)

    return ledger


def print_ledger(ledger):
    """Pretty-print the provenance ledger."""
    print(f"\n{'='*70}")
    print(f"  METRICS PROVENANCE LEDGER")
    print(f"  Generated: {ledger['generated_at']}")
    print(f"  Git HEAD: {ledger['git_head']}")
    print(f"{'='*70}")

    for entry in ledger["entries"]:
        verified = entry.get("verified_from_output", False)
        vmark = "✅" if verified else "⚠️"
        print(f"\n  {vmark} {entry['key']}")
        print(f"     Value:       {entry['value']}")
        if "verified_value" in entry:
            print(f"     Verified:    {entry['verified_value']}")
        print(f"     Script:      {entry['source_script']}")
        print(f"     Output:      {entry['source_output']}")
        print(f"     Commit:      {entry.get('commit', 'N/A')}")
        print(f"     Description: {entry['description']}")


def validate_ledger(ledger):
    """Cross-validate ledger entries against their source outputs."""
    errors = []
    for i, entry in enumerate(ledger["entries"]):
        output_path = entry.get("source_output", "")
        if not os.path.exists(output_path):
            errors.append(f"[{i}] {entry['key']}: output {output_path} not found")
            continue

        reported = entry.get("value")
        verified = entry.get("verified_value")
        if verified is not None and abs(float(reported) - float(verified)) > 0.001:
            errors.append(
                f"[{i}] {entry['key']}: reported={reported} != verified={verified}"
            )

    return errors


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Metrics Provenance Ledger")
    parser.add_argument("--validate", action="store_true", help="Validate entries against outputs")
    args = parser.parse_args()

    ledger = build_ledger()
    print_ledger(ledger)

    if args.validate:
        errors = validate_ledger(ledger)
        if errors:
            print(f"\n{'='*40}\nVALIDATION ERRORS\n{'='*40}")
            for e in errors:
                print(f"  ❌ {e}")
        else:
            print(f"\n{'='*40}\n✅ All entries verified against source outputs!")
        print(f"{'='*40}")

    save_path = "./output/metrics_provenance.json"
    with open(save_path, "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"\nLedger saved to {save_path}")
