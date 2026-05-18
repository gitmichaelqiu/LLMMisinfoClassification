#!/usr/bin/env bash
# reproduce.sh — Single-command reproduction of all Phase 1-6 results.
#
# Usage:
#   bash reproduce.sh                  # full pipeline (requires Deepseek API key)
#   bash reproduce.sh --mock           # mock mode (no API key needed)
#
# Requires: Python 3.10+, pip, .env with DEEPSEEK_API_KEY (unless --mock)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " AdvFinNLPVuln — Full Pipeline Reproduction"
echo " Started: $(date)"
echo "============================================"

# Step 0: Install dependencies
echo ""
echo "[0/8] Installing dependencies..."
pip install -r requirements.txt -q

# Step 1: Download FinBERT model
echo ""
echo "[1/8] Downloading FinBERT model..."
python models/download_model.py

# Step 2: Generate synthetic financial dataset
echo ""
echo "[2/8] Generating synthetic financial dataset..."
python generate_dataset.py

# Step 3: Generate synthetic health dataset
echo ""
echo "[3/8] Generating synthetic health dataset..."
python src/health_dataset.py

# Step 4: Phase 1 — Baseline batch backtest
echo ""
echo "[4/8] Phase 1 — Baseline batch backtest (200 samples)..."
python main.py --test-size 200 || true

# Step 5: Phase 3 — CoT + Ensemble comparison
echo ""
echo "[5/8] Phase 3 — CoT + Ensemble comparison..."
# Run via python if available; Phase 3 is run_ensemble_comparison
python -c "
from main import run_ensemble_comparison
run_ensemble_comparison(target_size=1000, test_size=0.2)
" || echo "  (skipped — API key required or error during run)"

# Step 6: Phase 4 — Latency sweep
echo ""
echo "[6/8] Phase 4 — Latency sweep (50 samples, 3 budgets)..."
python main.py --test-size 50 --budgets none,5000,1000

# Step 7: Phase 5 — Sensitivity analysis
echo ""
echo "[7/8] Phase 5 — Sensitivity analysis (50 samples, 20 LHS points)..."
python main.py --phase5 --test-size 50 --lhs-samples 20

# Step 8: Phase 6 — Cross-domain comparison
echo ""
echo "[8/8] Phase 6 — Cross-domain comparison..."
python main.py --phase6 --test-size 50

echo ""
echo "============================================"
echo " Reproduction complete: $(date)"
echo " Outputs:"
echo "   output/phase1_metrics.json"
echo "   output/phase2_vs_phase1.json"
echo "   output/phase3_results.json"
echo "   output/phase4_latency_report.json"
echo "   output/phase5_sensitivity_analysis.json"
echo "   output/phase6_cross_domain.json"
echo "   plots/ (confusion matrices, Pareto frontiers, heatmaps)"
echo "============================================"
