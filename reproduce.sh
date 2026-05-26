#!/usr/bin/env bash
# reproduce.sh — Single-command reproduction of all Phase 1-7 results and tradeoff analysis.
#
# Usage:
#   bash reproduce.sh                  # full pipeline (requires Deepseek API key)
#   bash reproduce.sh --mock           # mock mode (no API key needed)
#
# Requires: Python 3.10+, pip, .env with DEEPSEEK_API_KEY (unless --mock)

set -euo pipefail

export PYTHONUNBUFFERED=1

# Check for mock flag
if [[ "${1:-}" == "--mock" ]]; then
    echo "Running in MOCK mode (overriding DEEPSEEK_API_KEY)"
    export DEEPSEEK_API_KEY="your_actual_api_key_here"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " AdvFinNLPVuln — Full Pipeline Reproduction"
echo " Started: $(date)"
echo "============================================"

# Step 0: Check for optional Phase 7 dependencies
PHASE7_AVAIL=true
python -c "import hftbacktest" 2>/dev/null || PHASE7_AVAIL=false
python -c "import vectorbt" 2>/dev/null || PHASE7_AVAIL=false

# Step 0: Install dependencies
echo ""
echo "[0/13] Installing dependencies..."
pip install -r requirements.txt -q
if $PHASE7_AVAIL || pip install hftbacktest vectorbt -q 2>/dev/null; then
    PHASE7_AVAIL=true
    echo "  Phase 7 dependencies (hftbacktest, vectorbt) available."
else
    echo "  Phase 7 dependencies skipped (optional)."
fi

# Step 1: Download FinBERT model
echo ""
echo "[1/13] Downloading FinBERT model..."
python models/download_model.py

# Step 2: Generate synthetic financial dataset
echo ""
echo "[2/13] Generating synthetic financial dataset..."
python generate_dataset.py

# Step 3: Generate synthetic health dataset
echo ""
echo "[3/13] Generating synthetic health dataset..."
python src/health_dataset.py

# Step 4: Phase 1 — Baseline batch backtest
echo ""
echo "[4/13] Phase 1 — Baseline batch backtest (200 samples)..."
python main.py --test-size 200 || true

# Step 5: Phase 3 — CoT + Ensemble comparison
echo ""
echo "[5/13] Phase 3 — CoT + Ensemble comparison (Thinking Enabled & Disabled)..."
python -c "
from main import run_ensemble_comparison
run_ensemble_comparison(target_size=1000, test_size=0.2, thinking='enabled')
" || echo "  (skipped thinking=enabled — API key required or error during run)"
python -c "
from main import run_ensemble_comparison
run_ensemble_comparison(target_size=1000, test_size=0.2, thinking='disabled')
" || echo "  (skipped thinking=disabled — API key required or error during run)"

# Step 6: Phase 4 — Latency sweep
echo ""
echo "[6/13] Phase 4 — Latency sweep (Thinking Enabled & Disabled)..."
python main.py --test-size 50 --budgets none,5000,1000 --thinking enabled
python main.py --test-size 50 --budgets none,5000,1000 --thinking disabled

# Step 7: Phase 5 — Sensitivity analysis
echo ""
echo "[7/13] Phase 5 — Sensitivity analysis (Thinking Enabled & Disabled)..."
python main.py --phase5 --test-size 50 --lhs-samples 20 --thinking enabled
python main.py --phase5 --test-size 50 --lhs-samples 20 --thinking disabled

# Step 8: Phase 6 — Cross-domain comparison
echo ""
echo "[8/13] Phase 6 — Cross-domain comparison (Thinking Enabled & Disabled)..."
python main.py --phase6 --test-size 50 --thinking enabled
python main.py --phase6 --test-size 50 --thinking disabled

# Step 9: Phase 7a — Execution realism analysis
echo ""
echo "[9/13] Phase 7a — Execution realism analysis (Thinking Enabled & Disabled)..."
if $PHASE7_AVAIL; then
    python main.py --phase7a --test-size 50 --thinking enabled
    python main.py --phase7a --test-size 50 --thinking disabled
else
    echo "  (skipped — hftbacktest/vectorbt not installed)"
fi

# Step 10: Phase 7b — vectorbt signal sweep
echo ""
echo "[10/13] Phase 7b — vectorbt signal sweep (Thinking Enabled & Disabled)..."
if $PHASE7_AVAIL; then
    python main.py --phase7b --test-size 50 --thinking enabled
    python main.py --phase7b --test-size 50 --thinking disabled
else
    echo "  (skipped — hftbacktest/vectorbt not installed)"
fi

# Step 11: Base Rate Analysis
echo ""
echo "[11/13] Step 11 — Base Rate Fallacy Analysis..."
python src/base_rate_analysis.py

# Step 12: Verify-First vs. Trade-First Tradeoff Analysis
echo ""
echo "[12/13] Step 12 — Verify-First vs. Trade-First Tradeoff Analysis..."
python src/verify_first_model.py

# Step 13: Phase 8 — Liquidity sensitivity study
echo ""
echo "[13/13] Step 13 — Liquidity Sensitivity Study (Thinking Enabled & Disabled)..."
python main.py --phase8 --test-size 50 --thinking enabled
python main.py --phase8 --test-size 50 --thinking disabled

echo ""
echo "============================================"
echo " Reproduction complete: $(date)"
echo " Outputs:"
echo "   output/phase1_metrics.json"
echo "   output/phase2_vs_phase1.json"
echo "   output/*_phase3_results.json"
echo "   output/*_phase4_latency_report.json"
echo "   output/*_phase5_sensitivity_analysis.json"
echo "   output/*_phase6_cross_domain.json"
echo "   output/*_system0_phase7a_execution_realism.json"
echo "   output/*_no_system0_phase7a_execution_realism.json"
echo "   output/*_system0_phase7b_signal_sweep.json"
echo "   output/*_no_system0_phase7b_signal_sweep.json"
echo "   output/*_phase8_liquidity_sensitivity.json"
echo "   output/base_rate_analysis.json"
echo "   output/verify_first_tradeoff.json"
echo "   plots/ (confusion matrices, Pareto frontiers, heatmaps, *_ideal_vs_realized_pnl, *_vectorbt_heatmaps, *_phase8_liquidity_heatmaps, base_rate_analysis, verify_first_tradeoff)"
echo "============================================"
