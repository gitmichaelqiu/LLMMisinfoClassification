"""Hyperparameter optimizer for MFT verification arbitrage.

Grid search over the LLM confidence threshold to determine the optimal
cut-off for ESCALATE vs FAKE intervention. Per-liquidity-profile
optimization (High-Cap, Mid-Cap, Low-Cap) using the training set only.

Method:
1. Run the pipeline on the training set to collect raw LLM verdicts + confidence.
2. For each threshold t in [0.0, 0.1, ..., 0.9]:
   - Override FAKE→ESCALATE when confidence < t
   - Recompute P&L with filtered verdicts
3. Select the threshold that maximizes P&L saved for each liquidity profile.
4. Save optimal thresholds to output/optimal_thresholds.json.
"""

import os
import json
import time
import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict

from src.mft_simulator import MFTMarketSimulator, LIQUIDITY_PROFILES
from src.async_pipeline import MFTPipeline, DualRAGRetriever, VERDICT_FAKE, VERDICT_REAL, VERDICT_ESCALATE


@dataclass
class ThresholdResult:
    """Metrics for a single confidence threshold evaluation."""
    threshold: float
    total_pnl: float = 0.0
    total_pnl_saved: float = 0.0
    n_intervened: int = 0
    n_escalated: int = 0
    n_held: int = 0
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    fp_cost: float = 0.0
    tp_savings: float = 0.0


@dataclass
class OptimalThreshold:
    """Optimal threshold for a given liquidity profile."""
    liquidity_profile: str
    optimal_threshold: float
    max_pnl_saved: float
    metrics_at_optimal: dict = field(default_factory=dict)
    sweep_results: list = field(default_factory=list)


def _apply_threshold_filter(result_row, threshold, simulator):
    """Re-evaluate P&L for a single event with a confidence threshold applied.

    If verdict is FAKE but confidence < threshold, override to ESCALATE
    (don't intervene — the LLM isn't sure enough).

    Args:
        result_row: dict from pipeline results list
        threshold: float confidence threshold [0.0, 1.0]
        simulator: MFTMarketSimulator instance

    Returns:
        dict with filtered outcome, actual_pnl, pnl_saved
    """
    original_verdict = result_row["llm_verdict"]
    confidence = result_row["llm_confidence"]
    is_fake = result_row.get("is_fake_event")
    pos = result_row.get("position_size", 1000)

    # Apply threshold: if FAKE but low confidence, escalate instead
    if original_verdict == VERDICT_FAKE and confidence < threshold:
        effective_verdict = VERDICT_ESCALATE
    else:
        effective_verdict = original_verdict

    # Recompute P&L with the filtered verdict
    llm_intervened = (effective_verdict == VERDICT_FAKE)
    llm_escalated = (effective_verdict == VERDICT_ESCALATE)

    # Hold P&L
    hold_pnl = simulator.compute_hold_for_human_pnl(
        position_size=pos, is_fake=is_fake if is_fake is not None else True
    )
    # Intervene P&L
    intervene_pnl = simulator.compute_llm_intervene_pnl(
        position_size=pos, is_fake=is_fake if is_fake is not None else True,
        intervention_time=simulator.T1,
    )

    if llm_intervened and is_fake is True:
        actual_pnl = intervene_pnl["pnl"]
        pnl_saved = intervene_pnl["pnl"] - hold_pnl["pnl"]
        missed_profit = 0.0
        outcome = "true_positive"
    elif llm_intervened and is_fake is False:
        real_price_t1 = simulator.price_at(simulator.T1, is_fake=False)
        real_price_t2 = simulator.price_at(simulator.T2, is_fake=False)
        missed_profit = (real_price_t2 - real_price_t1) * pos
        actual_pnl = intervene_pnl["pnl"]
        pnl_saved = intervene_pnl["pnl"] - hold_pnl["pnl"]
        outcome = "false_positive"
    elif not llm_intervened and is_fake is True:
        actual_pnl = hold_pnl["pnl"]
        pnl_saved = 0.0
        missed_profit = 0.0
        outcome = "false_negative"
    elif not llm_intervened and is_fake is False:
        actual_pnl = hold_pnl["pnl"]
        pnl_saved = 0.0
        missed_profit = 0.0
        outcome = "true_negative"
    else:
        actual_pnl = intervene_pnl["pnl"] if llm_intervened else hold_pnl["pnl"]
        pnl_saved = 0.0
        missed_profit = 0.0
        outcome = "unknown"

    return {
        "effective_verdict": effective_verdict,
        "outcome": outcome,
        "actual_pnl": actual_pnl,
        "pnl_saved": pnl_saved,
        "missed_profit": missed_profit,
    }


def _compute_threshold_metrics(results_df, threshold, simulator):
    """Compute aggregate metrics for a single confidence threshold."""
    filtered = []
    for _, row in results_df.iterrows():
        f = _apply_threshold_filter(row.to_dict(), threshold, simulator)
        filtered.append(f)

    fdf = pd.DataFrame(filtered)

    tp = int((fdf["outcome"] == "true_positive").sum())
    fp = int((fdf["outcome"] == "false_positive").sum())
    tn = int((fdf["outcome"] == "true_negative").sum())
    fn = int((fdf["outcome"] == "false_negative").sum())

    total_pnl = float(fdf["actual_pnl"].sum())
    total_pnl_saved = float(fdf["pnl_saved"].sum())
    fp_cost = float(fdf[fdf["outcome"] == "false_positive"]["missed_profit"].sum())
    tp_savings = float(fdf[fdf["outcome"] == "true_positive"]["pnl_saved"].sum())

    n_total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

    n_intervened = int((fdf["effective_verdict"] == VERDICT_FAKE).sum())
    n_escalated = int((fdf["effective_verdict"] == VERDICT_ESCALATE).sum())
    n_held = int((fdf["effective_verdict"] == VERDICT_REAL).sum())

    return ThresholdResult(
        threshold=round(threshold, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_saved=round(total_pnl_saved, 2),
        n_intervened=n_intervened,
        n_escalated=n_escalated,
        n_held=n_held,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        accuracy=round(accuracy, 4),
        fp_cost=round(fp_cost, 2),
        tp_savings=round(tp_savings, 2),
    )


def run_threshold_grid_search(
    pipeline_results,
    liquidity_profile="mid_cap",
    thresholds=None,
):
    """Run grid search over confidence thresholds for a given liquidity profile.

    Args:
        pipeline_results: list of result dicts from MFTPipeline
        liquidity_profile: str key into LIQUIDITY_PROFILES
        thresholds: list of float thresholds to evaluate

    Returns:
        OptimalThreshold dataclass
    """
    if thresholds is None:
        thresholds = [round(i * 0.1, 1) for i in range(0, 10)]  # 0.0 to 0.9

    profile = LIQUIDITY_PROFILES.get(liquidity_profile, LIQUIDITY_PROFILES["mid_cap"])
    # Strip metadata keys not accepted by the simulator constructor
    sim_params = {k: v for k, v in profile.items() if k != "description"}
    simulator = MFTMarketSimulator(
        base_price=100.0,
        position_size=1000,
        **sim_params,
    )

    results_df = pd.DataFrame(pipeline_results)

    sweep = []
    for t in thresholds:
        tr = _compute_threshold_metrics(results_df, t, simulator)
        sweep.append(tr)

    # Find optimal: maximize total P&L saved
    best = max(sweep, key=lambda r: r.total_pnl_saved)

    return OptimalThreshold(
        liquidity_profile=liquidity_profile,
        optimal_threshold=best.threshold,
        max_pnl_saved=best.total_pnl_saved,
        metrics_at_optimal=asdict(best),
        sweep_results=[asdict(r) for r in sweep],
    )


def run_all_profiles_optimization(
    pipeline_results,
    output_path="./output/optimal_thresholds.json",
):
    """Run threshold grid search for all liquidity profiles and save results.

    Args:
        pipeline_results: list of result dicts from MFTPipeline
        output_path: JSON output path

    Returns:
        dict of profile -> OptimalThreshold
    """
    all_results = {}
    for profile in LIQUIDITY_PROFILES:
        print(f"\nOptimizing for {profile}...")
        opt = run_threshold_grid_search(pipeline_results, liquidity_profile=profile)
        all_results[profile] = asdict(opt)
        print(f"  Optimal threshold: {opt.optimal_threshold}")
        print(f"  Max P&L saved: ${opt.max_pnl_saved:,.2f}")
        print(f"  At optimal: TP={opt.metrics_at_optimal['true_positive']}, "
              f"FP={opt.metrics_at_optimal['false_positive']}, "
              f"FN={opt.metrics_at_optimal['false_negative']}")

    save_data = {
        "phase": "5_hyperparameter_optimization",
        "timestamp": time.ctime(),
        "method": "grid_search_confidence_threshold",
        "description": "Optimal confidence threshold for FAKE→ESCALATE override per liquidity profile",
        "results": all_results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nOptimal thresholds saved to {output_path}")

    return all_results


def run_optimization_backtest(
    temporal_events_path="./output/temporal_events.csv",
    model="deepseek",
    position_size=1000,
    base_price=100.0,
    thinking="enabled",
    use_system0=False,
    max_train_events=None,
    output_path="./output/optimal_thresholds.json",
):
    """Full end-to-end: run pipeline on training set, then optimize thresholds.

    Uses the training set only (no test set leakage). Run the LLM pipeline
    on training events, collect verdicts, then perform threshold grid search.

    Args:
        temporal_events_path: Path to temporal_events.csv
        model: LLM backend
        position_size: Shares per trade
        base_price: Base entry price
        thinking: DeepSeek thinking mode
        use_system0: Enable System 0 pre-filter
        max_train_events: Cap training set size
        output_path: Output path for optimal thresholds

    Returns:
        dict of profile -> OptimalThreshold
    """
    from dotenv import load_dotenv
    from openai import OpenAI
    from transformers import pipeline

    load_dotenv()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    print("=" * 60)
    print("PHASE 5: HYPERPARAMETER OPTIMIZATION")
    print(f"Model: {model} | Training set optimization")
    print("=" * 60)

    # Load training set
    from src.data_splitter import temporal_train_test_split
    train_ids, test_ids, split_meta = temporal_train_test_split(
        temporal_events_path=temporal_events_path,
        test_ratio=0.2,
    )
    events_df = pd.read_csv(temporal_events_path)
    train_df = events_df[events_df["event_id"].isin(train_ids)]
    if max_train_events and len(train_df) > max_train_events:
        train_df = train_df.head(max_train_events)
    print(f"Training set: {len(train_df)} events")

    # Initialize models
    print("\nInitializing FinBERT...")
    local_path = os.path.join(os.getcwd(), "models", "finbert")
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            finbert = pipeline("sentiment-analysis", model=local_path, tokenizer=local_path)
        else:
            finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        print(f"  FinBERT unavailable: {e}")
        finbert = None

    client = None
    if deepseek_key and deepseek_key != "your_actual_api_key_here":
        client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com",
            timeout=30.0,
        )
        print("DeepSeek client ready")
    else:
        print("No API key — running in mock mode")

    print("Initializing Dual RAG...")
    dual_rag = DualRAGRetriever()
    dual_rag.ensure_index()

    # Build pipeline
    pipeline = MFTPipeline(
        finbert_model=finbert,
        dual_rag=dual_rag,
        deepseek_client=client,
        deepseek_key=deepseek_key,
        model=model,
        position_size=position_size,
        base_price=base_price,
        use_system0=use_system0,
        thinking=thinking,
    )

    # Run pipeline on training set
    print(f"\nProcessing {len(train_df)} training events...")
    results = pipeline.process_events(train_df)
    pipeline.cleanup()

    # Run optimization over thresholds
    print("\nRunning threshold grid search...")
    all_results = run_all_profiles_optimization(results, output_path=output_path)

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--train-size", type=int, default=None, help="Cap training set size")
    parser.add_argument("--output", default="./output/optimal_thresholds.json")
    parser.add_argument("--position-size", type=int, default=1000)
    parser.add_argument("--base-price", type=float, default=100.0)
    parser.add_argument("--thinking", default="enabled", choices=["enabled", "disabled"])
    parser.add_argument("--system0", action="store_true")
    args = parser.parse_args()

    run_optimization_backtest(
        model=args.model,
        position_size=args.position_size,
        base_price=args.base_price,
        thinking=args.thinking,
        use_system0=args.system0,
        max_train_events=args.train_size,
        output_path=args.output,
    )
