#!/usr/bin/env python3
"""
Full pipeline re-execution with Phase 10-15 changes.

Steps:
  1. Historical Backtest (using HistoricalRAGBuilder + Voting)
  2. Adversarial Comparison (Single-Shot vs MoA)
  3. MFT Sensitivity Analysis (LHS over T1, T2, FP-cost)

Uses the canonical src.* module paths. Falls back gracefully in mock mode.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("DEEPSEEK_API_KEY")
client_avail = bool(key and key != "your_actual_api_key_here")
print(f"API key: {'YES' if client_avail else 'NO (mock mode)'}")
print()

OUTPUT_DIR = "./output/phase_replication"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# STEP 1: HISTORICAL BACKTEST
# ═════════════════════════════════════════════════════════════════════

def step1_historical():
    print("=" * 60)
    print("  STEP 1: HISTORICAL BACKTEST")
    print("=" * 60)

    from src.historical_rag_builder import HistoricalRAGBuilder, HistoricalDualRAGRetriever
    from src.async_pipeline import MFTPipeline
    from src.data_splitter import temporal_train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # ── Build artifacts ──
    builder = HistoricalRAGBuilder()
    builder.print_event_summary()

    corpus = builder.build_pre_t0_corpus(
        output_path=os.path.join(OUTPUT_DIR, "historical_pret0_corpus.json")
    )
    builder.generate_historical_social_stream(
        output_path=os.path.join(OUTPUT_DIR, "historical_social_stream.csv")
    )
    events_path = os.path.join(OUTPUT_DIR, "historical_temporal_events.csv")
    builder.build_historical_events_csv(output_path=events_path)
    builder.validate_no_lookahead(corpus)

    events_df = pd.read_csv(events_path)
    print(f"Loaded {len(events_df)} historical events")

    # ── Train heuristic fallback from all events (mock-mode survival) ──
    heuristic_predictor = None
    if not client_avail:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        clf = LogisticRegression(max_iter=1000, random_state=42)
        texts = events_df["T0_headline"].fillna("")
        labels = events_df["T2_human_verdict"]
        X = vectorizer.fit_transform(texts)
        clf.fit(X, labels)
        def _pred(c):
            return int(clf.predict(vectorizer.transform([c]))[0])
        heuristic_predictor = _pred
        print(f"  Heuristic baseline trained on {len(texts)} events")

    # ── Init RAG ──
    rag = HistoricalDualRAGRetriever(
        pre_t0_corpus=corpus,
        social_stream_path=os.path.join(OUTPUT_DIR, "historical_social_stream.csv"),
    )
    rag.ensure_index()

    # ── Pipeline (uses single-shot by default) ──
    from transformers import pipeline as hf_pipeline
    from openai import OpenAI

    local_path = os.path.join(os.getcwd(), "models", "finbert")
    try:
        if os.path.exists(local_path) and os.listdir(local_path):
            finbert = hf_pipeline("sentiment-analysis", model=local_path, tokenizer=local_path)
        else:
            finbert = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        print(f"  FinBERT unavailable: {e}")
        finbert = None

    deepseek_client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=30.0) if client_avail else None

    pipeline = MFTPipeline(
        finbert_model=finbert,
        dual_rag=rag,
        deepseek_client=deepseek_client,
        deepseek_key=key,
        position_size=1000,
        base_price=100.0,
        heuristic_predictor=heuristic_predictor,
    )

    print(f"\nProcessing {len(events_df)} historical events...")
    t0 = time.time()
    results = pipeline.process_events(events_df)
    elapsed = time.time() - t0
    report = pipeline.generate_report(output_dir=OUTPUT_DIR, plots_dir="./plots")
    pipeline.cleanup()

    # Add historical metadata
    report["phase"] = "10_historical_backtest"
    report["dataset"] = "historical_hoaxes.json"

    hist_details = []
    for r in results:
        eid = r["event_id"]
        ev = builder.event_map.get(eid, {})
        hist_details.append({
            "event_id": eid,
            "title": ev.get("title", ""),
            "llm_verdict": r["llm_verdict_label"],
            "llm_confidence": r["llm_confidence"],
            "outcome": r["outcome"],
            "pnl_saved": r["pnl_saved"],
            "actual_pnl": r["actual_pnl"],
        })
    report["historical_event_details"] = hist_details

    out_path = os.path.join(OUTPUT_DIR, "step1_historical_voting.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  RESULTS:")
    print(f"  Time: {elapsed:.1f}s ({len(results)} events)")
    print(f"  Accuracy: {report['accuracy_metrics']['accuracy']:.3f}")
    print(f"  Precision: {report['accuracy_metrics']['precision']:.3f}")
    print(f"  Recall: {report['accuracy_metrics']['recall']:.3f}")
    print(f"  Total P&L: ${report['pnl_metrics']['total_pnl']:,.2f}")
    print(f"Step 1 results -> {out_path}\n")
    return report


# ═════════════════════════════════════════════════════════════════════
# STEP 2: ADVERSARIAL COMPARISON (Single-Shot vs MoA)
# ═════════════════════════════════════════════════════════════════════

def step2_adversarial():
    print("=" * 60)
    print("  STEP 2: ADVERSARIAL COMPARISON (Single-Shot vs MoA)")
    print("=" * 60)

    from src.stress_test import run_full_stress_test

    bot_levels = [0.0, 0.25, 0.50, 0.75]
    max_test = 10  # 5 REAL + 5 FAKE for cost

    all_metrics = {}

    for use_moa, mode_name in [(False, "single_shot"), (True, "moa")]:
        print(f"\n--- Mode: {mode_name} ---")
        metrics = run_full_stress_test(
            bot_levels=bot_levels,
            max_test_events=max_test,
            use_moa=use_moa,
            output_path=os.path.join(OUTPUT_DIR, f"adversarial_{mode_name}.json"),
        )
        all_metrics[mode_name] = metrics

        for m in metrics:
            print(f"  Bot {m['bot_pct']:.0%}: Prec={m['precision']:.3f}  "
                  f"Rec={m['recall']:.3f}  P&L=${m['total_pnl']:,.0f}")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  {'COMPARISON TABLE':^66}")
    print(f"{'='*70}")
    print(f"  {'Intensity':<12} {'Method':<16} {'Prec':<8} {'Rec':<8} {'P&L Saved':<12}")
    print(f"  {'-'*56}")
    for i, bot_pct in enumerate(bot_levels):
        for mode_name in ["single_shot", "moa"]:
            m = all_metrics[mode_name][i] if len(all_metrics[mode_name]) > i else {}
            label = mode_name.replace("_", " ").title()
            prec = m.get("precision", 0)
            rec = m.get("recall", 0)
            pnl = m.get("total_pnl_saved", 0)
            print(f"  {f'{bot_pct:.0%}':<12} {label:<16} "
                  f"{prec:<8.3f} {rec:<8.3f} ${pnl:<8,.0f}")

    out_path = os.path.join(OUTPUT_DIR, "step2_adversarial_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nStep 2 results -> {out_path}\n")
    return all_metrics


# ═════════════════════════════════════════════════════════════════════
# STEP 3: MFT SENSITIVITY ANALYSIS (LHS over T1, T2, FP-cost)
# ═════════════════════════════════════════════════════════════════════

def step3_sensitivity():
    """MFT Sensitivity via LHS over (T1, T2, FP-cost) using pre-collected results.

    NOTE: This replays pipeline results through the simulator with varying
    timing parameters. It does NOT re-run the LLM with dynamically regenerated
    RAG at each T1 cutoff (Phase 13 time-aware sensitivity). For a full
    time-aware sweep, use src.mft_sensitivity with dynamically re-queried results.
    """
    print("=" * 60)
    print("  STEP 3: MFT SENSITIVITY ANALYSIS (LHS)")
    print("=" * 60)

    # First, run a quick pipeline to get results
    from src.async_pipeline import run_mft_backtest

    report = run_mft_backtest(
        test_ratio=0.2,
        position_size=1000,
        base_price=100.0,
        max_events=10,
        thinking="enabled",
    )

    if report and report.get("n_events", 0) > 0:
        # Run LHS sensitivity on these results
        from src.mft_sensitivity import run_lhs_sensitivity

        # Convert report results back into per-sample list
        # (in production this reads from the saved JSON)
        from src.historical_rag_builder import HistoricalRAGBuilder
        builder = HistoricalRAGBuilder()
        corpus = builder.build_pre_t0_corpus(
            output_path=os.path.join(OUTPUT_DIR, "historical_pret0_corpus.json")
        )

        # Re-run sensitivity with N=50 samples
        output_path = os.path.join(OUTPUT_DIR, "time_aware_sensitivity",
                                   "mft_sensitivity_results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            results = run_lhs_sensitivity(
                n_samples=50,
                pipeline_results_path=None,
                pipeline_results=report.get("per_sample", []),
                output_path=output_path,
            )
        except (ValueError, KeyError) as e:
            print(f"  LHS sensitivity requires per-sample results; run pipeline first. ({e})")
            results = []
    else:
        print("  No pipeline results available; skipping sensitivity.")
        results = []

    out_path = os.path.join(OUTPUT_DIR, "step3_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump({"n_samples": len(results), "results": str(results)[:500]}, f, indent=2)
    print(f"Step 3 results -> {out_path}\n")
    return results


# ═════════════════════════════════════════════════════════════════════
# EXECUTION
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    steps = {"1": step1_historical, "2": step2_adversarial, "3": step3_sensitivity}
    run_all = "ALL" in sys.argv or len(sys.argv) == 1

    if run_all or "1" in sys.argv:
        r1 = step1_historical()
    if run_all or "2" in sys.argv:
        r2 = step2_adversarial()
    if run_all or "3" in sys.argv:
        r3 = step3_sensitivity()

    print("\n" + "=" * 60)
    print("  REPLICATION COMPLETE")
    print(f"  Results in: {OUTPUT_DIR}/")
    print("=" * 60)
