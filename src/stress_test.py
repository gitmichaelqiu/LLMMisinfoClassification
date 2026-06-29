"""Stress testing & adversarial evaluation for the MFT verification pipeline.

Evaluates the LLM's robustness against poisoned Social RAG indices
where coordinated bot accounts actively defend fake news, creating
conflicting consensus signals.

Method:
1. Generate adversarial social streams at varying bot intensity levels.
2. For each level, rebuild the Social RAG index and run the pipeline.
3. Measure Precision, Recall, P&L degradation at each intensity.
4. Generate adversarial degradation plot.

Reference: Phase 6 of CLAUDE.md
"""

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.social_stream_generator import generate_adversarial_stream
from src.rag_retriever import DualRAGRetriever, SocialStreamRetriever
from src.async_pipeline import MFTPipeline, run_mft_backtest
from src.data_splitter import temporal_train_test_split


def _train_heuristic_baseline(train_df):
    """Train a TF-IDF + Logistic Regression baseline from training data.

    Returns a callable predictor(content) -> int 0/1, or None if training fails.
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, random_state=42)
    texts = train_df.get("T0_headline", train_df.get("headline", train_df.get("content", "")))
    labels = train_df.get("T2_human_verdict", train_df.get("label", -1))
    if len(texts) < 10 or (labels == -1).all():
        return None
    X = vectorizer.fit_transform(texts.fillna(""))
    clf.fit(X, labels)
    print(f"[Heuristic Baseline] Trained on {len(train_df)} samples (acc={clf.score(X, labels):.2%})")

    def predictor(content):
        Xt = vectorizer.transform([content])
        return int(clf.predict(Xt)[0])
    return predictor


def run_stress_test_at_intensity(
    bot_pct,
    temporal_events_path="./output/temporal_events.csv",
    test_ratio=0.2,
    model="deepseek",
    position_size=1000,
    base_price=100.0,
    thinking="enabled",
    use_system0=False,
    use_moa=False,
    max_test_events=None,
    seed=42,
):
    """Run the MFT pipeline with an adversarially poisoned social stream.

    Args:
        bot_pct: Fraction of bot posts to inject (0.0 to 1.0)
        temporal_events_path: Path to temporal events CSV
        test_ratio: Fraction of events for test set
        model: LLM backend
        position_size: Shares per trade
        base_price: Base entry price
        thinking: DeepSeek thinking mode
        use_system0: Enable System 0 pre-filter
        use_moa: Enable MoA debate architecture (Phase 9)
        max_test_events: Cap test set size
        seed: Random seed

    Returns:
        dict with metrics for this intensity level
    """
    from dotenv import load_dotenv
    from openai import OpenAI
    from transformers import pipeline

    load_dotenv()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    bot_label = str(int(bot_pct * 100))
    mode = "MoA" if use_moa else "SingleShot"
    print(f"\n{'='*60}")
    print(f"  STRESS TEST: bot_pct={bot_pct:.0%} mode={mode}")
    print(f"{'='*60}")

    # Generate adversarial social stream at this intensity
    adv_stream_path = f"./output/social_stream_adversarial_bots{bot_label}.csv"
    generate_adversarial_stream(
        temporal_events_path=temporal_events_path,
        output_path=adv_stream_path,
        seed=seed,
        bot_pct=bot_pct,
    )

    # Load test events
    events_df = pd.read_csv(temporal_events_path)
    train_ids, test_ids, split_meta = temporal_train_test_split(
        temporal_events_path=temporal_events_path,
        test_ratio=test_ratio,
    )
    test_df = events_df[events_df["event_id"].isin(test_ids)]
    if max_test_events and len(test_df) > max_test_events:
        # Stratified sample to ensure FAKE events included
        fake_df = test_df[test_df["T2_human_verdict"] == 1]
        real_df = test_df[test_df["T2_human_verdict"] == 0]
        n_fake = min(len(fake_df), max(max_test_events // 2, 1))
        n_real = min(len(real_df), max_test_events - n_fake)
        test_df = pd.concat([fake_df.head(n_fake), real_df.head(n_real)], ignore_index=True)
        print(f"Test set: {len(test_df)} events")

    # Train heuristic baseline from training set for mock-mode fallback
    train_df = events_df[events_df["event_id"].isin(train_ids)]
    heuristic_predictor = _train_heuristic_baseline(train_df)

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

    # Initialize Dual RAG with adversarial social stream
    print(f"Initializing Dual RAG with adversarial social stream ({bot_pct:.0%} bots)...")
    dual_rag = DualRAGRetriever()

    # Override social retriever before ensure_index so it loads the poisoned stream
    if bot_pct > 0 and adv_stream_path and os.path.exists(adv_stream_path):
        dual_rag.social_retriever = SocialStreamRetriever(social_stream_path=adv_stream_path)

    dual_rag.ensure_index()
    print(f"  Social stream has {len(dual_rag.social_retriever._posts_df)} posts")

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
        use_moa=use_moa,
        heuristic_predictor=heuristic_predictor,
    )

    # Run test set
    print(f"\nProcessing {len(test_df)} test events...")
    results = pipeline.process_events(test_df)

    # Generate report and extract metrics
    report = pipeline.generate_report()
    pipeline.cleanup()

    return {
        "bot_pct": bot_pct,
        "n_events": report.get("n_events", 0),
        "accuracy": report.get("accuracy_metrics", {}).get("accuracy", 0),
        "precision": report.get("accuracy_metrics", {}).get("precision", 0),
        "recall": report.get("accuracy_metrics", {}).get("recall", 0),
        "total_pnl": report.get("pnl_metrics", {}).get("total_pnl", 0),
        "total_pnl_saved": report.get("pnl_metrics", {}).get("total_pnl_saved", 0),
        "tp": report.get("outcome_counts", {}).get("true_positive", 0),
        "fp": report.get("outcome_counts", {}).get("false_positive", 0),
        "tn": report.get("outcome_counts", {}).get("true_negative", 0),
        "fn": report.get("outcome_counts", {}).get("false_negative", 0),
        "intervened": report.get("verdict_distribution", {}).get("intervene", 0),
        "escalated": report.get("verdict_distribution", {}).get("escalate", 0),
    }


def run_full_stress_test(
    bot_levels=None,
    max_test_events=None,
    model="deepseek",
    use_moa=False,
    thinking="enabled",
    output_path="./output/adversarial_results.json",
):
    """Run stress test at all bot intensity levels and generate degradation plot.

    Args:
        bot_levels: List of bot percentages to test
        max_test_events: Cap test set size per run
        model: LLM backend
        use_moa: Enable MoA debate architecture (Phase 9)
        output_path: Output JSON path

    Returns:
        list of metrics dicts per intensity level
    """
    if bot_levels is None:
        bot_levels = [0.0, 0.25, 0.50, 0.75]

    mode_str = "MoA" if use_moa else "Single-Shot"
    print("=" * 60)
    print(f"STRESS TESTING & ADVERSARIAL EVALUATION ({mode_str})")
    print("=" * 60)

    all_metrics = []

    for bot_pct in bot_levels:
        metrics = run_stress_test_at_intensity(
            bot_pct=bot_pct,
            max_test_events=max_test_events,
            model=model,
            thinking=thinking,
            use_system0=False,
            use_moa=use_moa,
        )
        all_metrics.append(metrics)

        print(f"\n  RESULTS (bot_pct={bot_pct:.0%}):")
        print(f"    Accuracy={metrics['accuracy']:.3f}, "
              f"Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}")
        print(f"    Total P&L=${metrics['total_pnl']:,.2f}, "
              f"P&L Saved=${metrics['total_pnl_saved']:,.2f}")

    # Save metrics
    save_data = {
        "phase": "6_adversarial_stress_test",
        "timestamp": time.ctime(),
        "method": "bot_pct_intensity_sweep",
        "mode": "moa" if use_moa else "single_shot",
        "bot_levels_tested": bot_levels,
        "results": all_metrics,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nAdversarial results saved to {output_path}")

    # Generate degradation plot
    suffix = "_moa" if use_moa else ""
    _generate_degradation_plot(all_metrics, suffix=suffix)

    return all_metrics


def _generate_degradation_plot(all_metrics, suffix=""):
    """Generate adversarial degradation plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[StressTest] Matplotlib not available, skipping degradation plot.")
        return

    bot_levels = [m["bot_pct"] for m in all_metrics]
    precisions = [m["precision"] for m in all_metrics]
    recalls = [m["recall"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]
    pnl_saveds = [m["total_pnl_saved"] / 1000 for m in all_metrics]  # scale to $k

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Precision/Recall degradation
    ax1.plot(bot_levels, precisions, "bo-", linewidth=2, markersize=8, label="Precision")
    ax1.plot(bot_levels, recalls, "gs-", linewidth=2, markersize=8, label="Recall")
    ax1.plot(bot_levels, accuracies, "r^--", linewidth=2, markersize=8, label="Accuracy")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax1.set_xlabel("Bot Intensity (% of adversarial posts)")
    ax1.set_ylabel("Score")
    mode_tag = " (MoA)" if "_moa" in suffix else ""
    ax1.set_title(f"LLM Verifier Degradation Under Adversarial Attack{mode_tag}", fontsize=12, fontweight="bold")
    ax1.set_xticks(bot_levels)
    ax1.set_xticklabels([f"{int(b*100)}%" for b in bot_levels])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right panel: P&L impact
    bars = ax2.bar([f"{int(b*100)}%" for b in bot_levels],
            pnl_saveds,
            color=["green" if p >= 0 else "red" for p in pnl_saveds],
            alpha=0.7, edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.set_xlabel("Bot Intensity")
    ax2.set_ylabel("P&L Saved ($K)")
    ax2.set_title("Economic Impact of Social RAG Poisoning", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(pnl_saveds):
        label_y = v + 1.5 if v >= 0 else v - 3.0
        ax2.text(i, label_y, f"${v:.1f}k",
                 ha="center", fontsize=10, fontweight="bold",
                 color="green" if v >= 0 else "red")

    plot_path = f"./plots/adversarial_degradation{suffix}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[StressTest] Adversarial degradation plot saved to {plot_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MFT Adversarial Stress Test")
    parser.add_argument("--model", default="deepseek")
    parser.add_argument("--test-size", type=int, default=None, help="Cap test set size")
    parser.add_argument("--output", default="./output/adversarial_results.json")
    parser.add_argument("--moa", action="store_true", help="Enable MoA debate architecture")
    parser.add_argument("--thinking", default="enabled", choices=["enabled", "disabled"],
                        help="DeepSeek thinking mode (default: enabled)")
    args = parser.parse_args()

    run_full_stress_test(
        max_test_events=args.test_size,
        model=args.model,
        use_moa=args.moa,
        thinking=args.thinking,
        output_path=args.output,
    )
