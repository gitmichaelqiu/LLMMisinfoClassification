"""Finance case study orchestration script.

Runs the full verification pipeline on the finance dataset through
a specified verifier and reports results.

Usage:
    python -m src.finance.finance_case_study --verifier single-shot --test-size 50
    python -m src.finance.finance_case_study --verifier voting --n-voters 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from src.base_rate import ppv_curve
from src.datasets import load_dataset
from src.finance.finance_metrics import FinanceMetrics
from src.logging_utils import ExperimentLogger
from src.metrics import classification_metrics
from src.schemas import Verdict, VerifierConfig


def run_case_study(
    verifier_type: str = "single-shot",
    test_size: int = 50,
    model: str = "gpt-4o-mini",
    mock: bool = True,
    n_voters: int = 5,
    output_dir: str = "results",
):
    """Run the finance case study end-to-end.

    Args:
        verifier_type: "single-shot", "voting", "moa", or "rag".
        test_size: Number of test items.
        model: Model identifier.
        mock: If True, use MockClient (no API key needed).
        n_voters: Number of voters (voting verifier only).
        output_dir: Output directory for results.
    """
    # 1. Load data
    print("Loading finance dataset...")
    adapter = load_dataset("finance")
    train, test = adapter.train_test_split(test_size=test_size)
    counts = adapter.item_counts()
    print(f"Dataset: {counts['total']} items ({counts['real']} real, {counts['fake']} fake)")

    subset = test[:test_size]
    print(f"Test set: {len(subset)} items")

    # 2. Create verifier
    config = VerifierConfig(model=model, n_voters=n_voters)
    print(f"Creating verifier: {verifier_type} (mock={mock})")

    if verifier_type == "single-shot":
        from src.verifier_single_shot import SingleShotVerifier
        from src.llm_clients import create_client
        client = create_client(mock=mock)
        verifier = SingleShotVerifier(config=config, client=client)
    elif verifier_type == "voting":
        from src.verifier_voting import VotingVerifier
        from src.llm_clients import create_client
        client = create_client(mock=mock)
        verifier = VotingVerifier(config=config, client=client, n_voters=n_voters)
    elif verifier_type == "moa":
        from src.verifier_moa import MoAVerifier
        from src.llm_clients import create_client
        client = create_client(mock=mock)
        verifier = MoAVerifier(config=config, client=client)
    elif verifier_type == "rag":
        from src.verifier_rag import RAGVerifier
        from src.llm_clients import create_client
        client = create_client(mock=mock)
        verifier = RAGVerifier(config=config, client=client)
    else:
        raise ValueError(f"Unknown verifier: {verifier_type}")

    # 3. Run verification
    print("Running verification...")
    results = verifier.verify_batch(subset)

    # 4. Compute metrics
    ground_truths = [item.ground_truth or Verdict.REAL for item in subset]
    metrics = classification_metrics(results, ground_truths)
    print(f"\n=== Results ===")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"F1:        {metrics.f1:.4f}")
    print(f"FPR:       {metrics.fpr:.4f}")
    print(f"FNR:       {metrics.fnr:.4f}")
    print(f"Accuracy:  {metrics.accuracy:.4f}")
    print(f"Confusion: TP={metrics.confusion.tp} FP={metrics.confusion.fp} "
          f"TN={metrics.confusion.tn} FN={metrics.confusion.fn}")

    # 5. Compute finance-specific metrics
    fm = FinanceMetrics()
    pnl = fm.expected_pnl(metrics, prevalence=0.05)
    print(f"\nExpected P&L (at 5% base rate): ${pnl['expected_pnl']:,.0f}")

    # 6. PPV curve
    specificity = 1.0 - metrics.fpr
    ppv_points = ppv_curve(metrics.recall, specificity)
    print(f"\nPPV at key base rates:")
    for prev, ppv, npv in ppv_points:
        if prev in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50):
            print(f"  P(fake)={prev:.1%}: PPV={ppv:.4f}, NPV={npv:.4f}")

    # 7. Log results
    logger = ExperimentLogger(output_dir=os.path.join(output_dir, f"phase_{verifier_type}"))
    run_id = logger.log_run(
        config={
            "verifier_type": verifier_type,
            "model": model,
            "mock": mock,
            "test_size": test_size,
            "n_voters": n_voters,
        },
        metrics={
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fpr": metrics.fpr,
            "fnr": metrics.fnr,
            "accuracy": metrics.accuracy,
            "tp": metrics.confusion.tp,
            "fp": metrics.confusion.fp,
            "tn": metrics.confusion.tn,
            "fn": metrics.confusion.fn,
            "expected_pnl_5pct": pnl["expected_pnl"],
        },
        artifact_paths={},
        notes=f"Finance case study run with {verifier_type} verifier",
    )
    print(f"\nResults logged to: {logger.output_dir}/{run_id}.json")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finance case study")
    parser.add_argument("--verifier", default="single-shot", choices=["single-shot", "voting", "moa", "rag"])
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-mock", action="store_true", help="Use real API calls")
    parser.add_argument("--n-voters", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_case_study(
        verifier_type=args.verifier,
        test_size=args.test_size,
        model=args.model,
        mock=not args.no_mock,
        n_voters=args.n_voters,
        output_dir=args.output_dir,
    )
