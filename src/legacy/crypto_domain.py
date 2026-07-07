"""Cryptocurrency Market Domain (Phase 20).

Extends MFTMarketSimulator with a crypto-specific market profile
for testing on stablecoin de-pegging, exchange insolvency, and
token flash hack scenarios.

Crypto profile characteristics:
  - Extreme volatility: Y=2.0, σ=0.80
  - Slow liquidity replenishment: decay_constant=1200s
  - Higher spreads and lower depth than even low-cap equities
"""

import json

import numpy as np
import pandas as pd
from src.mft_simulator import LIQUIDITY_PROFILES, MFTMarketSimulator

# Crypto-specific liquidity profile
LIQUIDITY_PROFILES["crypto"] = {
    "normal_bid_depth": 200,
    "min_bid_depth": 5,
    "panic_depth_decay_s": 1200.0,   # extremely slow replenishment
    "spread_normal_bps": 10.0,       # 10 bps even in normal conditions
    "spread_max_bps": 500.0,         # 500 bps during panic
    "impact_coefficient": 2.0,       # Y = 2.0 (vs 0.25-1.0 for equities)
    "volatility": 0.80,              # σ = 80% annualized
    "description": "Altcoin / shitcoin token — extreme volatility, thin books, slow recovery",
}

CRYPTO_HOAX_TEMPLATES = [
    # Stablecoin de-pegging
    {
        "headline": "USDT issuer Tether admits reserves backed by only 12% cash equivalents",
        "entity": "Tether",
        "type": "stablecoin_depeg",
        "is_fake": True,
        "expected_drop_pct": 0.15,
    },
    {
        "headline": "Circle confirms USDC reserve shortfall following Silicon Valley Bank collapse",
        "entity": "Circle",
        "type": "stablecoin_depeg",
        "is_fake": True,
        "expected_drop_pct": 0.12,
    },
    # Exchange insolvency
    {
        "headline": "Binance CEO Changpeng Zhao arrested in Nigeria, exchange freezes withdrawals",
        "entity": "Binance",
        "type": "exchange_insolvency",
        "is_fake": True,
        "expected_drop_pct": 0.25,
    },
    {
        "headline": "Coinbase discloses $3.8B hole in custody accounts, SEC investigation launched",
        "entity": "Coinbase",
        "type": "exchange_insolvency",
        "is_fake": True,
        "expected_drop_pct": 0.20,
    },
    # Token flash hack
    {
        "headline": "BREAKING: Cross-chain bridge exploited — $420M drained from Solana ecosystem",
        "entity": "Solana",
        "type": "flash_hack",
        "is_fake": True,
        "expected_drop_pct": 0.30,
    },
    {
        "headline": "Uniswap v4 oracle manipulated — $200M in LP funds stolen via flash loan attack",
        "entity": "Uniswap",
        "type": "flash_hack",
        "is_fake": True,
        "expected_drop_pct": 0.18,
    },
    # REAL crypto events (not fake — used for TN/FP testing)
    {
        "headline": "Bitcoin ETF approval by SEC drives institutional inflows, BTC breaks $100K",
        "entity": "Bitcoin",
        "type": "real_adoption",
        "is_fake": False,
    },
    {
        "headline": "Ethereum Dencun upgrade reduces L2 fees by 90%, ETH staking yield rises",
        "entity": "Ethereum",
        "type": "real_upgrade",
        "is_fake": False,
    },
    {
        "headline": "MicroStrategy adds 12,000 BTC to treasury, total holdings exceed 200K BTC",
        "entity": "MicroStrategy",
        "type": "real_adoption",
        "is_fake": False,
    },
]


def run_crypto_stress_test(n_events=50, seed=42, position_size=500):
    """Run the MFT pipeline on crypto-domain hoaxes with extreme volatility.

    Args:
        n_events: Number of test events
        seed: Random seed
        position_size: Position size in tokens (smaller due to thin books)

    Returns:
        dict with results summary
    """
    rng = np.random.default_rng(seed)
    events = []

    for i in range(n_events):
        template = rng.choice(CRYPTO_HOAX_TEMPLATES)
        events.append({
            "event_id": f"CRYPTO-{i:05d}",
            "T0_headline": template["headline"],
            "T2_human_verdict": 1 if template.get("is_fake", True) else 0,
            "entity": template["entity"],
            "type": template.get("type", "unknown"),
        })

    events_df = pd.DataFrame(events)
    print(f"[Crypto] Created {len(events_df)} crypto-domain events")

    # Run backtest using the MFT pipeline with crypto profile
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from src.async_pipeline import MFTPipeline
    from src.rag_retriever import DualRAGRetriever

    # Train heuristic fallback from headline patterns
    texts = events_df["T0_headline"].fillna("")
    labels = events_df["T2_human_verdict"]
    vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000, random_state=42)
    X = vec.fit_transform(texts)
    clf.fit(X, labels)

    def crypto_heuristic(content):
        Xt = vec.transform([content])
        return int(clf.predict(Xt)[0])

    # Use crypto simulation profile
    sim = MFTMarketSimulator(
        liquidity_profile="crypto",
        base_price=50.0,        # token price
        position_size=position_size,
        panic_drop_pct=0.15,
        sustained_dislocation_pct=0.30,
        enable_air_pockets=True,
    )

    dual_rag = DualRAGRetriever()
    dual_rag.ensure_index()

    pipeline = MFTPipeline(
        dual_rag=dual_rag,
        position_size=position_size,
        base_price=50.0,
        use_three_tier=True,
        heuristic_predictor=crypto_heuristic,
    )
    # Override simulator with crypto-configured one
    pipeline.simulator = sim

    results = pipeline.process_events(events_df)
    report = pipeline.generate_report()
    pipeline.cleanup()

    report["phase"] = "20_crypto_stress_test"
    report["simulator_config"] = {
        "impact_coefficient": sim.impact_coefficient,
        "volatility": sim.volatility,
        "panic_drop_pct": sim.panic_drop_pct,
        "liquidity_profile": "crypto",
    }

    results_json = "./output/phase_20_crypto_metrics.json"
    with open(results_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Crypto] Results saved to {results_json}")
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 20: CRYPTOCURRENCY STRESS TEST")
    print("=" * 60)

    results = run_crypto_stress_test(n_events=20, position_size=500)

    print("\nResults:")
    print(f"  Events: {results['n_events']}")
    print(f"  Accuracy: {results['accuracy_metrics']['accuracy']:.3f}")
    print(f"  Precision: {results['accuracy_metrics']['precision']:.3f}")
    print(f"  Recall: {results['accuracy_metrics']['recall']:.3f}")
    print(f"  Total P&L: ${results['pnl_metrics']['total_pnl']:,.2f}")
    print(f"  P&L Saved: ${results['pnl_metrics']['total_pnl_saved']:,.2f}")
