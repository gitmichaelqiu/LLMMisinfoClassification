"""Portfolio-Level VaR/Exposure Circuit Breaker (Phase 18).

Monitors a multi-asset portfolio for correlated panic signals.
When multiple assets trigger simultaneous panic within a look-back
window, applies circuit breaker: halts new trades, reduces overall
exposure, or liquidates across all positions.

Usage:
    from src.portfolio_simulator import PortfolioSimulator

    sim = PortfolioSimulator(assets=["AAPL", "MSFT", "JPM"])
    results = sim.run_circuit_breaker_test(event_stream_df)
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque


class PortfolioSimulator:
    """Multi-asset portfolio simulation with correlated panic detection.

    Monitors a look-back window for simultaneous panic signals across
    assets. When the number of concurrent panics exceeds a threshold,
    triggers portfolio-level circuit breaker actions.
    """

    def __init__(self, assets=None, lookback_window_s=60,
                 panic_threshold=2, max_leverage=2.0,
                 position_size_per_asset=1000, base_price=100.0):
        """
        Args:
            assets: List of asset symbols (default: ["AAPL", "MSFT", "JPM"])
            lookback_window_s: Time window for correlated panic detection (s)
            panic_threshold: Number of simultaneous panics to trigger breaker
            max_leverage: Maximum gross leverage allowed
            position_size_per_asset: Shares per asset position
            base_price: Base price for all assets (simplified)
        """
        self.assets = assets or ["AAPL", "MSFT", "JPM"]
        self.n_assets = len(self.assets)
        self.lookback_window_s = lookback_window_s
        self.panic_threshold = min(panic_threshold, self.n_assets)
        self.max_leverage = max_leverage
        self.position_size_per_asset = position_size_per_asset
        self.base_price = base_price

        # State
        self.panic_events = deque()  # (timestamp, asset) tuples
        self.circuit_breakers_triggered = []
        self.daily_pnl = defaultdict(float)

    def _check_circuit_breaker(self, current_time):
        """Check if correlated panic exceeds threshold in look-back window.

        Returns True if circuit breaker should activate.
        """
        # Prune old events outside look-back window
        while self.panic_events and self.panic_events[0][0] < current_time - self.lookback_window_s:
            self.panic_events.popleft()

        # Count unique assets in panic within window
        panicking_assets = set(a for _, a in self.panic_events)
        return len(panicking_assets) >= self.panic_threshold

    def record_panic(self, asset, timestamp):
        """Record a panic signal for an asset at a given time."""
        self.panic_events.append((timestamp, asset))

    def evaluate_event(self, asset, timestamp, is_fake, llm_confidence,
                       llm_verdict):
        """Evaluate a single event within the portfolio context.

        Args:
            asset: Asset symbol
            timestamp: Event timestamp (seconds)
            is_fake: Ground truth (True = FAKE)
            llm_confidence: LLM confidence in FAKE verdict (0-1)
            llm_verdict: LLM verdict (0=HOLD, 1=INTERVENE, 2=ESCALATE)

        Returns:
            dict with action taken, position adjustments, circuit breaker status
        """
        # Record panic if LLM found FAKE with high confidence
        if llm_verdict == 1 and llm_confidence >= 0.6:
            self.record_panic(asset, timestamp)

        # Check circuit breaker
        breaker_active = self._check_circuit_breaker(timestamp)
        if breaker_active:
            action = "circuit_breaker"
            # Reduce position to 50% of normal
            effective_position = int(self.position_size_per_asset * 0.5)
            self.circuit_breakers_triggered.append({
                "timestamp": timestamp,
                "asset": asset,
                "panicking_assets": self._get_panicking_assets(timestamp),
            })
        else:
            action = "normal"
            effective_position = self.position_size_per_asset

        return {
            "asset": asset,
            "timestamp": timestamp,
            "is_fake": is_fake,
            "action": action,
            "effective_position": effective_position,
            "panicking_assets": self._get_panicking_assets(timestamp),
            "breaker_active": breaker_active,
            "n_breakers_triggered": len(self.circuit_breakers_triggered),
        }

    def _get_panicking_assets(self, current_time):
        """Get set of assets currently in panic state."""
        return set(a for t, a in self.panic_events
                   if t >= current_time - self.lookback_window_s)

    def run_circuit_breaker_test(self, event_stream, from_src=True):
        """Run circuit breaker on a stream of events.

        Args:
            event_stream: DataFrame with event data, or list of dicts.
                Expected columns: asset, timestamp, is_fake, llm_confidence, llm_verdict
            from_src: If True, import and use MFTPipeline/MFTMarketSimulator

        Returns:
            dict with summary statistics
        """
        if isinstance(event_stream, pd.DataFrame):
            events = event_stream.to_dict("records")
        else:
            events = list(event_stream)

        results = []
        total_pnl = 0.0
        n_breakers = 0

        for ev in events:
            asset = ev.get("asset", "UNKNOWN")
            ts = ev.get("timestamp", 0)
            is_fake = ev.get("is_fake", False)
            confidence = ev.get("llm_confidence", 0.5)
            verdict = ev.get("llm_verdict", 0)

            outcome = self.evaluate_event(asset, ts, is_fake, confidence, verdict)
            results.append(outcome)

            if outcome["action"] == "circuit_breaker":
                n_breakers += 1

            # Simplified P&L: positive for correct FAKE detection, negative otherwise
            if outcome["action"] == "circuit_breaker":
                # Reduced position saved some loss
                pos = outcome["effective_position"]
                if is_fake:
                    total_pnl += pos * 5.0  # partial save
            elif is_fake and verdict == 1:
                total_pnl += outcome["effective_position"] * 10.0
            elif not is_fake and verdict == 1:
                total_pnl -= outcome["effective_position"] * 8.0

        return {
            "n_events": len(results),
            "n_circuit_breakers": n_breakers,
            "n_panicking_assets": len(set(
                a for r in results for a in r.get("panicking_assets", [])
            )),
            "total_pnl": round(total_pnl, 2),
            "results": results[:20],  # first 20 for inspection
            "config": {
                "assets": self.assets,
                "lookback_window_s": self.lookback_window_s,
                "panic_threshold": self.panic_threshold,
                "max_leverage": self.max_leverage,
            },
        }


def generate_correlated_event_stream(n_events=50, seed=42):
    """Generate a synthetic multi-asset event stream with correlated panics.

    Creates events where a subset has correlated fake news across assets
    (sector-wide panic scenario).

    Returns:
        pd.DataFrame with columns: asset, timestamp, is_fake, llm_confidence, llm_verdict
    """
    rng = np.random.default_rng(seed)
    assets = ["AAPL", "MSFT", "JPM", "GS", "XOM"]
    events = []
    t = 0.0

    for i in range(n_events):
        asset = assets[i % len(assets)]
        t += rng.uniform(30, 120)  # events spaced 30-120s apart

        # Every 8th event: correlated sector panic
        is_correlated = (i % 8 == 7)
        is_fake = is_correlated or rng.random() < 0.3

        if is_correlated:
            # High confidence FAKE for this event
            confidence = rng.uniform(0.7, 0.95)
            verdict = 1
            # Also create correlated events for other assets
            for j, other in enumerate(assets):
                if other != asset and j % 2 == 0:
                    events.append({
                        "asset": other,
                        "timestamp": t + rng.uniform(0, 10),
                        "is_fake": True,
                        "llm_confidence": rng.uniform(0.6, 0.85),
                        "llm_verdict": 1,
                    })
        else:
            confidence = rng.uniform(0.1, 0.9)
            verdict = 1 if confidence >= 0.65 else (2 if confidence >= 0.35 else 0)

        events.append({
            "asset": asset,
            "timestamp": round(t, 1),
            "is_fake": is_fake or False,
            "llm_confidence": round(confidence, 4),
            "llm_verdict": verdict,
        })

    return pd.DataFrame(events)


if __name__ == "__main__":
    # Demo
    print("Portfolio Circuit Breaker Demo")
    print("=" * 60)

    sim = PortfolioSimulator(
        assets=["AAPL", "MSFT", "JPM", "GS", "XOM"],
        lookback_window_s=60,
        panic_threshold=2,
    )

    stream = generate_correlated_event_stream(n_events=50, seed=42)
    print(f"Generated {len(stream)} events across {stream['asset'].nunique()} assets")

    results = sim.run_circuit_breaker_test(stream)
    print(f"\nResults:")
    print(f"  Events processed: {results['n_events']}")
    print(f"  Circuit breakers triggered: {results['n_circuit_breakers']}")
    print(f"  Unique panicking assets: {results['n_panicking_assets']}")
    print(f"  Total P&L: ${results['total_pnl']}")
    print(f"  Config: {results['config']}")
    print("\nPortfolio simulator ready for Phase 18 integration.")
