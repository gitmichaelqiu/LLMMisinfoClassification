"""Reflexive MFT Market Simulator with sustained dislocation model.

Replaces the microsecond HFT flash crash model (l2_data_generator.py)
with a minute-scale sustained dislocation model for Mid-Frequency Trading.

Timescale: T0 = 0s (trade entry), T1 = 5s (LLM evaluation),
            T2 = 300s (human risk manager verification)

Key behaviors:
1. FAKE events: sharp panic drop, sustained dislocation, violent snapback at T2
2. REAL events: genuine sustained move in news direction, no snapback
3. Reflexivity: firm's own intervention at T1 moves price against it
4. Liquidity: bid depth evaporates during panic, recovers at T2

── Dynamic Sizing Multiplier Reconciliation ──────────────────────
The external review flags a gap between three "improvement factor" claims.
Here is the reconciliation:

1. Per-share execution cost (42.4x):
   Full reversal Q=1000 into V=100 at mid=$82 yields ~$50/share in total
   execution cost (slippage + reflexivity + adverse selection). With dynamic
   sizing (cap to 0.5 * V = 50 shares), cost drops to ~$1.18/share.
   Ratio: 50 / 1.18 ≈ 42.4x. This is the single-trade execution cost
   improvement, NOT the aggregate P&L improvement.

2. Square-root scaling (≈4.47x):
   Theoretical reduction from sqrt(Q_intervene / Q_full) = sqrt(50 / 1000) =
   sqrt(0.05) ≈ 0.224. The inverse (1/0.224) ≈ 4.47x represents the
   theoretical impact reduction from the square-root market impact formula
   ΔP ∝ √(Q/V). This only captures the reflexivity HALF of the cost, not
   the full slippage + reflexivity cost.

3. Aggregate P&L improvement (1.2x - 2.3x):
   Portfolio-level improvement across ALL trades. This is lower than the
   single-trade 42.4x because: (a) real events (no intervention) dilute
   the metric, (b) TP trades already had positive savings, so the
   improvement is on the delta rather than the base cost, (c) FP trades
   can become MORE costly with dynamic sizing (partial exposure drag).

The 124x number referenced in some materials appears to be 42.4x * 2.9
(an interaction factor), but is not consistently reproducible across
parameter regimes. Each metric (execution cost, sqrt-impact, aggregate
P&L) measures a different level of analysis and should be reported with
its methodological scope clearly documented.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Liquidity Profiles ──────────────────────────────────────────

LIQUIDITY_PROFILES = {
    "high_cap": {
        "normal_bid_depth": 20000,
        "min_bid_depth": 500,
        "panic_depth_decay_s": 2.5,
        "spread_normal_bps": 0.3,
        "spread_max_bps": 40.0,
        "description": "Mega-cap tech / large-cap financial (AAPL, MSFT, JPM)",
    },
    "mid_cap": {
        "normal_bid_depth": 5000,
        "min_bid_depth": 100,
        "panic_depth_decay_s": 1.5,
        "spread_normal_bps": 0.5,
        "spread_max_bps": 80.0,
        "description": "Mid-cap equities, moderate liquidity (default)",
    },
    "low_cap": {
        "normal_bid_depth": 500,
        "min_bid_depth": 20,
        "panic_depth_decay_s": 0.8,
        "spread_normal_bps": 2.0,
        "spread_max_bps": 200.0,
        "description": "Small-cap / micro-cap biotech (high volatility, thin books)",
    },
}


class MFTMarketSimulator:
    """Sustained dislocation market model for MFT verification arbitrage.

    Models the price curve, liquidity dynamics, and P&L for the
    T0 (trade) → T1 (LLM intervene) → T2 (human verify) timeline.
    """

    # MFT timeline constants (seconds)
    T0 = 0.0
    T1 = 5.0
    T2 = 300.0
    SNAP_DURATION = 10.0  # seconds over which T2 snapback occurs

    def __init__(self, base_price=100.0, panic_drop_pct=0.18,
                 sustained_dislocation_pct=0.30, real_appreciation_pct=0.08,
                 snapback_recovery_pct=0.97, permanent_impact_pct=0.02,
                 normal_bid_depth=5000, min_bid_depth=100,
                 panic_depth_decay_s=1.5, spread_normal_bps=0.5,
                 spread_max_bps=80.0, position_size=1000,
                 liquidity_profile=None):
        """
        Args:
            base_price: Entry price at T0 ($)
            panic_drop_pct: Fractional drop at T1 due to panic (e.g., 0.18 = 18%)
            sustained_dislocation_pct: Max fractional drop by T2 (e.g., 0.30 = 30%)
            real_appreciation_pct: Max fractional move for real events (e.g., 0.08)
            snapback_recovery_pct: Fraction of base_price recovered after T2 (0-1)
            permanent_impact_pct: Permanent price impact from the event (0-1)
            normal_bid_depth: Normal bid liquidity (shares)
            min_bid_depth: Minimum bid depth during panic trough (shares)
            panic_depth_decay_s: Exponential decay constant for depth erosion (s)
            spread_normal_bps: Normal bid-ask spread (bps)
            spread_max_bps: Maximum spread during panic (bps)
            position_size: Default position size for P&L calculations (shares)
            liquidity_profile: str key into LIQUIDITY_PROFILES, or None for manual params.
                              Overrides depth/spread params when set.
        """
        self.base_price = base_price
        self.panic_drop_pct = panic_drop_pct
        self.sustained_dislocation_pct = sustained_dislocation_pct
        self.real_appreciation_pct = real_appreciation_pct
        self.snapback_recovery_pct = snapback_recovery_pct
        self.permanent_impact_pct = permanent_impact_pct
        self.position_size = position_size
        self.liquidity_profile = liquidity_profile

        # Apply liquidity profile if specified (overrides individual params)
        if liquidity_profile and liquidity_profile in LIQUIDITY_PROFILES:
            prof = LIQUIDITY_PROFILES[liquidity_profile]
            self.normal_bid_depth = prof["normal_bid_depth"]
            self.min_bid_depth = prof["min_bid_depth"]
            self.panic_depth_decay_s = prof["panic_depth_decay_s"]
            self.spread_normal_bps = prof["spread_normal_bps"]
            self.spread_max_bps = prof["spread_max_bps"]
        else:
            self.normal_bid_depth = normal_bid_depth
            self.min_bid_depth = min_bid_depth
            self.panic_depth_decay_s = panic_depth_decay_s
            self.spread_normal_bps = spread_normal_bps
            self.spread_max_bps = spread_max_bps

    # ── Price Model ──────────────────────────────────────────────

    def price_at(self, t, is_fake=True):
        """Mid-price at time t (seconds).

        For FAKE events (deepfake crash):
          T0: base_price, T1: sharp drop, T1-T2: sustained dislocation,
          T2: violent snapback

        For REAL events:
          T0: base_price, T0-T2: gradual appreciation (or decline),
          T2: sustained at new level (no snapback)
        """
        if is_fake:
            return self._fake_price(t)
        return self._real_price(t)

    def _fake_price(self, t):
        """Price during a fake news event — deepfake crash pattern."""
        b = self.base_price
        panic_price = b * (1.0 - self.panic_drop_pct)
        trough_price = b * (1.0 - self.sustained_dislocation_pct)

        if t <= self.T1:
            # Phase 1: Exponential panic drop (0s → 5s)
            tau = 2.0  # seconds — fast decay
            fraction = 1.0 - np.exp(-t / tau) if t >= 0 else 0.0
            # Normalize so at t=T1, fraction ≈ 1.0
            norm = 1.0 - np.exp(-self.T1 / tau)
            frac = min(1.0, fraction / norm) if norm > 0 else 1.0
            return b - (b - panic_price) * frac

        elif t <= self.T2:
            # Phase 2: Sustained dislocation (5s → 300s)
            # Slow drift from panic_price down to trough_price
            drift_frac = (t - self.T1) / (self.T2 - self.T1)
            return panic_price - (panic_price - trough_price) * drift_frac

        else:
            # Phase 3: Snapback at T2 (300s → 310s+)
            snap_start = self.T2
            snap_end = self.T2 + self.SNAP_DURATION
            recovery_price = b * self.snapback_recovery_pct

            if t <= snap_end:
                snap_frac = (t - snap_start) / self.SNAP_DURATION
                # Sigmoid-like snapback
                snap_factor = snap_frac ** 2 / (snap_frac ** 2 + (1 - snap_frac) ** 2)
                return trough_price + (recovery_price - trough_price) * snap_factor
            else:
                return recovery_price

    def _real_price(self, t):
        """Price during a real news event — genuine sustained move, no snapback."""
        b = self.base_price
        peak_price = b * (1.0 + self.real_appreciation_pct)

        if t <= self.T1:
            # Phase 1: Gradual initial reaction
            tau = 3.0
            fraction = 1.0 - np.exp(-t / tau) if t >= 0 else 0.0
            norm = 1.0 - np.exp(-self.T1 / tau)
            frac = min(1.0, fraction / norm) if norm > 0 else 1.0
            t1_price = b + (peak_price - b) * 0.15  # ~15% of total move by T1
            return b + (t1_price - b) * frac

        elif t <= self.T2:
            # Phase 2: Continued appreciation to T2
            drift_frac = (t - self.T1) / (self.T2 - self.T1)
            t1_price = b + (peak_price - b) * 0.15
            return t1_price + (peak_price - t1_price) * drift_frac

        else:
            # Phase 3: Sustained at peak (no snapback)
            return peak_price

    # ── Liquidity Model ──────────────────────────────────────────

    def bid_depth_at(self, t, is_fake=True):
        """Best-bid depth (shares) at time t.

        For fake events: depth evaporates during panic, stays thin
        during dislocation, recovers during snapback.
        For real events: depth stays near normal throughout.
        """
        if not is_fake:
            return self.normal_bid_depth

        if t <= self.T1:
            # Exponential decay of depth during panic drop
            tau_d = self.panic_depth_decay_s
            frac = 1.0 - np.exp(-t / tau_d) if t >= 0 else 0.0
            norm = 1.0 - np.exp(-self.T1 / tau_d)
            decay_frac = min(1.0, frac / norm) if norm > 0 else 1.0
            return int(self.normal_bid_depth - (self.normal_bid_depth - self.min_bid_depth) * decay_frac)

        elif t <= self.T2:
            # Depth stays at minimum during sustained dislocation
            return self.min_bid_depth

        else:
            # Depth recovers during snapback
            snap_start = self.T2
            snap_end = self.T2 + self.SNAP_DURATION
            if t <= snap_end:
                snap_frac = (t - snap_start) / self.SNAP_DURATION
                return int(self.min_bid_depth + (self.normal_bid_depth - self.min_bid_depth) * snap_frac)
            return self.normal_bid_depth

    def spread_at(self, t, is_fake=True):
        """Bid-ask spread in basis points at time t."""
        if not is_fake:
            return self.spread_normal_bps

        if t <= self.T1:
            tau_s = 1.0
            frac = 1.0 - np.exp(-t / tau_s) if t >= 0 else 0.0
            norm = 1.0 - np.exp(-self.T1 / tau_s)
            spread_frac = min(1.0, frac / norm) if norm > 0 else 1.0
            spread_range = self.spread_max_bps - self.spread_normal_bps
            return self.spread_normal_bps + spread_range * spread_frac

        elif t <= self.T2:
            return self.spread_max_bps

        else:
            snap_start = self.T2
            snap_end = self.T2 + self.SNAP_DURATION
            if t <= snap_end:
                snap_frac = (t - snap_start) / self.SNAP_DURATION
                return self.spread_max_bps - (self.spread_max_bps - self.spread_normal_bps) * snap_frac
            return self.spread_normal_bps

    # ── P&L Calculations ─────────────────────────────────────────

    def compute_trade_pnl(self, entry_price, exit_price, position_size):
        """Simple P&L for a long trade: (exit - entry) * size."""
        return (exit_price - entry_price) * position_size

    def compute_hold_for_human_pnl(self, position_size=None, is_fake=True):
        """Hold-for-Human baseline P&L.

        Trade is entered at T0 (long). Position is held until T2.
        For fake events: apply max drawdown + extreme exit slippage.
        For real events: rider the appreciation to T2.

        Exit slippage model: at T2, during snapback (fake) or
        sustained move (real), the exit fills at a price worse than
        mid by a fraction of the spread.

        Returns dict with pnl, entry_price, exit_price, slippage_cost.
        """
        pos = position_size or self.position_size
        entry_price = self.base_price
        exit_time = self.T2 + 2.0  # exit happens just after T2

        mid_exit = self.price_at(exit_time, is_fake=is_fake)
        spread_bps = self.spread_at(exit_time, is_fake=is_fake)
        half_spread = mid_exit * spread_bps / 10000.0

        # Slippage: exit fills at bid (less favorable for long)
        if is_fake:
            # Extreme slippage during snapback — bid is far from mid
            slippage_multiple = 3.0
        else:
            slippage_multiple = 1.0

        slippage_cost = half_spread * slippage_multiple
        exit_price = mid_exit - slippage_cost

        pnl = self.compute_trade_pnl(entry_price, exit_price, pos)

        return {
            "pnl": round(pnl, 2),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "mid_exit": round(mid_exit, 2),
            "slippage_cost": round(slippage_cost, 2),
            "position_size": pos,
            "hold_duration_s": self.T2 - self.T0,
        }

    def compute_llm_intervene_pnl(self, position_size=None, is_fake=True,
                                   intervention_time=None):
        """LLM-Intervene P&L with execution reflexivity penalty.

        Trade is entered at T0 (long). LLM intervenes at intervention_time
        (default: T1 = 5s) to reverse the trade.

        Reflexivity penalty: the firm's reversal adds selling pressure in
        an illiquid market. The execution price slips by an amount
        proportional to position_size / bid_depth_at_T1.

        Returns dict with pnl, entry_price, exit_price,
        reflexivity_penalty, slippage_cost.
        """
        pos = position_size or self.position_size
        inv_time = intervention_time if intervention_time is not None else self.T1

        entry_price = self.base_price
        mid_exit = self.price_at(inv_time, is_fake=is_fake)
        spread_bps = self.spread_at(inv_time, is_fake=is_fake)
        half_spread = mid_exit * spread_bps / 10000.0

        # Baseline slippage: crossing the spread
        slippage_cost = half_spread

        # Reflexivity penalty: the firm's own order moves the market
        bid_depth = self.bid_depth_at(inv_time, is_fake=is_fake)
        if bid_depth > 0:
            # Penalty scales with how much of remaining liquidity we consume
            fill_ratio = pos / bid_depth
            reflexivity_penalty = half_spread * min(5.0, fill_ratio * 2.0)
        else:
            # No liquidity — extreme penalty and sentinel fill_ratio
            fill_ratio = float("inf")
            reflexivity_penalty = half_spread * 5.0  # extreme if no liquidity

        total_cost = slippage_cost + reflexivity_penalty
        exit_price = mid_exit - total_cost  # selling long position at discount

        pnl = self.compute_trade_pnl(entry_price, exit_price, pos)

        return {
            "pnl": round(pnl, 2),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "mid_exit": round(mid_exit, 2),
            "slippage_cost": round(slippage_cost, 2),
            "reflexivity_penalty": round(reflexivity_penalty, 2),
            "total_cost": round(total_cost, 2),
            "position_size": pos,
            "intervention_time_s": inv_time,
            "bid_depth_at_intervention": bid_depth,
            "fill_ratio": round(fill_ratio, 4),
        }

    def compute_pnl_saved(self, is_fake=True, intervention_time=None,
                           position_size=None):
        """P&L saved by LLM intervention vs holding to T2.

        Only meaningful for fake events: saved = hold_pnl - intervene_pnl.
        For real events: saved = 0 (LLM should NOT intervene).

        Returns dict with hold_pnl, intervene_pnl, pnl_saved.
        """
        pos = position_size or self.position_size

        hold = self.compute_hold_for_human_pnl(position_size=pos, is_fake=is_fake)
        intervene = self.compute_llm_intervene_pnl(
            position_size=pos, is_fake=is_fake, intervention_time=intervention_time
        )

        # P&L saved: intervention is better than holding
        # For fake events: intervene_pnl > hold_pnl (smaller loss) → positive saved
        # For real events: intervene_pnl < hold_pnl (missed gains) → negative saved
        pnl_saved = intervene["pnl"] - hold["pnl"]

        return {
            "hold_pnl": hold["pnl"],
            "intervene_pnl": intervene["pnl"],
            "pnl_saved": round(pnl_saved, 2),
            "is_fake_event": is_fake,
        }

    # ── Visualization ────────────────────────────────────────────

    def generate_price_curves(self, save_path="./plots/mft_price_curves.png"):
        """Generate and save price curve comparison plot.

        Shows price paths for both fake and real events, with
        intervention points and P&L regions annotated.
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # Time axis: 0 to 320s with fine resolution
        t_fine = np.linspace(0, 320, 2000)
        fake_prices = [self.price_at(t, is_fake=True) for t in t_fine]
        real_prices = [self.price_at(t, is_fake=False) for t in t_fine]
        bid_depths_fake = [self.bid_depth_at(t, is_fake=True) for t in t_fine]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top panel: Price curves
        ax1 = axes[0]
        ax1.plot(t_fine, fake_prices, "r-", linewidth=2, label="FAKE Event (Deepfake Crash)")
        ax1.plot(t_fine, real_prices, "g-", linewidth=2, label="REAL Event (Genuine Move)")

        # Annotate T0, T1, T2
        for t_label, label, color in [
            (self.T0, "T₀ (Trade)", "blue"),
            (self.T1, "T₁ (LLM Eval)", "orange"),
            (self.T2, "T₂ (Human Verify)", "purple"),
        ]:
            ax1.axvline(x=t_label, color=color, linestyle="--", alpha=0.5)
            y_pos = self.base_price * 1.12
            ax1.text(t_label, y_pos, label, ha="center", fontsize=10,
                     bbox=dict(facecolor=color, alpha=0.1))

        # Annotate P&L saved zone (for fake event)
        t1_idx = np.searchsorted(t_fine, self.T1)
        fake_t1_price = fake_prices[t1_idx]
        fake_t2_price = self.price_at(self.T2, is_fake=True)

        ax1.fill_between(
            [self.T1, self.T2],
            [fake_t1_price, fake_t2_price],
            [fake_t1_price, fake_t1_price],
            alpha=0.15, color="green",
            label="P&L Saved by Intervention"
        )

        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.set_title("MFT Price Curves: FAKE vs REAL Events", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Bottom panel: Bid depth during fake event
        ax2 = axes[1]
        ax2.plot(t_fine, bid_depths_fake, "b-", linewidth=2, label="Bid Depth (FAKE)")
        ax2.axvline(x=self.T1, color="orange", linestyle="--", alpha=0.5, label="T₁ (LLM)")
        ax2.axvline(x=self.T2, color="purple", linestyle="--", alpha=0.5, label="T₂ (Verify)")
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Bid Depth (shares)", fontsize=12)
        ax2.set_title("Liquidity Dynamics During Fake News Event", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[MFTSim] Price curves saved to {save_path}")


def demo():
    """Run a demo of the MFT market simulator across all liquidity profiles."""
    for prof_name in LIQUIDITY_PROFILES:
        print(f"\n{'='*60}")
        print(f"  LIQUIDITY PROFILE: {prof_name}")
        print(f"  {LIQUIDITY_PROFILES[prof_name]['description']}")
        print(f"{'='*60}")
        sim = MFTMarketSimulator(base_price=100.0, liquidity_profile=prof_name)
        sim.generate_price_curves(save_path=f"./plots/mft_price_curves_{prof_name}.png")

        print("\n=== HOLD-FOR-HUMAN P&L ===")
        for is_fake in [True, False]:
            result = sim.compute_hold_for_human_pnl(is_fake=is_fake)
            label = "FAKE" if is_fake else "REAL"
            print(f"  {label}: P&L=${result['pnl']:.2f}  "
                  f"Entry=${result['entry_price']}  Exit=${result['exit_price']}  "
                  f"Slippage=${result['slippage_cost']:.2f}")

        print("\n=== LLM-INTERVENE P&L (at T₁=5s) ===")
        for is_fake in [True, False]:
            result = sim.compute_llm_intervene_pnl(is_fake=is_fake)
            label = "FAKE" if is_fake else "REAL"
            print(f"  {label}: P&L=${result['pnl']:.2f}  "
                  f"Exit=${result['exit_price']:.2f}  "
                  f"ReflexPen=${result['reflexivity_penalty']:.2f}  "
                  f"FillRatio={result['fill_ratio']:.3f}")

        print("\n=== P&L SAVED (Intervene vs Hold) ===")
        for is_fake in [True, False]:
            result = sim.compute_pnl_saved(is_fake=is_fake)
            label = "FAKE" if is_fake else "REAL"
            print(f"  {label}: Hold=${result['hold_pnl']:.2f}  "
                  f"Intervene=${result['intervene_pnl']:.2f}  "
                  f"Saved=${result['pnl_saved']:.2f}")


if __name__ == "__main__":
    demo()
