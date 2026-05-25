"""Synthetic L2 order book data with flash-crash dynamics.

Generates tick-by-tick trade data and depth snapshots that model
realistic market microstructure during a flash crash:

- Normal regime: tight spread, deep book
- Crash onset: bid depth evaporates exponentially, spread widens
- Trough: minimum liquidity, maximum spread
- Recovery: depth rebuilds, spread normalizes

Output formats:
  - standalone: dicts/lists for MicrostructureSimulator
  - hftbacktest: structured numpy arrays for HashMapMarketDepthBacktest
"""

import numpy as np
from dataclasses import dataclass, field


# hftbacktest event type constants
DEPTH_CLEAR_EVENT = 3
DEPTH_EVENT = 4
DEPTH_SNAPSHOT_EVENT = 5
TRADE_EVENT = 6
DEPTH_BBO_EVENT = 9

# Side constants
BUY = 1
SELL = -1


@dataclass
class FlashCrashL2Config:
    """Parameters governing the flash crash L2 dynamics."""
    base_price: float = 190.0
    trough_price: float = 150.0
    drop_duration_ms: float = 3000.0
    recovery_duration_ms: float = 7000.0

    # Spread parameters
    normal_spread_bps: float = 0.5          # 0.5 bps = 0.005%
    crash_max_spread_bps: float = 50.0       # 50 bps max during trough
    spread_widening_ms: float = 100.0        # time constant for spread widening

    # Depth parameters
    normal_bid_depth: int = 1000             # shares at best bid (normal)
    normal_ask_depth: int = 1000             # shares at best ask (normal)
    min_bid_depth: int = 10                  # shares at worst (trough)
    depth_decay_ms: float = 200.0            # time constant for depth decay
    depth_levels: int = 10                   # number of price levels per side
    price_tick: float = 0.01

    # Fill probability during crash
    min_fill_prob: float = 0.05             # worst-case fill probability at trough
    fill_decay_ms: float = 150.0             # time constant for fill prob decay

    def price_at(self, t_ms):
        """Mid price at time t during flash crash."""
        if t_ms is None or t_ms < 0:
            return self.base_price
        drop_magnitude = self.base_price - self.trough_price
        if t_ms <= self.drop_duration_ms:
            return self.base_price - (t_ms / self.drop_duration_ms) * drop_magnitude
        elif t_ms <= self.drop_duration_ms + self.recovery_duration_ms:
            recovery_t = t_ms - self.drop_duration_ms
            return self.trough_price + (recovery_t / self.recovery_duration_ms) * drop_magnitude
        return self.base_price

    def spread_bps_at(self, t_ms):
        """Spread in basis points at time t. Widens during crash, narrows during recovery."""
        crash_magnitude = self.base_price - self.price_at(t_ms)
        max_crash = self.base_price - self.trough_price
        crash_frac = crash_magnitude / max_crash if max_crash > 0 else 0
        extra_spread = (self.crash_max_spread_bps - self.normal_spread_bps) * crash_frac
        return self.normal_spread_bps + extra_spread

    def bid_depth_at(self, t_ms):
        """Best bid depth (shares) at time t."""
        crash_magnitude = self.base_price - self.price_at(t_ms)
        max_crash = self.base_price - self.trough_price
        crash_frac = crash_magnitude / max_crash if max_crash > 0 else 0
        return int(self.normal_bid_depth +
                   (self.min_bid_depth - self.normal_bid_depth) * crash_frac)

    def fill_probability_at(self, t_ms):
        """Probability a limit order fills within reasonable time at time t."""
        crash_magnitude = self.base_price - self.price_at(t_ms)
        max_crash = self.base_price - self.trough_price
        crash_frac = crash_magnitude / max_crash if max_crash > 0 else 0
        return 1.0 - (1.0 - self.min_fill_prob) * crash_frac


def generate_depth_snapshot(t_ms, config=None):
    """Generate a single L2 depth snapshot at time t_ms.

    Returns dict with bids and asks at multiple price levels.
    """
    if config is None:
        config = FlashCrashL2Config()

    mid = config.price_at(t_ms)
    spread_bps = config.spread_bps_at(t_ms)
    half_spread = mid * spread_bps / 10000.0
    half_spread = max(half_spread, config.price_tick)

    best_bid = mid - half_spread
    best_ask = mid + half_spread
    bid_depth = config.bid_depth_at(t_ms)
    ask_depth = config.normal_ask_depth  # ask depth stays roughly constant

    bids = []
    asks = []

    for level in range(config.depth_levels):
        depth_factor = max(0.1, 1.0 - level * 0.15)
        bids.append({
            "price": round(best_bid - level * config.price_tick, 2),
            "qty": int(bid_depth * depth_factor * (0.5 + 0.5 * np.random.random())),
        })
        asks.append({
            "price": round(best_ask + level * config.price_tick, 2),
            "qty": int(ask_depth * depth_factor * (0.5 + 0.5 * np.random.random())),
        })

    return {
        "mid": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "spread_bps": spread_bps,
        "bids": bids,
        "asks": asks,
    }


def generate_flash_crash_sequence(config=None, dt_ms=50):
    """Generate full L2 + trade sequence for a flash crash.

    Args:
        config: FlashCrashL2Config
        dt_ms: time step between snapshots (ms)

    Returns:
        list of dicts, each a depth snapshot with timestamp
    """
    if config is None:
        config = FlashCrashL2Config()

    total_duration = config.drop_duration_ms + config.recovery_duration_ms
    n_steps = int(total_duration / dt_ms) + 1
    timestamps = np.linspace(0, total_duration, n_steps)

    sequence = []
    for t in timestamps:
        snapshot = generate_depth_snapshot(t, config)
        snapshot["t_ms"] = t
        sequence.append(snapshot)

    return sequence


def sequence_to_hftbacktest_data(sequence, config=None, seed=42):
    """Convert a flash crash sequence to hftbacktest-compatible numpy arrays.

    Returns (depth_data, trade_data): two structured numpy arrays.
    Each event has: ev, exch_ts, local_ts, px, qty, order_id, ival, fval

    depth_data contains DEPTH_CLEAR_EVENT + DEPTH_EVENTs for each snapshot.
    trade_data contains TRADE_EVENTs for simulated market trades.
    """
    if config is None:
        config = FlashCrashL2Config()

    # hftbacktest structured dtype for depth
    depth_dtype = np.dtype([
        ('ev', '<u8'),
        ('exch_ts', '<i8'),
        ('local_ts', '<i8'),
        ('px', '<f8'),
        ('qty', '<f8'),
        ('order_id', '<u8'),
        ('ival', '<i8'),
        ('fval', '<f8'),
    ])

    trade_dtype = np.dtype([
        ('ev', '<u8'),
        ('exch_ts', '<i8'),
        ('local_ts', '<i8'),
        ('px', '<f8'),
        ('qty', '<f8'),
        ('order_id', '<u8'),
        ('ival', '<i8'),
        ('fval', '<f8'),
    ])

    depth_events = []
    trade_events = []
    order_id = 1000000
    rng = np.random.default_rng(seed)

    for snap in sequence:
        t_ns = int(snap["t_ms"] * 1_000_000)  # ms → ns

        # Clear previous depth levels, then re-populate
        depth_events.append((
            DEPTH_CLEAR_EVENT, t_ns, t_ns, 0.0, 0.0, 0, 0, 0.0
        ))

        # Bids (negative qty convention in some feeds; use ival = BUY)
        for i, level in enumerate(snap["bids"]):
            depth_events.append((
                DEPTH_EVENT, t_ns, t_ns,
                level["price"], float(level["qty"]),
                order_id + i, BUY, 0.0
            ))

        # Asks (ival = SELL)
        for i, level in enumerate(snap["asks"]):
            depth_events.append((
                DEPTH_EVENT, t_ns, t_ns,
                level["price"], float(level["qty"]),
                order_id + 100 + i, SELL, 0.0
            ))

        # Simulate occasional trades at mid
        if rng.random() < 0.3:
            trade_qty = rng.integers(10, 100)
            trade_px = snap["mid"] + rng.normal(0, 0.02)
            trade_side = BUY if rng.random() < 0.5 else SELL
            trade_events.append((
                TRADE_EVENT, t_ns, t_ns,
                round(trade_px, 2), float(trade_qty),
                order_id + 200, trade_side, 0.0
            ))

    depth_arr = np.array(depth_events, dtype=depth_dtype) if depth_events else np.empty(0, dtype=depth_dtype)
    trade_arr = np.array(trade_events, dtype=trade_dtype) if trade_events else np.empty(0, dtype=trade_dtype)

    return depth_arr, trade_arr


def compute_ideal_pnl(intervention_t_ms, position_size, base_price, trough_price,
                      drop_duration_ms, recovery_duration_ms):
    """Ideal P&L: instant fill at intervention price, no spread/friction."""
    if intervention_t_ms is None or intervention_t_ms < 0:
        return 0.0
    p_intervention = FlashCrashL2Config(
        base_price=base_price, trough_price=trough_price,
        drop_duration_ms=drop_duration_ms,
        recovery_duration_ms=recovery_duration_ms
    ).price_at(intervention_t_ms)
    return position_size * max(0.0, p_intervention - trough_price)


def compute_realized_pnl(intervention_t_ms, position_size, config=None, seed=42):
    """Realistic P&L: includes spread crossing, partial fills, and adverse selection.

    Models three microstructure costs:
    1. Spread crossing: must sell at bid (not mid) when reversing
    2. Partial fills: probability of fill decreases during crash trough
    3. Adverse selection: expected slippage from decision to fill

    Returns dict with: pnl, fill_rate, spread_cost, adverse_selection_cost, slippage_pct
    """
    if config is None:
        config = FlashCrashL2Config()
    if intervention_t_ms is None or intervention_t_ms < 0:
        return {"pnl": 0.0, "fill_rate": 0.0, "spread_cost": 0.0,
                "adverse_selection_cost": 0.0, "slippage_pct": 0.0}

    rng = np.random.default_rng(seed)

    mid = config.price_at(intervention_t_ms)
    spread_bps = config.spread_bps_at(intervention_t_ms)
    half_spread = mid * spread_bps / 10000.0
    best_bid = mid - half_spread

    # Ideal P&L based on mid-price fill
    ideal = position_size * max(0.0, mid - config.trough_price)

    # Fill probability during crash
    fill_prob = config.fill_probability_at(intervention_t_ms)

    # Simulate: how many shares actually fill at the bid?
    bid_depth_available = config.bid_depth_at(intervention_t_ms)
    filled_qty = min(position_size, int(bid_depth_available * fill_prob * rng.uniform(0.5, 1.0)))
    unfilled_qty = position_size - filled_qty

    fill_rate = filled_qty / position_size if position_size > 0 else 0.0

    # Spread cost: selling at bid instead of mid
    spread_cost = filled_qty * half_spread

    # Adverse selection: market moves against us between decision and fill
    adverse_move = rng.normal(0, half_spread * 0.5)
    adverse_cost = filled_qty * abs(adverse_move) if adverse_move < 0 else 0.0

    # Realized P&L: filled shares at bid, unfilled shares ride crash to trough
    realized = filled_qty * max(0.0, best_bid - config.trough_price)
    realized -= spread_cost
    realized -= adverse_cost

    # Unfilled shares ride to trough — same outcome as no-intervention
    # baseline, so they contribute $0 to P&L_saved (not negative).
    # No penalty term needed.

    slippage_pct = ((ideal - realized) / ideal * 100) if ideal > 0 else 0.0

    return {
        "pnl": round(realized, 2),
        "ideal_pnl": round(ideal, 2),
        "fill_rate": round(fill_rate, 4),
        "spread_cost": round(spread_cost, 2),
        "adverse_selection_cost": round(adverse_cost, 2),
        "slippage_pct": round(slippage_pct, 2),
        "filled_qty": filled_qty,
        "unfilled_qty": unfilled_qty,
    }
