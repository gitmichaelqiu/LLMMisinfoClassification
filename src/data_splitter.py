"""Strict temporal data splitting for the MFT pipeline.

Ensures train/test splits isolate events completely by time so the LLM
meta-classifier cannot learn future social media patterns from the
training set. This prevents look-ahead bias.

Strategy:
1. Each event is assigned a simulated T0 timestamp (sequential from
   a reference epoch, maintaining the event_id ordering).
2. Events are sorted by T0 timestamp.
3. A temporal cutoff is chosen: all events before the cutoff go to
   train, all events after go to test.
4. The split indices (event_ids) are saved to output/temporal_split_indices.json.

Usage:
    from src.data_splitter import temporal_train_test_split
    train_ids, test_ids, metadata = temporal_train_test_split()

Output: output/temporal_split_indices.json
"""

import os
import json
import numpy as np
import pandas as pd


def _assign_t0_timestamps(events_df, reference_date="2026-01-05 09:30:00",
                          interval_minutes=15):
    """Assign simulated T0 timestamps to events based on their sequential order.

    Events arrive sequentially in the market. We assign timestamps in
    chronological order, spaced `interval_minutes` apart.

    Args:
        events_df: DataFrame with event_id column
        reference_date: Base date for first event (default: market open)
        interval_minutes: Minutes between consecutive events

    Returns:
        Series of datetime timestamps
    """
    ref = pd.Timestamp(reference_date)
    n = len(events_df)
    minutes_offset = np.arange(n) * interval_minutes
    timestamps = pd.Series([ref + pd.Timedelta(minutes=int(m)) for m in minutes_offset])
    return timestamps


def temporal_train_test_split(temporal_events_path="./output/temporal_events.csv",
                              output_path="./output/temporal_split_indices.json",
                              test_ratio=0.2,
                              reference_date="2026-01-05 09:30:00",
                              interval_minutes=15,
                              seed=42):
    """Perform a strict temporal train/test split on temporal events.

    Args:
        temporal_events_path: Path to temporal_events.csv
        output_path: Output JSON path for split indices
        test_ratio: Fraction of events to hold out for testing (default: 0.2)
        reference_date: Reference date for T0 assignment
        interval_minutes: Minutes between consecutive events
        seed: Random seed for within-class shuffling of event ordering

    Returns:
        (train_ids, test_ids, metadata) tuple
    """
    rng = np.random.default_rng(seed)
    events_df = pd.read_csv(temporal_events_path).copy()
    n_total = len(events_df)

    # Shuffle within each label class to randomize ordering while
    # maintaining class balance across the time series.
    real_df = events_df[events_df["T2_human_verdict"] == 0].sample(frac=1, random_state=rng)
    fake_df = events_df[events_df["T2_human_verdict"] == 1].sample(frac=1, random_state=rng)

    # Interleave shuffled class groups using proportional sampling.
    # Probability of picking from real = remaining_real / total_remaining,
    # ensuring fair class representation throughout the temporal sequence.
    interleaved_rows = []
    i = j = 0
    n_real, n_fake = len(real_df), len(fake_df)
    while i < n_real or j < n_fake:
        remaining_real = n_real - i
        remaining_fake = n_fake - j
        p_real = remaining_real / (remaining_real + remaining_fake)
        if i < n_real and (j >= n_fake or rng.random() < p_real):
            interleaved_rows.append(real_df.iloc[i])
            i += 1
        else:
            interleaved_rows.append(fake_df.iloc[j])
            j += 1

    ordered_df = pd.DataFrame(interleaved_rows).reset_index(drop=True)

    # Assign T0 timestamps based on this sequential order
    ordered_df["T0_timestamp"] = _assign_t0_timestamps(
        ordered_df, reference_date, interval_minutes
    )

    # Temporal split: first (1 - test_ratio) fraction of events → train
    cutoff_idx = int(len(ordered_df) * (1 - test_ratio))
    temporal_cutoff = ordered_df.iloc[cutoff_idx]["T0_timestamp"]

    train_df = ordered_df.iloc[:cutoff_idx]
    test_df = ordered_df.iloc[cutoff_idx:]

    train_ids = train_df["event_id"].tolist()
    test_ids = test_df["event_id"].tolist()

    # Verify no temporal leakage
    train_max_ts = train_df["T0_timestamp"].max()
    test_min_ts = test_df["T0_timestamp"].min()

    metadata = {
        "n_total": n_total,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_fake_ratio": float(train_df["T2_human_verdict"].mean()),
        "test_fake_ratio": float(test_df["T2_human_verdict"].mean()),
        "temporal_cutoff": str(temporal_cutoff),
        "train_max_t0": str(train_max_ts),
        "test_min_t0": str(test_min_ts),
        "no_temporal_leakage": train_max_ts < test_min_ts,
        "reference_date": reference_date,
        "interval_minutes": interval_minutes,
        "test_ratio": test_ratio,
        "seed": seed,
        "method": "strict_temporal_split",
    }

    split_data = {
        "train_indices": train_ids,
        "test_indices": test_ids,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"[DataSplitter] Temporal train/test split saved to {output_path}")
    print(f"  Total events: {n_total}")
    print(f"  Train: {len(train_ids)} events (fake_ratio={metadata['train_fake_ratio']:.3f})")
    print(f"  Test:  {len(test_ids)} events (fake_ratio={metadata['test_fake_ratio']:.3f})")
    print(f"  Temporal cutoff: {temporal_cutoff}")
    print(f"  No temporal leakage: {metadata['no_temporal_leakage']}")

    return train_ids, test_ids, metadata


if __name__ == "__main__":
    temporal_train_test_split()
