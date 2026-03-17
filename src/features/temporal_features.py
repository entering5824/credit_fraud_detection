from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_count_previous(times: np.ndarray, window_seconds: float) -> np.ndarray:
    """
    For each index i, count j < i such that times[i] - times[j] <= window_seconds.
    Assumes `times` is sorted ascending.
    """
    out = np.zeros(times.shape[0], dtype=np.int32)
    start = 0
    for i in range(times.shape[0]):
        t = times[i]
        while start < i and (t - times[start]) > window_seconds:
            start += 1
        out[i] = i - start
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features that commonly show up in production fraud systems.

    Notes:
    - The Kaggle `creditcard.csv` uses `Time` as seconds since the first transaction.
    - We treat it as a monotonically increasing timestamp within the dataset.
    """
    if "Time" not in df.columns:
        return df.copy()

    out = df.copy()
    out["Time"] = out["Time"].astype(float)

    # Global cyclical-ish context (cheap, works even without real calendar time)
    seconds_in_day = 24 * 60 * 60
    out["time_of_day_sin"] = np.sin(2 * np.pi * (out["Time"] % seconds_in_day) / seconds_in_day)
    out["time_of_day_cos"] = np.cos(2 * np.pi * (out["Time"] % seconds_in_day) / seconds_in_day)

    # If synthetic_user_id exists, compute per-user deltas and windowed counts.
    if "synthetic_user_id" in out.columns:
        out = out.sort_values("Time").reset_index(drop=True)
        grp = out.groupby("synthetic_user_id", sort=False)
        prev_time = grp["Time"].shift(1)
        out["time_since_last_transaction"] = (out["Time"] - prev_time).fillna(seconds_in_day)

        # True rolling window counts (previous transactions in window).
        tx_1h = np.zeros(len(out), dtype=np.int32)
        tx_24h = np.zeros(len(out), dtype=np.int32)
        for _, idx in grp.indices.items():
            idx_arr = np.asarray(idx, dtype=np.int64)
            t = out.loc[idx_arr, "Time"].to_numpy(dtype=float)
            tx_1h[idx_arr] = _rolling_count_previous(t, 3600.0)
            tx_24h[idx_arr] = _rolling_count_previous(t, float(seconds_in_day))
        out["transactions_last_1h"] = tx_1h
        out["transactions_last_24h"] = tx_24h

        # Velocity features derived from counts (transactions per hour/day in history window)
        out["transaction_velocity_1h"] = out["transactions_last_1h"].astype(float)
        out["transaction_velocity_24h"] = out["transactions_last_24h"].astype(float) / 24.0
    else:
        out["time_since_last_transaction"] = seconds_in_day
        out["transactions_last_1h"] = 0
        out["transactions_last_24h"] = 0
        out["transaction_velocity_1h"] = 0.0
        out["transaction_velocity_24h"] = 0.0

    return out

