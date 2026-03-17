from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _stable_bucket(value: str, n_buckets: int) -> int:
    h = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    as_int = int.from_bytes(h, byteorder="big", signed=False)
    return int(as_int % n_buckets)


def _quantize(x: pd.Series, step: float) -> pd.Series:
    if step <= 0:
        raise ValueError("step must be > 0")
    # NaN-safe quantization: treat missing as 0 before bucketing
    x = x.fillna(0.0)
    return (x / step).round().astype("int64")


@dataclass(frozen=True)
class SyntheticIdentityConfig:
    seed: int = 42
    n_users: int = 5000
    n_merchants: int = 2000
    amount_step: float = 25.0
    v_step: float = 0.5


def add_synthetic_ids(
    df: pd.DataFrame, cfg: SyntheticIdentityConfig = SyntheticIdentityConfig()
) -> pd.DataFrame:
    """
    Deterministically simulate `synthetic_user_id` and `synthetic_merchant_id`.

    This enables behavior features on creditcard.csv without adding new raw columns.
    """
    out = df.copy()
    amount_q = _quantize(out["Amount"].astype(float), cfg.amount_step)

    # Use a few V features as a stable-ish signature; keep it cheap.
    v_cols = [c for c in ["V1", "V2", "V3", "V4", "V5"] if c in out.columns]
    if v_cols:
        v_sig = (
            _quantize(out[v_cols[0]].astype(float), cfg.v_step).astype(str)
            + "|"
            + _quantize(out[v_cols[1]].astype(float), cfg.v_step).astype(str)
        ) if len(v_cols) >= 2 else _quantize(out[v_cols[0]].astype(float), cfg.v_step).astype(str)
    else:
        v_sig = pd.Series(["0"] * len(out), index=out.index)

    base = (
        "u|"
        + str(cfg.seed)
        + "|"
        + amount_q.astype(str)
        + "|"
        + v_sig.astype(str)
    )
    merch_base = (
        "m|"
        + str(cfg.seed + 1337)
        + "|"
        + amount_q.astype(str)
        + "|"
        + v_sig.astype(str)
    )

    out["synthetic_user_id"] = base.map(lambda s: _stable_bucket(s, cfg.n_users)).astype(
        "int32"
    )
    out["synthetic_merchant_id"] = merch_base.map(
        lambda s: _stable_bucket(s, cfg.n_merchants)
    ).astype("int32")
    return out


def add_synthetic_behavior_features(
    df: pd.DataFrame,
    seed: int = 42,
    n_users: int = 5000,
    n_merchants: int = 2000,
) -> pd.DataFrame:
    """
    Add realistic fraud-style behavioral features based on synthetic IDs.
    Requires: `Time` and `Amount`.
    """
    out = add_synthetic_ids(
        df,
        cfg=SyntheticIdentityConfig(seed=seed, n_users=n_users, n_merchants=n_merchants),
    )

    if "Time" not in out.columns:
        return out

    out = out.sort_values("Time").reset_index(drop=True)
    out["Amount"] = out["Amount"].astype(float)

    # Rolling user spend stats (use expanding as a cheap proxy without time windows).
    grp = out.groupby("synthetic_user_id", sort=False)
    out["user_avg_amount"] = grp["Amount"].transform(lambda s: s.shift(1).expanding().mean())
    out["user_med_amount"] = grp["Amount"].transform(lambda s: s.shift(1).expanding().median())

    # Spending spike ratio: amount / rolling median (robust)
    denom = out["user_med_amount"].replace(0.0, np.nan)
    out["spending_spike_ratio"] = (out["Amount"] / denom).fillna(1.0)

    # Merchant frequency: how often user sees this merchant historically
    out["user_merchant_count"] = (
        out.groupby(["synthetic_user_id", "synthetic_merchant_id"], sort=False)
        .cumcount()
        .astype("int32")
    )

    # Merchant novelty: first time seeing merchant for that user
    out["is_new_merchant_for_user"] = (out["user_merchant_count"] == 0).astype("int8")

    # Distinct merchants seen so far by user (history-only)
    first_seen = out["is_new_merchant_for_user"].astype("int8")
    out["user_distinct_merchants_so_far"] = (
        out.groupby("synthetic_user_id", sort=False)[first_seen.name]
        .cumsum()
        .shift(1)
        .fillna(0)
        .astype("int32")
    )

    # Frequency-style score: high when merchant is frequently seen
    out["merchant_frequency_score"] = 1.0 / (1.0 + out["user_merchant_count"].astype(float))

    # Fill missing early-history stats with global priors
    global_avg = float(out["Amount"].mean())
    global_med = float(out["Amount"].median())
    out["user_avg_amount"] = out["user_avg_amount"].fillna(global_avg)
    out["user_med_amount"] = out["user_med_amount"].fillna(global_med)

    return out

