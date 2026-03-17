from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.behavior_features import add_synthetic_behavior_features
from src.features.temporal_features import add_temporal_features


@dataclass(frozen=True)
class FeatureConfig:
    add_temporal: bool = True
    add_behavior: bool = True
    seed: int = 42
    n_users: int = 5000
    n_merchants: int = 2000


def build_features(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """
    Build a production-style feature set from the raw creditcard schema.

    Input expectations:
    - creditcard.csv-like columns: Time, V1..V28, Amount
    Output:
    - base features + optional engineered temporal/behavior features
    """
    out = df.copy()
    if cfg.add_behavior:
        out = add_synthetic_behavior_features(
            out, seed=cfg.seed, n_users=cfg.n_users, n_merchants=cfg.n_merchants
        )
    if cfg.add_temporal:
        out = add_temporal_features(out)
    return out

