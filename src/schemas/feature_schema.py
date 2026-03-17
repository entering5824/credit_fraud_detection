from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


BASE_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
OPTIONAL_FEATURES = [
    "Time",
    # Engineered features (optional for inference; required only if the model expects them)
    "synthetic_user_id",
    "synthetic_merchant_id",
    "user_avg_amount",
    "user_med_amount",
    "spending_spike_ratio",
    "user_merchant_count",
    "is_new_merchant_for_user",
    "user_distinct_merchants_so_far",
    "merchant_frequency_score",
    "time_of_day_sin",
    "time_of_day_cos",
    "time_since_last_transaction",
    "transactions_last_1h",
    "transactions_last_24h",
    "transaction_velocity_1h",
    "transaction_velocity_24h",
]


def validate_feature_dict(features: dict[str, Any], required: Optional[list[str]] = None) -> None:
    required = BASE_FEATURES if required is None else required
    missing = [c for c in required if c not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    # Basic numeric validation
    for k in required:
        v = features.get(k)
        try:
            float(v)
        except Exception as e:
            raise ValueError(f"Feature '{k}' must be numeric; got {v!r}") from e


def validate_feature_frame(
    df: pd.DataFrame,
    required: Optional[list[str]] = None,
    allow_extra: bool = True,
) -> None:
    required = BASE_FEATURES if required is None else required
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type check required columns are numeric-ish
    bad = []
    for c in required:
        if not np.issubdtype(df[c].dtype, np.number):
            # allow object if convertible
            try:
                pd.to_numeric(df[c], errors="raise")
            except Exception:
                bad.append(c)
    if bad:
        raise ValueError(f"Non-numeric required columns: {bad}")

    if not allow_extra:
        allowed = set(required)
        extra = sorted(set(df.columns) - allowed)
        if extra:
            raise ValueError(f"Unexpected extra columns: {extra}")


def coerce_feature_frame(df: pd.DataFrame, required: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Return a copy with required columns coerced to float.
    """
    required = BASE_FEATURES if required is None else required
    out = df.copy()
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out

