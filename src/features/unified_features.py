from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import build_features


def _build_creditcard_features(base: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    v_cols = [f"V{i}" for i in range(1, 29)]
    cols = [c for c in (["Time"] + v_cols + ["Amount"]) if c in base.columns]
    sub = base.loc[mask, cols].copy()
    feats = build_features(sub)
    return feats


def _build_generic_features(base: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    # Drop target; keep everything else
    X_raw = base.loc[mask].drop(columns=["Class"], errors="ignore")
    num = X_raw.select_dtypes(include=[np.number]).copy()
    if not num.empty:
        # Clip numeric extremes to avoid exploding scales from outliers.
        num = num.clip(lower=-1e6, upper=1e6)
    cat = X_raw.select_dtypes(exclude=[np.number]).copy()
    if not cat.empty:
        cat = cat.fillna("NA").astype(str)
        cat_oh = pd.get_dummies(cat, prefix=cat.columns, drop_first=False)
        X = pd.concat([num, cat_oh], axis=1)
    else:
        X = num
    return X


def build_unified_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a model-ready feature frame from a *canonicalized* multi-dataset dataframe.

    Input expectations:
      - Must include `Time`, `Amount`, `Class`, `dataset_name`
      - May include creditcard PCA columns (V1..V28) or other raw fields

    Strategy (hybrid):
      - Per-row detect creditcard-like rows by NaN ratio on V1..V28
      - For creditcard-like rows: use existing fraud feature pipeline (behavior+temporal)
      - For generic rows: numeric + one-hot categoricals
      - Add `feature_source` flag (1=creditcard pipeline, 0=generic)
      - Merge two spaces, align columns, fill NaNs with 0.0
    """
    base = df.copy()

    # Always ensure these exist
    if "Time" not in base.columns:
        base["Time"] = 0.0
    if "Amount" not in base.columns:
        base["Amount"] = 0.0

    v_cols = [f"V{i}" for i in range(1, 29)]
    present_v = [c for c in v_cols if c in base.columns]

    if present_v:
        nan_ratio = base[present_v].isna().mean(axis=1)
        credit_mask = nan_ratio < 0.1
    else:
        credit_mask = pd.Series(False, index=base.index)

    generic_mask = ~credit_mask

    parts = []
    if credit_mask.any():
        credit_feats = _build_creditcard_features(base, credit_mask)
        credit_feats.index = base.index[credit_mask]
        credit_feats["feature_source"] = 1
        parts.append(credit_feats)

    if generic_mask.any():
        generic_feats = _build_generic_features(base, generic_mask)
        generic_feats.index = base.index[generic_mask]
        # ensure feature_source column exists
        generic_feats["feature_source"] = 0
        parts.append(generic_feats)

    if not parts:
        return pd.DataFrame(index=base.index)

    all_features = pd.concat(parts, axis=0).reindex(base.index)
    # Normalize schema: sorted columns, NaNs → 0
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.reindex(columns=sorted(all_features.columns))
    all_features = all_features.fillna(0.0)
    return all_features

