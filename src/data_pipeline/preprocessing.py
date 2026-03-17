from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover
    SMOTE = None


@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    val_size_adjusted = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=cfg.random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X[feature_cols].to_numpy())
    return scaler


def transform_with_scaler(
    X: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler
) -> pd.DataFrame:
    X_out = X.copy()
    X_out[feature_cols] = scaler.transform(X[feature_cols].to_numpy())
    return X_out


def apply_smote_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    if SMOTE is None:
        raise ImportError("imbalanced-learn is required for SMOTE.")
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train.to_numpy(), y_train.to_numpy())
    return X_res, y_res

