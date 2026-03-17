from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

from src.core.paths import get_paths


DEFAULT_TARGET_COL = "Class"


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    target_col: str = DEFAULT_TARGET_COL
    time_col: str = "Time"
    dataset_name: str = "creditcard"
    labeled: bool = True


def default_creditcard_spec() -> DatasetSpec:
    paths = get_paths()
    # Keep backward compatibility: the repo already uses `data/creditcard.csv`
    return DatasetSpec(path=paths.data_dir / "creditcard.csv", dataset_name="creditcard")


def default_all_dataset_specs() -> list[DatasetSpec]:
    """
    Specs for all datasets under `data/` that the project supports.

    Note: Some datasets are extremely large; `load_all_datasets` supports row limits.
    """
    paths = get_paths()
    d = paths.data_dir
    return [
        DatasetSpec(path=d / "bank_transactions_data_2.csv", target_col="", time_col="TransactionDate", dataset_name="bank_transactions_data_2", labeled=False),
        DatasetSpec(path=d / "credit_card_fraud_10k.csv", target_col="is_fraud", time_col="transaction_hour", dataset_name="credit_card_fraud_10k"),
        DatasetSpec(path=d / "creditcard_2023.csv", target_col="Class", time_col="Time", dataset_name="creditcard_2023"),
        DatasetSpec(path=d / "creditcard.csv", target_col="Class", time_col="Time", dataset_name="creditcard"),
        DatasetSpec(path=d / "onlinefraud.csv", target_col="isFraud", time_col="step", dataset_name="onlinefraud"),
        DatasetSpec(path=d / "PS_20174392719_1491204439457_log.csv", target_col="isFraud", time_col="step", dataset_name="PS_log"),
        DatasetSpec(path=d / "Synthetic_Financial_datasets_log.csv", target_col="isFraud", time_col="step", dataset_name="Synthetic_Financial_datasets_log"),
        DatasetSpec(path=d / "transactions.csv", target_col="is_fraud", time_col="transaction_time", dataset_name="transactions"),
    ]


def load_dataset(spec: DatasetSpec | None = None) -> pd.DataFrame:
    spec = default_creditcard_spec() if spec is None else spec
    if not spec.path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {spec.path}. Place creditcard.csv under `data/`."
        )
    df = pd.read_csv(spec.path)
    return df


def _to_time_seconds(series: pd.Series, dataset_name: str) -> pd.Series:
    """
    Convert a dataset-specific time column into a numeric `Time` (seconds).
    """
    s = series
    # already numeric (creditcard Time, Paysim step, etc.)
    if np.issubdtype(s.dtype, np.number):
        return pd.to_numeric(s, errors="coerce").astype(float).fillna(0.0)

    # parse datetime strings
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        # epoch seconds
        return (dt.astype("int64") / 1e9).astype(float).fillna(0.0)

    # fallback: try numeric conversion
    return pd.to_numeric(s, errors="coerce").astype(float).fillna(0.0)


def canonicalize_dataset(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """
    Canonicalize any supported dataset to include:
      - dataset_name (str)
      - Class (0/1 int)
      - Time (float seconds-like)
      - Amount (float) if an amount-like column exists

    The remaining columns are kept as raw features (categoricals remain object).
    """
    out = df.copy()
    out["dataset_name"] = spec.dataset_name
    out["is_labeled"] = int(bool(spec.labeled))

    # label
    if spec.labeled:
        if not spec.target_col or spec.target_col not in out.columns:
            raise ValueError(f"[{spec.dataset_name}] missing target column: {spec.target_col}")
        out["Class"] = pd.to_numeric(out[spec.target_col], errors="coerce").fillna(0).astype(int)
    else:
        out["Class"] = np.nan

    # time
    if spec.time_col in out.columns:
        out["Time"] = _to_time_seconds(out[spec.time_col], dataset_name=spec.dataset_name)
    else:
        out["Time"] = 0.0

    # amount mapping (best-effort per dataset family)
    amount_candidates = [
        "Amount",
        "amount",
        "TransactionAmount",
        "transaction_amount",
        "Transaction Amount",
    ]
    amt_col = next((c for c in amount_candidates if c in out.columns), None)
    if amt_col is None and "TransactionAmount" in out.columns:
        amt_col = "TransactionAmount"
    if amt_col is not None:
        out["Amount"] = pd.to_numeric(out[amt_col], errors="coerce").fillna(0.0).astype(float)
    else:
        # Some datasets don't have a clear amount; set to 0 (still trainable via other signals)
        out["Amount"] = 0.0

    # drop duplicated label/time source columns to reduce leakage/confusion (keep original cols if you want)
    # Keep raw cols; only ensure canonical cols exist.
    return out


def iter_datasets(
    specs: list[DatasetSpec] | None = None,
    max_rows_per_dataset: int | None = 200_000,
):
    """
    Yield (spec, dataframe) for each existing dataset, one at a time.
    Use this for low-memory pipelines: only one dataset is in memory.
    """
    specs = default_all_dataset_specs() if specs is None else specs
    for spec in specs:
        if not spec.path.exists():
            continue
        if max_rows_per_dataset is None:
            df = pd.read_csv(spec.path)
        else:
            df = pd.read_csv(spec.path, nrows=int(max_rows_per_dataset))
        df = canonicalize_dataset(df, spec)
        yield spec, df


def load_all_datasets(
    specs: list[DatasetSpec] | None = None,
    max_rows_per_dataset: int | None = 200_000,
) -> pd.DataFrame:
    """
    Load and canonicalize all supported datasets and concatenate them.

    `max_rows_per_dataset` defaults to 200k to keep local runs feasible with very large CSVs
    (e.g. PaySim-like datasets with millions of rows). Set to None to load full files.
    """
    specs = default_all_dataset_specs() if specs is None else specs
    frames: list[pd.DataFrame] = []
    for spec in specs:
        if not spec.path.exists():
            continue
        if max_rows_per_dataset is None:
            df = pd.read_csv(spec.path)
        else:
            df = pd.read_csv(spec.path, nrows=int(max_rows_per_dataset))
        frames.append(canonicalize_dataset(df, spec))
    if not frames:
        raise FileNotFoundError("No datasets found to load under `data/`.")
    df_all = pd.concat(frames, ignore_index=True)
    validate_dataset_schema(df_all)
    return df_all


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """
    Lightweight schema validation for training.

    - requires `Class`
    - checks dataset is not >90% missing overall
    """
    if "Class" not in df.columns:
        raise ValueError("Dataset missing target column 'Class'.")

    missing_ratio = df.isna().mean().mean()
    if missing_ratio > 0.9:
        raise ValueError("Dataset mostly empty (>90% NaN).")


def validate_creditcard_schema(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    require_time: bool = True,
) -> None:
    required = {target_col, "Amount"} | {f"V{i}" for i in range(1, 29)}
    if require_time:
        required.add("Time")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def select_base_features(
    df: pd.DataFrame,
    include_time: bool = True,
) -> pd.DataFrame:
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if include_time and "Time" in df.columns:
        cols = ["Time"] + cols
    return df[cols].copy()


def get_splits(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    mode: Literal["random"] = "random",
) -> tuple[pd.DataFrame, pd.Series]:
    # This function intentionally stays minimal; splitting logic lives in preprocessing.
    if mode != "random":
        raise NotImplementedError("Only random split is implemented initially.")
    X = select_base_features(df, include_time=True)
    y = df[target_col].astype(int)
    return X, y

