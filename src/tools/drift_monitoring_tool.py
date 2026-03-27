"""Drift monitoring tool: single-row or batch analysis against a baseline distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.paths import get_paths
from src.monitoring.drift_detection import DriftConfig, drift_report
from src.schemas.feature_schema import BASE_FEATURES
from src.tools.base import Tool
from src.tools.registry import register_tool

_baseline_df: Optional[pd.DataFrame] = None
_baseline_path: Optional[Path] = None

DRIFT_MONITORING_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "object",
            "description": "Single transaction to evaluate.",
        },
        "current_batch": {
            "type": "array",
            "description": "Batch of transaction dicts instead of a single row.",
        },
        "baseline_path": {
            "type": "string",
            "description": "Absolute path to a baseline CSV (defaults to data/creditcard.csv).",
        },
        "psi_threshold": {
            "type": "number",
            "description": "PSI ≥ this marks a feature as drifted.",
            "default": 0.2,
        },
    },
    "required": [],
}

DRIFT_MONITORING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_drifted": {"type": "boolean"},
        "dataset_psi": {
            "type": "number",
            "description": "Mean PSI across all evaluated features.",
        },
        "fraud_rate_shift": {
            "type": "number",
            "description": "Difference in fraud rate between baseline and current.",
        },
        "feature_drift": {
            "type": "array",
            "description": "Per-feature PSI sorted descending.",
        },
        "top_drifted_features": {
            "type": "array",
            "description": "Features where PSI ≥ psi_threshold.",
        },
        "psi_summary": {"type": "object"},
    },
}


def _load_baseline(baseline_path: Optional[str] = None) -> pd.DataFrame:
    global _baseline_df, _baseline_path
    path: Optional[Path] = None
    if baseline_path:
        path = Path(baseline_path)
    if path is None:
        paths = get_paths()
        for candidate in [
            paths.data_dir / "creditcard.csv",
            paths.data_raw_dir / "creditcard.csv",
            paths.data_dir / "raw" / "creditcard.csv",
        ]:
            if candidate.exists():
                path = candidate
                break
    if path is None or not path.exists():
        return pd.DataFrame()
    if _baseline_df is not None and _baseline_path == path:
        return _baseline_df
    try:
        df = pd.read_csv(path).head(50_000)
        if "Class" not in df.columns:
            df["Class"] = 0
        _baseline_df = df
        _baseline_path = path
        return _baseline_df
    except Exception:
        return pd.DataFrame()


def _row_to_df(features: dict) -> pd.DataFrame:
    row: dict = {k: float(features.get(k, 0.0)) for k in BASE_FEATURES}
    row["Time"] = float(features.get("Time", 0.0))
    row["Class"] = int(features.get("Class", 0))
    return pd.DataFrame([row])


def _execute(
    features: Optional[dict] = None,
    current_batch: Optional[list[dict]] = None,
    baseline_path: Optional[str] = None,
    psi_threshold: float = 0.2,
) -> dict:
    baseline = _load_baseline(baseline_path)
    if baseline.empty:
        return {
            "is_drifted": False,
            "dataset_psi": 0.0,
            "fraud_rate_shift": 0.0,
            "feature_drift": [],
            "top_drifted_features": [],
            "psi_summary": {},
            "note": "Baseline not available; drift check skipped.",
        }
    # Build current DataFrame
    if current_batch:
        current_df = pd.concat([_row_to_df(t) for t in current_batch], ignore_index=True)
    elif features:
        current_df = _row_to_df(features)
    else:
        return {
            "is_drifted": False,
            "dataset_psi": 0.0,
            "fraud_rate_shift": 0.0,
            "feature_drift": [],
            "top_drifted_features": [],
            "psi_summary": {},
            "note": "No input provided.",
        }
    if "Class" not in current_df.columns:
        current_df["Class"] = 0

    try:
        rep = drift_report(baseline, current_df, cfg=DriftConfig())
    except Exception as exc:
        return {
            "is_drifted": False,
            "dataset_psi": 0.0,
            "fraud_rate_shift": 0.0,
            "feature_drift": [],
            "top_drifted_features": [],
            "psi_summary": {},
            "note": f"Drift check failed: {exc!s}",
        }

    all_drifts = rep.get("all_feature_drifts", [])
    top_drifted = [r for r in all_drifts if r.get("psi", 0) >= psi_threshold]
    dataset_psi = float(np.mean([r.get("psi", 0) for r in all_drifts])) if all_drifts else 0.0
    fraud_rate_shift = float(rep.get("fraud_rate_delta", 0.0))

    return {
        "is_drifted": len(top_drifted) > 0,
        "dataset_psi": round(dataset_psi, 4),
        "fraud_rate_shift": round(fraud_rate_shift, 6),
        "feature_drift": [
            {"feature": r["feature"], "psi": round(r.get("psi", 0), 4)}
            for r in all_drifts[:20]
        ],
        "top_drifted_features": [
            {"feature": r["feature"], "psi": round(r.get("psi", 0), 4)}
            for r in top_drifted[:10]
        ],
        "psi_summary": rep.get("psi_guidance", {}),
    }


drift_monitoring_tool = register_tool(
    Tool(
        name="drift_monitoring",
        description=(
            "Checks whether a transaction or a batch deviates from the training distribution. "
            "Returns dataset_psi, per-feature drift list, fraud_rate_shift, "
            "and top drifted features (PSI ≥ threshold)."
        ),
        input_schema=DRIFT_MONITORING_INPUT_SCHEMA,
        output_schema=DRIFT_MONITORING_OUTPUT_SCHEMA,
        execute=_execute,
        timeout_seconds=30,
    )
)
