from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.core.paths import get_paths
from src.data_pipeline.dataset_loader import validate_creditcard_schema
from src.features.feature_engineering import build_features


@dataclass(frozen=True)
class DriftConfig:
    bins: int = 10
    max_features: int = 60


def _psi_from_counts(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    eps = 1e-12
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Use expected quantiles as bin edges for stability.
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, q))
    if edges.size < 3:  # too few unique edges
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    return _psi_from_counts(exp_pct, act_pct)


def ks_pvalue(expected: np.ndarray, actual: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import ks_2samp
    except Exception:
        return None
    res = ks_2samp(expected, actual, alternative="two-sided", mode="auto")
    return float(res.pvalue)


def build_monitoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the same engineered feature set used by the pipeline for drift monitoring.
    """
    validate_creditcard_schema(df, target_col="Class", require_time=True)
    feats = build_features(df.drop(columns=["Class"]))
    feats["Class"] = df["Class"].astype(int).to_numpy()
    return feats


def drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    cfg: DriftConfig = DriftConfig(),
) -> dict:
    base = build_monitoring_frame(baseline_df)
    cur = build_monitoring_frame(current_df)

    numeric_cols = [c for c in base.columns if c != "Class" and np.issubdtype(base[c].dtype, np.number)]
    numeric_cols = numeric_cols[: cfg.max_features]

    feature_rows = []
    for col in numeric_cols:
        exp = base[col].to_numpy()
        act = cur[col].to_numpy()
        feature_rows.append(
            {
                "feature": col,
                "psi": psi(exp, act, bins=cfg.bins),
                "ks_pvalue": ks_pvalue(exp, act),
                "baseline_mean": float(np.nanmean(exp)),
                "current_mean": float(np.nanmean(act)),
            }
        )

    feature_rows = sorted(feature_rows, key=lambda r: r["psi"], reverse=True)
    base_fraud = float(base["Class"].mean())
    cur_fraud = float(cur["Class"].mean())

    return {
        "baseline_rows": int(len(base)),
        "current_rows": int(len(cur)),
        "baseline_fraud_rate": base_fraud,
        "current_fraud_rate": cur_fraud,
        "fraud_rate_delta": cur_fraud - base_fraud,
        "top_feature_drifts": feature_rows[:20],
        "all_feature_drifts": feature_rows,
        "psi_guidance": {
            "psi<0.1": "no_material_shift",
            "0.1<=psi<0.2": "moderate_shift_investigate",
            "psi>=0.2": "large_shift_retraining_candidate",
        },
    }


def main() -> None:
    paths = get_paths()
    paths.ensure_dirs()

    parser = argparse.ArgumentParser(description="Compute feature drift report (PSI/KS + fraud rate).")
    parser.add_argument("--baseline", type=str, default=str(paths.data_dir / "creditcard.csv"))
    parser.add_argument("--current", type=str, default=str(paths.data_dir / "creditcard.csv"))
    parser.add_argument("--out", type=str, default=str(paths.results_monitoring_dir / "drift_report.json"))
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    baseline = pd.read_csv(Path(args.baseline)).head(int(args.max_rows))
    current = pd.read_csv(Path(args.current)).head(int(args.max_rows))

    rep = drift_report(baseline, current, cfg=DriftConfig(bins=int(args.bins)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()

