from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline.dataset_loader import load_all_datasets  # noqa: E402
from src.features.unified_features import build_unified_model_frame  # noqa: E402


def main() -> None:
    df = load_all_datasets(max_rows_per_dataset=2000)
    df_labeled = df[df["Class"].isin([0, 1])].copy()

    X = build_unified_model_frame(df_labeled)

    print("Rows:", len(df_labeled), "Cols:", X.shape[1])
    print("Datasets:", df_labeled["dataset_name"].value_counts().to_dict())

    if "feature_source" in X.columns:
        src = X["feature_source"].to_numpy()
        total = len(src)
        print(
            "feature_source ratio:",
            {
                "credit_like": float(np.mean(src == 1)) if total else 0.0,
                "generic": float(np.mean(src == 0)) if total else 0.0,
            },
        )

    nan_count = int(np.isnan(X.to_numpy(dtype=float)).sum())
    print("NaN count in feature matrix:", nan_count)


if __name__ == "__main__":
    main()

