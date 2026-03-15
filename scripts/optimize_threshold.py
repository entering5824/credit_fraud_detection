"""
Optimize decision threshold to minimize expected cost (cost_fn * FN + cost_fp * FP).
Saves optimal_threshold to config/cost_config.json.
Run from project root: python scripts/optimize_threshold.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pickle
import pandas as pd
from src.data_preprocessing import scale_features, split_data
from src.cost_sensitive import find_optimal_threshold, save_cost_config

COST_FN = 1000.0
COST_FP = 10.0


def main():
    print("Loading data...")
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols]
    y = df["Class"]

    X_scaled, scaler = scale_features(X, feature_cols=feature_cols, fit=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_scaled, y, test_size=0.15, val_size=0.15, random_state=42
    )
    X_val_np = X_val.values if hasattr(X_val, "values") else X_val
    y_val_np = y_val.values if hasattr(y_val, "values") else y_val

    model_path = ROOT / "models" / "xgboost.pkl"
    if not model_path.exists():
        print("XGBoost model not found. Train it first (notebook 02 or compare_anomaly_supervised.py).")
        return
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_proba = model.predict_proba(X_val_np)[:, 1]
    best_t, min_cost = find_optimal_threshold(
        y_val_np, y_proba, cost_fn=COST_FN, cost_fp=COST_FP
    )
    print(f"Optimal threshold: {best_t:.4f}, min expected cost (val): {min_cost:.0f}")

    save_cost_config(best_t, cost_fn=COST_FN, cost_fp=COST_FP)
    print("Saved config/cost_config.json.")


if __name__ == "__main__":
    main()
