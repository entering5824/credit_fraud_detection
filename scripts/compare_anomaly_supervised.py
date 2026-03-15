"""
Compare supervised (XGBoost) vs unsupervised (Isolation Forest, LOF, Autoencoder)
on credit card fraud detection. Same test set; report AUC and F1.
Run from project root: python scripts/compare_anomaly_supervised.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from src.data_preprocessing import scale_features, split_data
from src.models import train_xgboost, load_model
from src.anomaly import (
    train_isolation_forest,
    train_lof,
    train_autoencoder,
    anomaly_scores_if,
    anomaly_scores_lof,
    anomaly_scores_ae,
    evaluate_anomaly_model,
    save_anomaly_models,
)

def main():
    print("Loading data...")
    df = pd.read_csv(ROOT / "data" / "creditcard.csv")
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols]
    y = df["Class"]

    print("Scaling and splitting...")
    X_scaled, scaler = scale_features(X, feature_cols=feature_cols, fit=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_scaled, y, test_size=0.15, val_size=0.15, random_state=42
    )
    # Use numpy for anomaly (some APIs expect array)
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    X_test_np = X_test.values if hasattr(X_test, "values") else X_test
    y_test_np = y_test.values if hasattr(y_test, "values") else y_test

    # --- Unsupervised: fit on train only (no labels) ---
    print("\n--- Unsupervised models (fit on train) ---")
    if_model = train_isolation_forest(X_train_np, contamination=0.002, verbose=1)
    lof_model = train_lof(X_train_np, n_neighbors=20, contamination=0.002, verbose=1)
    ae_model = train_autoencoder(X_train_np, hidden_dims=(32, 16), verbose=1)

    save_anomaly_models(
        {"isolation_forest": if_model, "lof": lof_model, "autoencoder": ae_model},
        scaler,
        save_dir=str(ROOT / "models"),
    )

    # --- Scores on test ---
    scores_if = anomaly_scores_if(if_model, X_test_np)
    scores_lof = anomaly_scores_lof(lof_model, X_test_np)
    scores_ae = anomaly_scores_ae(ae_model, X_test_np)

    # --- Supervised: XGBoost (load if exists, else train) ---
    print("\n--- Supervised (XGBoost) ---")
    xgb_path = ROOT / "models" / "xgboost.pkl"
    if xgb_path.exists():
        import pickle
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        print("Loaded existing XGBoost model.")
    else:
        from src.models import get_class_weights
        cw = get_class_weights(y_train)
        scale_pos = cw.get(1, 1.0) / max(cw.get(0, 1.0), 1e-9)
        xgb_model = train_xgboost(
            X_train_np, y_train.values if hasattr(y_train, "values") else y_train,
            scale_pos_weight=scale_pos, verbose=1
        )
        ROOT.joinpath("models").mkdir(exist_ok=True)
        with open(xgb_path, "wb") as f:
            import pickle
            pickle.dump(xgb_model, f)
    xgb_proba = xgb_model.predict_proba(X_test_np)[:, 1]

    # --- Evaluate ---
    print("\n--- Comparison (test set) ---")
    results = []
    for name, scores in [
        ("Isolation Forest", scores_if),
        ("LOF", scores_lof),
        ("Autoencoder", scores_ae),
        ("XGBoost (supervised)", xgb_proba),
    ]:
        m = evaluate_anomaly_model(scores, y_test_np, model_name=name)
        results.append({"model": name, "auc": m["auc"], "f1": m["f1"], "threshold": m["threshold"]})
        print(f"  {name}: AUC={m['auc']:.4f}, F1={m['f1']:.4f} (threshold={m['threshold']:.4f})")

    out = ROOT / "results" / "anomaly_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
