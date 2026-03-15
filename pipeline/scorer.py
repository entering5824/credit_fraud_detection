"""
Load model + scaler + threshold and score a transaction (feature dict).
"""

import sys
from pathlib import Path
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

_model = None
_scaler = None
_threshold = 0.5


def _load():
    global _model, _scaler, _threshold
    if _model is not None:
        return
    from src.cost_sensitive import load_cost_config
    with open(ROOT / "models" / "xgboost.pkl", "rb") as f:
        _model = pickle.load(f)
    with open(ROOT / "models" / "scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)
    cfg = load_cost_config()
    _threshold = cfg.get("optimal_threshold", 0.5)


def score(features: dict) -> dict:
    """
    Score one transaction. features: dict with V1..V28, Amount.
    Returns: fraud_score (prob), alert (bool), threshold_used.
    """
    _load()
    row = [float(features.get(k, 0)) for k in FEATURE_COLS]
    X = np.array([row])
    X_scaled = _scaler.transform(X)
    proba = _model.predict_proba(X_scaled)[0][1]
    alert = proba >= _threshold
    return {
        "fraud_score": float(proba),
        "alert": bool(alert),
        "threshold_used": float(_threshold),
    }
