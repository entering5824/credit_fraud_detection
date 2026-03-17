from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.paths import get_paths
from src.models.model_registry import load_artifacts_from_registry


@dataclass(frozen=True)
class Explanation:
    fraud_probability: float
    prediction: int
    top_features_contributing: list[dict]
    narrative: str


def _load_legacy_xgboost() -> tuple[Any, Any, list[str]]:
    """
    Backward compatibility: use `models/xgboost.pkl` + `models/scaler.pkl`.
    """
    paths = get_paths()
    model_path = paths.models_dir / "xgboost.pkl"
    scaler_path = paths.models_dir / "scaler.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return model, scaler, feature_names


def _predict_proba(model: Any, X: np.ndarray) -> float:
    return float(model.predict_proba(X)[0][1])


def _as_row(features: dict, feature_names: list[str]) -> np.ndarray:
    return np.array([[float(features.get(k, 0.0)) for k in feature_names]], dtype=float)


def _get_shap_values(model: Any, X: np.ndarray) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(model)
    sh = explainer.shap_values(X)
    if isinstance(sh, list):
        sh = sh[1]
    return np.asarray(sh)


def _top_k_explanations(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    feature_values_row: np.ndarray,
    top_k: int,
) -> list[dict]:
    sv = shap_values_row.reshape(-1)
    fv = feature_values_row.reshape(-1)
    idx = np.argsort(np.abs(sv))[::-1][:top_k]
    out = []
    for i in idx:
        out.append(
            {
                "feature": feature_names[int(i)],
                "shap_value": float(sv[int(i)]),
                "feature_value": float(fv[int(i)]),
                "direction": "increases_risk" if sv[int(i)] > 0 else "decreases_risk",
            }
        )
    return out


def _narrative_from_top_features(top: list[dict]) -> str:
    """
    Small ruleset that turns top features into a portfolio-friendly explanation.
    """
    flags: list[str] = []
    feats = {t["feature"] for t in top}

    if "Amount" in feats:
        flags.append("unusually high transaction amount (relative to learned patterns)")
    if "spending_spike_ratio" in feats:
        flags.append("spending spike vs. the account's typical amount")
    if "is_new_merchant_for_user" in feats:
        flags.append("new merchant pattern for this user")
    if "transactions_last_1h" in feats or "transaction_velocity_1h" in feats:
        flags.append("high short-term transaction velocity")
    if "time_since_last_transaction" in feats:
        flags.append("abnormal timing since the previous transaction")

    if not flags:
        return "Transaction flagged due to a combination of anomalous feature patterns."

    bullets = "\n".join([f"- {x}" for x in flags[:4]])
    return f"Transaction flagged due to:\n{bullets}"


def explain_transaction(
    features: dict,
    model_name: str = "xgboost",
    threshold: float = 0.5,
    top_k: int = 5,
    version: Optional[str] = None,
) -> Explanation:
    """
    Explain a single transaction using SHAP.

    - Uses the model registry if available (models/<model_name>/<version>/...)
    - Falls back to legacy `models/xgboost.pkl` for backward compatibility.
    """
    model = scaler = feature_names = None
    try:
        model, scaler, feature_names, _ = load_artifacts_from_registry(
            model_name=model_name, version=version
        )
    except Exception:
        if model_name != "xgboost":
            raise
        model, scaler, feature_names = _load_legacy_xgboost()

    x_raw = _as_row(features, feature_names)
    x_model = scaler.transform(x_raw) if scaler is not None else x_raw
    proba = _predict_proba(model, x_model)
    pred = int(proba >= float(threshold))

    shap_vals = _get_shap_values(model, x_model)[0]
    top = _top_k_explanations(shap_vals, feature_names, x_raw, top_k=top_k)
    narrative = _narrative_from_top_features(top)

    return Explanation(
        fraud_probability=float(proba),
        prediction=pred,
        top_features_contributing=top,
        narrative=narrative,
    )

