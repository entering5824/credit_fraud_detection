"""SHAP explainability with cached TreeExplainer (warm-start per model version)."""

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


# ---------------------------------------------------------------------------
# Explainer cache — keyed by (model_name, version_str)
# Creating shap.TreeExplainer is expensive; we do it once per model version.
# ---------------------------------------------------------------------------

_EXPLAINER_CACHE: dict[str, Any] = {}


def _cache_key(model_name: str, version: Optional[str]) -> str:
    return f"{model_name}::{version or 'latest'}"


def _get_or_create_explainer(model: Any, cache_key: str) -> Any:
    if cache_key not in _EXPLAINER_CACHE:
        import shap
        _EXPLAINER_CACHE[cache_key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[cache_key]


def warm_start_explainer(
    model_name: str = "xgboost",
    version: Optional[str] = None,
) -> None:
    """
    Pre-load model and create TreeExplainer at process startup.
    Call this in your ASGI lifespan / startup hook to avoid cold-start latency.
    """
    try:
        model, scaler, feature_names, _ = load_artifacts_from_registry(model_name, version)
    except Exception:
        try:
            model, scaler, feature_names = _load_legacy_xgboost()
        except Exception:
            return
    key = _cache_key(model_name, version)
    _get_or_create_explainer(model, key)


def clear_explainer_cache() -> None:
    """Clear the explainer cache (e.g. after a model deployment)."""
    _EXPLAINER_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_legacy_xgboost() -> tuple[Any, Any, list[str]]:
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


def _get_shap_values(explainer: Any, X: np.ndarray) -> np.ndarray:
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
    return [
        {
            "feature": feature_names[int(i)],
            "shap_value": float(sv[int(i)]),
            "feature_value": float(fv[int(i)]),
            "direction": "increases_risk" if sv[int(i)] > 0 else "decreases_risk",
        }
        for i in idx
    ]


def _narrative_from_top_features(top: list[dict]) -> str:
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
    bullets = "\n".join(f"- {x}" for x in flags[:4])
    return f"Transaction flagged due to:\n{bullets}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_transaction(
    features: dict,
    model_name: str = "xgboost",
    threshold: float = 0.5,
    top_k: int = 5,
    version: Optional[str] = None,
) -> Explanation:
    """
    Explain a single transaction using SHAP.

    The TreeExplainer is cached per (model_name, version) so it is created once
    per process rather than on every request.
    Falls back to legacy `models/xgboost.pkl` if the registry is unavailable.
    """
    model = scaler = feature_names = None
    try:
        model, scaler, feature_names, _ = load_artifacts_from_registry(model_name, version)
    except Exception:
        if model_name != "xgboost":
            raise
        model, scaler, feature_names = _load_legacy_xgboost()

    key = _cache_key(model_name, version)
    explainer = _get_or_create_explainer(model, key)

    x_raw = _as_row(features, feature_names)
    x_model = scaler.transform(x_raw) if scaler is not None else x_raw
    proba = _predict_proba(model, x_model)
    pred = int(proba >= float(threshold))

    shap_vals = _get_shap_values(explainer, x_model)[0]
    top = _top_k_explanations(shap_vals, feature_names, x_raw, top_k=top_k)
    narrative = _narrative_from_top_features(top)

    return Explanation(
        fraud_probability=float(proba),
        prediction=pred,
        top_features_contributing=top,
        narrative=narrative,
    )
