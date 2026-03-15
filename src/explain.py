"""
Explainability: SHAP for XGBoost/RF (TreeExplainer).
Global feature importance and local SHAP values per transaction.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _load_model():
    with open(ROOT / "models" / "xgboost.pkl", "rb") as f:
        return pickle.load(f)


def get_shap_explainer(model=None, X_background: Optional[np.ndarray] = None):
    """Create SHAP TreeExplainer for XGBoost. Optional X_background for background set."""
    import shap
    if model is None:
        model = _load_model()
    # TreeExplainer for tree models
    explainer = shap.TreeExplainer(model, data=X_background)
    return explainer


def global_importance(model=None, X: Optional[np.ndarray] = None) -> Tuple[List[str], np.ndarray]:
    """
    Global feature importance (mean |SHAP|). If X is None, returns model's native importance.
    Returns (feature_names, importance_array).
    """
    if model is None:
        model = _load_model()
    try:
        import shap
        if X is not None and len(X) > 0:
            explainer = shap.TreeExplainer(model)
            sh = explainer.shap_values(X)
            if isinstance(sh, list):
                sh = sh[1]  # binary: index 1 = positive class
            imp = np.mean(np.abs(sh), axis=0)
        else:
            imp = model.feature_importances_
        return FEATURE_COLS, np.asarray(imp)
    except Exception:
        return FEATURE_COLS, np.asarray(model.feature_importances_)


def local_shap_values(X: np.ndarray, model=None) -> np.ndarray:
    """SHAP values for one or more samples. X: (n_samples, n_features)."""
    if model is None:
        model = _load_model()
    import shap
    explainer = shap.TreeExplainer(model)
    sh = explainer.shap_values(X)
    if isinstance(sh, list):
        sh = sh[1]
    return np.asarray(sh)
