"""
Anomaly detection (unsupervised) for credit card fraud.

Provides:
- Isolation Forest
- Local Outlier Factor (LOF)
- Autoencoder (reconstruction error as anomaly score)

Comparison with supervised (XGBoost) via AUC/F1 on same test set.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, f1_score
from typing import Tuple, Optional, Dict, Any


def train_isolation_forest(
    X: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.01,
    random_state: int = 42,
    verbose: int = 1,
) -> IsolationForest:
    """Train Isolation Forest. Fit on normal-like data (no labels)."""
    if verbose >= 1:
        print("Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X)
    if verbose >= 1:
        print("Isolation Forest training completed.")
    return clf


def train_lof(
    X: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.01,
    verbose: int = 1,
) -> LocalOutlierFactor:
    """Train LOF. Fit on X (no labels)."""
    if verbose >= 1:
        print("Training Local Outlier Factor...")
    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
    )
    clf.fit(X)
    if verbose >= 1:
        print("LOF training completed.")
    return clf


def train_autoencoder(
    X: np.ndarray,
    hidden_dims: Tuple[int, ...] = (32, 16),
    max_iter: int = 200,
    random_state: int = 42,
    verbose: int = 1,
) -> MLPRegressor:
    """
    Train autoencoder: input -> hidden -> output (reconstruct).
    Anomaly score = mean squared reconstruction error per sample.
    """
    if verbose >= 1:
        print("Training Autoencoder...")
    n_features = X.shape[1]
    # Build layer sizes: input -> hidden -> ... -> input
    layer_sizes = (n_features,) + hidden_dims + (n_features,)
    hidden_layer_sizes = layer_sizes[1:-1]
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X, X)
    if verbose >= 1:
        print("Autoencoder training completed.")
    return model


def anomaly_scores_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Isolation Forest: use negative decision_function so higher = more anomalous."""
    return -model.decision_function(X)


def anomaly_scores_lof(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    """LOF: use negative decision_function so higher = more anomalous."""
    return -model.decision_function(X)


def anomaly_scores_ae(model: MLPRegressor, X: np.ndarray) -> np.ndarray:
    """Autoencoder: reconstruction MSE per sample (higher = more anomalous)."""
    X_recon = model.predict(X)
    return np.mean((X - X_recon) ** 2, axis=1)


def scores_to_binary(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert anomaly scores to binary (1 = anomaly/fraud). Above threshold -> 1."""
    return (scores >= threshold).astype(np.int64)


def optimal_threshold_from_expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fn: float = 1000.0,
    cost_fp: float = 10.0,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Find threshold that minimizes expected cost.
    scores: higher = more likely fraud (anomaly).
    Returns (best_threshold, min_expected_cost).
    """
    if thresholds is None:
        thresholds = np.percentile(scores, np.linspace(1, 99, 99))
    min_cost = np.inf
    best_t = 0.0
    for t in thresholds:
        y_pred = (scores >= t).astype(np.int64)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        cost = fn * cost_fn + fp * cost_fp
        if cost < min_cost:
            min_cost = cost
            best_t = t
    return best_t, min_cost


def evaluate_anomaly_model(
    scores: np.ndarray,
    y_true: np.ndarray,
    model_name: str = "Anomaly",
) -> Dict[str, Any]:
    """
    Compute AUC (scores as predictor of y_true) and best F1 threshold.
    For anomaly, fraud = 1; higher score = more anomalous = fraud.
    """
    if np.unique(y_true).size < 2:
        return {"auc": 0.0, "f1": 0.0, "threshold": 0.0}
    auc = roc_auc_score(y_true, scores)
    # Grid search threshold for max F1
    thresh_vals = np.percentile(scores, np.linspace(1, 99, 50))
    best_f1 = 0.0
    best_t = 0.0
    for t in thresh_vals:
        y_pred = (scores >= t).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return {"auc": float(auc), "f1": float(best_f1), "threshold": float(best_t)}


def save_anomaly_models(
    models: Dict[str, Any],
    scaler: Any,
    save_dir: str = "models",
) -> Path:
    """Save anomaly models and scaler to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in models.items():
        path = save_dir / f"anomaly_{name.lower().replace(' ', '_')}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    with open(save_dir / "anomaly_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return save_dir


def load_anomaly_model(name: str, model_dir: str = "models") -> Any:
    """Load a single anomaly model by name (isolation_forest, lof, autoencoder)."""
    path = Path(model_dir) / f"anomaly_{name.lower().replace(' ', '_')}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
