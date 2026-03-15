"""
Cost-sensitive learning: optimize decision threshold to minimize expected cost.

Business costs:
- FN (miss fraud): cost_fn (e.g. $1000)
- FP (false alert): cost_fp (e.g. $10)

Expected cost = FN * cost_fn + FP * cost_fp.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


DEFAULT_COST_FN = 1000.0
DEFAULT_COST_FP = 10.0


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """Compute expected cost given binary predictions."""
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * cost_fn + fp * cost_fp)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Grid search threshold to minimize expected cost.
    Returns (best_threshold, min_expected_cost).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    min_cost = np.inf
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(np.int64)
        cost = expected_cost(y_true, y_pred, cost_fn, cost_fp)
        if cost < min_cost:
            min_cost = cost
            best_t = t
    return best_t, min_cost


def load_cost_config(config_path: Optional[Path] = None) -> dict:
    """Load cost config (cost_fn, cost_fp, optimal_threshold) from JSON."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "cost_config.json"
    if not config_path.exists():
        return {
            "cost_fn": DEFAULT_COST_FN,
            "cost_fp": DEFAULT_COST_FP,
            "optimal_threshold": 0.5,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cost_config(
    optimal_threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    config_path: Optional[Path] = None,
) -> Path:
    """Save optimal threshold and costs to config JSON."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "cost_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "optimal_threshold": optimal_threshold,
        "description": "cost_fn = cost of missing fraud (FN), cost_fp = cost of false alert (FP). optimal_threshold minimizes expected cost on validation set.",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return config_path
