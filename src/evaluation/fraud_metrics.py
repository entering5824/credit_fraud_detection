from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.cost_sensitive import expected_cost, find_optimal_threshold
from src.evaluation.metrics import confusion_at_threshold, threshold_at_fpr


@dataclass(frozen=True)
class CostConfig:
    cost_fn: float = 1000.0
    cost_fp: float = 10.0


def cost_sensitive_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn: float = 1000.0,
    cost_fp: float = 10.0,
    n_thresholds: int = 200,
) -> dict:
    t, min_cost = find_optimal_threshold(
        y_true=y_true,
        y_proba=y_score,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        n_thresholds=n_thresholds,
    )
    cm = confusion_at_threshold(y_true, y_score, t)
    return {
        "threshold": float(t),
        "expected_cost": float(min_cost),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "precision": cm.precision,
        "recall": cm.recall,
        "tp": cm.tp,
        "fp": cm.fp,
        "tn": cm.tn,
        "fn": cm.fn,
    }


def recall_at_fixed_fpr_thresholding(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> dict:
    t = threshold_at_fpr(y_true, y_score, target_fpr=target_fpr)
    cm = confusion_at_threshold(y_true, y_score, t)
    # realized fpr at this threshold
    realized_fpr = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0
    return {
        "threshold": float(t),
        "target_fpr": float(target_fpr),
        "realized_fpr": float(realized_fpr),
        "recall": cm.recall,
        "precision": cm.precision,
        "tp": cm.tp,
        "fp": cm.fp,
        "tn": cm.tn,
        "fn": cm.fn,
    }


def expected_cost_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float = 1000.0,
    cost_fp: float = 10.0,
) -> float:
    y_pred = (y_score >= threshold).astype(np.int64)
    return expected_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)

