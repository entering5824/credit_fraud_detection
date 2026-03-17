from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def save_roc_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir,
    prefix: str,
) -> dict:
    """
    Save ROC and PR curves for an imbalanced classifier.
    Returns paths as strings.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roc_path = out_dir / f"{prefix}_roc.png"
    pr_path = out_dir / f"{prefix}_pr.png"

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {prefix}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, linewidth=2, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {prefix}")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {"roc_path": str(roc_path), "pr_path": str(pr_path)}


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ThresholdMetrics:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return ThresholdMetrics(
        threshold=float(threshold),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        precision=float(precision),
        recall=float(recall),
    )


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one extra point; align with thresholds
    precision = precision[:-1]
    recall = recall[:-1]
    denom = (precision + recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    idx = int(np.nanargmax(f1)) if f1.size else 0
    return float(thresholds[idx]) if thresholds.size else 0.5


def threshold_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> float:
    """
    Return the *largest* threshold whose FPR <= target_fpr.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # roc_curve thresholds are decreasing; pick feasible thresholds then max threshold (most strict)
    mask = fpr <= target_fpr
    if not np.any(mask):
        # If even the strictest threshold exceeds target_fpr, return 1.0 (never alert)
        return 1.0
    feasible_thresholds = thresholds[mask]
    return float(np.max(feasible_thresholds))


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def summarize_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
    fpr_targets: tuple[float, ...] = (0.01, 0.001),
) -> dict:
    """
    Standard summary for imbalanced fraud detection.
    """
    out = {
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
    }

    for fpr_t in fpr_targets:
        out[f"recall_at_fpr_{fpr_t:g}"] = recall_at_fpr(y_true, y_score, fpr_t)

    if threshold is None:
        threshold = best_f1_threshold(y_true, y_score)
        out["threshold_strategy"] = "best_f1"
    else:
        out["threshold_strategy"] = "fixed"
    out["threshold"] = float(threshold)

    cm = confusion_at_threshold(y_true, y_score, float(threshold))
    out.update(
        {
            "precision": cm.precision,
            "recall": cm.recall,
            "tp": cm.tp,
            "fp": cm.fp,
            "tn": cm.tn,
            "fn": cm.fn,
        }
    )
    return out

