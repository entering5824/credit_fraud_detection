"""
Evaluation Runner — benchmarks the fraud agent against simulated and historical datasets.

Metrics computed
----------------
  auc              ROC AUC
  pr_auc           Precision-Recall AUC
  recall_at_k      Recall when flagging top-K% of transactions
  false_positive_rate  FPR at the configured threshold
  precision        Precision at threshold
  f1               F1 score at threshold
  detection_latency_ms  Mean agent investigation latency
  partial_report_rate  Fraction of investigations that returned degraded results

Usage
-----
    from src.evaluation.evaluation_runner import EvaluationRunner
    from src.simulation.fraud_simulator import FraudSimulator

    sim = FraudSimulator(seed=42)
    dataset = sim.generate_dataset(n_normal=500, n_fraud=50)

    runner = EvaluationRunner()
    results = runner.evaluate(dataset)
    print(results.summary())

    # Save to JSON
    results.save("results/evaluation/run_001.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.core.paths import get_paths


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    run_id:              str
    dataset_size:        int
    fraud_count:         int
    threshold:           float
    auc:                 float
    pr_auc:              float
    recall_at_5pct:      float
    recall_at_10pct:     float
    false_positive_rate: float
    precision:           float
    recall:              float
    f1:                  float
    detection_latency_ms_mean: float
    detection_latency_ms_p95:  float
    partial_report_rate:       float
    per_pattern:         dict[str, dict] = field(default_factory=dict)
    raw_scores:          list[float]     = field(default_factory=list)
    raw_labels:          list[int]       = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"EvaluationResult run_id={self.run_id}\n"
            f"  Dataset:    {self.dataset_size} total, {self.fraud_count} fraud "
            f"({self.fraud_count/max(self.dataset_size,1):.1%})\n"
            f"  AUC:        {self.auc:.4f}\n"
            f"  PR-AUC:     {self.pr_auc:.4f}\n"
            f"  Recall@5%:  {self.recall_at_5pct:.4f}\n"
            f"  Recall@10%: {self.recall_at_10pct:.4f}\n"
            f"  Precision:  {self.precision:.4f}\n"
            f"  Recall:     {self.recall:.4f}\n"
            f"  F1:         {self.f1:.4f}\n"
            f"  FPR:        {self.false_positive_rate:.4f}\n"
            f"  Latency p95:{self.detection_latency_ms_p95:.1f} ms\n"
            f"  Partial:    {self.partial_report_rate:.1%}\n"
        )

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]
        d.pop("raw_scores", None)
        d.pop("raw_labels", None)
        return d

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvaluationRunner:
    """
    Runs the agent over a dataset and computes fraud-detection metrics.

    Parameters
    ----------
    threshold      : decision threshold (default 0.5)
    tools_registry : inject fake tools for offline evaluation
    recall_k_pcts  : thresholds for Recall@K evaluation
    """

    def __init__(
        self,
        threshold: float = 0.5,
        tools_registry: Optional[dict] = None,
        recall_k_pcts: tuple[float, ...] = (0.05, 0.10),
    ) -> None:
        self._threshold = threshold
        self._tools_reg = tools_registry
        self._k_pcts    = recall_k_pcts

    def evaluate(
        self,
        dataset,   # list[SimTransaction]
        run_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Score every transaction in *dataset* and compute metrics.
        """
        import uuid as _uuid
        run_id = run_id or _uuid.uuid4().hex[:8]

        from src.agents.agent_orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(tools_registry=self._tools_reg)

        scores:   list[float] = []
        labels:   list[int]   = []
        latencies: list[float] = []
        partials: int = 0
        per_pattern: dict[str, list] = {}

        total = len(dataset)
        for i, tx in enumerate(dataset):
            if show_progress and i % max(1, total // 10) == 0:
                print(f"  [{i}/{total}]")

            t0 = time.time()
            try:
                report = orchestrator.run(
                    features=tx.features,
                    request_type="analyze",
                    threshold=self._threshold,
                )
                prob = float(report.get("fraud_probability", 0.0))
                if report.get("partial_report"):
                    partials += 1
            except Exception:
                prob = 0.0
                partials += 1

            elapsed = (time.time() - t0) * 1000
            scores.append(prob)
            labels.append(tx.label)
            latencies.append(elapsed)

            p = tx.pattern
            per_pattern.setdefault(p, [])
            per_pattern[p].append((prob, tx.label))

        return self._compute_metrics(
            run_id=run_id,
            scores=scores,
            labels=labels,
            latencies=latencies,
            partials=partials,
            per_pattern=per_pattern,
        )

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #

    def _compute_metrics(
        self,
        run_id: str,
        scores: list[float],
        labels: list[int],
        latencies: list[float],
        partials: int,
        per_pattern: dict[str, list],
    ) -> EvaluationResult:
        import numpy as np

        y_score = np.array(scores)
        y_true  = np.array(labels)
        n       = len(y_true)
        n_fraud = int(y_true.sum())

        # AUC
        auc = _roc_auc(y_true, y_score)
        # PR-AUC
        pr_auc = _pr_auc(y_true, y_score)

        # Metrics at threshold
        y_pred = (y_score >= self._threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        fpr       = fp / max(fp + tn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)

        # Recall@K%
        sorted_idx = np.argsort(y_score)[::-1]
        r_at = {}
        for k_pct in self._k_pcts:
            k = max(1, int(n * k_pct))
            top_k_labels = y_true[sorted_idx[:k]]
            r_at[k_pct] = float(top_k_labels.sum()) / max(n_fraud, 1)

        # Latency
        lats = np.array(latencies)
        lat_mean = float(lats.mean())
        lat_p95  = float(np.percentile(lats, 95))

        # Per-pattern metrics
        pp: dict[str, dict] = {}
        for pat, pairs in per_pattern.items():
            sc  = np.array([p for p, _ in pairs])
            lb  = np.array([l for _, l in pairs])
            pred = (sc >= self._threshold).astype(int)
            tp_p = int(((pred == 1) & (lb == 1)).sum())
            total_fraud_p = int(lb.sum())
            total_p = len(lb)
            pp[pat] = {
                "count":       total_p,
                "fraud_count": total_fraud_p,
                "recall":      round(tp_p / max(total_fraud_p, 1), 4),
                "mean_score":  round(float(sc.mean()), 4),
            }

        return EvaluationResult(
            run_id=run_id,
            dataset_size=n,
            fraud_count=n_fraud,
            threshold=self._threshold,
            auc=round(auc, 4),
            pr_auc=round(pr_auc, 4),
            recall_at_5pct=round(r_at.get(0.05, 0.0), 4),
            recall_at_10pct=round(r_at.get(0.10, 0.0), 4),
            false_positive_rate=round(fpr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            detection_latency_ms_mean=round(lat_mean, 2),
            detection_latency_ms_p95=round(lat_p95, 2),
            partial_report_rate=round(partials / max(n, 1), 4),
            per_pattern=pp,
            raw_scores=scores,
            raw_labels=labels,
        )


# ---------------------------------------------------------------------------
# Pure-Python metric helpers (no sklearn dependency)
# ---------------------------------------------------------------------------

def _roc_auc(y_true, y_score) -> float:
    import numpy as np
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.5
    sorted_idx = np.argsort(y_score)[::-1]
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for i in sorted_idx:
        if y_true[i] == 1:
            tp += 1
        else:
            auc += tp * (fp - prev_fp + 1)
            prev_fp = fp
            fp += 1
    auc += tp * (fp - prev_fp)
    return auc / (pos.sum() * neg.sum())


def _pr_auc(y_true, y_score) -> float:
    import numpy as np
    sorted_idx = np.argsort(y_score)[::-1]
    tp = fp = 0
    precisions = []
    recalls    = []
    n_pos = (y_true == 1).sum()
    if n_pos == 0:
        return 0.0
    for i in sorted_idx:
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    # Trapezoidal integration
    return float(np.trapz(precisions, recalls)) if len(recalls) > 1 else 0.0
