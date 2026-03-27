"""
Feedback Collector — closes the autonomous improvement loop.

Flow:
    Analyst reviews case
        │
        ▼
    POST /cases/{id}/status  →  confirmed_fraud | false_positive
        │
        ▼
    FeedbackCollector.record()
        │
        ▼
    feedback dataset (JSONL)
        │
        ▼
    RetrainTrigger (evaluates if retraining is warranted)
        │
        ▼
    model retraining (pluggable — Jupyter / training script / MLflow run)

Feedback entry structure:
  {
    "feedback_id":     "<uuid>",
    "timestamp":       "2026-03-18T10:30:00Z",
    "case_id":         "...",
    "transaction_id":  "...",
    "analyst_id":      "...",
    "analyst_label":   1,          # 1=fraud, 0=legitimate
    "model_score":     0.87,       # what the model predicted
    "model_version":   "1.2.0",
    "fraud_pattern":   "account_takeover",
    "features_hash":   "abcdef...",
    "was_correct":     True,       # model agreed with analyst
    "feedback_type":   "confirmed_fraud" | "false_positive" | "false_negative"
  }

Retraining trigger heuristics:
  • false_positive_rate  > 0.20 in last 500 samples  → trigger
  • false_negative_rate  > 0.05 in last 500 samples  → trigger
  • min_samples reached  (default 200)                → trigger
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.core.paths import get_paths

logger = logging.getLogger(__name__)

# Thresholds for auto-retraining
_FP_RATE_THRESHOLD  = 0.20
_FN_RATE_THRESHOLD  = 0.05
_MIN_SAMPLES        = 200
_WINDOW_SIZE        = 500


class FeedbackEntry:
    def __init__(
        self,
        case_id: str,
        transaction_id: Optional[str],
        analyst_id: str,
        analyst_label: int,        # 1=fraud, 0=legitimate
        model_score: float,
        model_threshold: float,
        model_version: Optional[str] = None,
        fraud_pattern: Optional[str] = None,
        features_hash: Optional[str] = None,
        features: Optional[dict] = None,
    ) -> None:
        self.feedback_id    = str(uuid.uuid4())
        self.timestamp      = datetime.now(timezone.utc).isoformat()
        self.case_id        = case_id
        self.transaction_id = transaction_id
        self.analyst_id     = analyst_id
        self.analyst_label  = analyst_label
        self.model_score    = model_score
        self.model_threshold = model_threshold
        self.model_version  = model_version
        self.fraud_pattern  = fraud_pattern
        self.features_hash  = features_hash
        self.features       = features or {}

        model_pred = int(model_score >= model_threshold)
        self.was_correct   = model_pred == analyst_label
        if analyst_label == 1 and model_pred == 0:
            self.feedback_type = "false_negative"
        elif analyst_label == 0 and model_pred == 1:
            self.feedback_type = "false_positive"
        elif analyst_label == 1 and model_pred == 1:
            self.feedback_type = "confirmed_fraud"
        else:
            self.feedback_type = "confirmed_legit"

    def to_dict(self) -> dict:
        return {
            "feedback_id":    self.feedback_id,
            "timestamp":      self.timestamp,
            "case_id":        self.case_id,
            "transaction_id": self.transaction_id,
            "analyst_id":     self.analyst_id,
            "analyst_label":  self.analyst_label,
            "model_score":    self.model_score,
            "model_threshold": self.model_threshold,
            "model_version":  self.model_version,
            "fraud_pattern":  self.fraud_pattern,
            "features_hash":  self.features_hash,
            "was_correct":    self.was_correct,
            "feedback_type":  self.feedback_type,
        }


class FeedbackCollector:
    """
    Records analyst feedback and triggers retraining when quality degrades.

    Parameters
    ----------
    feedback_path   : JSONL path for persisting feedback
    retrain_callback: called when retraining is warranted
    fp_threshold    : false-positive rate that triggers retraining
    fn_threshold    : false-negative rate that triggers retraining
    min_samples     : minimum feedback entries before triggering
    window_size     : sliding window size for rate computation
    """

    def __init__(
        self,
        feedback_path: Optional[Path] = None,
        retrain_callback: Optional[Callable[[dict], None]] = None,
        fp_threshold: float = _FP_RATE_THRESHOLD,
        fn_threshold: float = _FN_RATE_THRESHOLD,
        min_samples: int = _MIN_SAMPLES,
        window_size: int = _WINDOW_SIZE,
    ) -> None:
        if feedback_path is None:
            fb_dir = get_paths().results_dir / "feedback"
            fb_dir.mkdir(parents=True, exist_ok=True)
            feedback_path = fb_dir / "feedback.jsonl"
        self._path        = feedback_path
        self._retrain_cb  = retrain_callback or _default_retrain_callback
        self._fp_thresh   = fp_threshold
        self._fn_thresh   = fn_threshold
        self._min_samples = min_samples
        self._window      = window_size
        self._buffer: list[FeedbackEntry] = []

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def record(
        self,
        case_id: str,
        analyst_label: int,
        model_score: float,
        model_threshold: float = 0.5,
        transaction_id: Optional[str] = None,
        analyst_id: str = "system",
        model_version: Optional[str] = None,
        fraud_pattern: Optional[str] = None,
        features_hash: Optional[str] = None,
        features: Optional[dict] = None,
    ) -> FeedbackEntry:
        """
        Record one analyst feedback entry and check retraining trigger.
        """
        entry = FeedbackEntry(
            case_id=case_id,
            transaction_id=transaction_id,
            analyst_id=analyst_id,
            analyst_label=analyst_label,
            model_score=model_score,
            model_threshold=model_threshold,
            model_version=model_version,
            fraud_pattern=fraud_pattern,
            features_hash=features_hash,
            features=features,
        )
        self._buffer.append(entry)
        # Persist
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as exc:
            logger.error("Feedback write failed: %s", exc)

        # Check retraining trigger
        self._check_retrain()
        return entry

    def record_from_case(self, case, analyst_id: str = "system") -> Optional[FeedbackEntry]:
        """
        Derive analyst_label from a closed FraudCase and record feedback.

        case.status must be 'confirmed_fraud' or 'false_positive'.
        """
        label_map = {"confirmed_fraud": 1, "false_positive": 0}
        label = label_map.get(case.status)
        if label is None:
            return None

        report = case.agent_report or {}
        return self.record(
            case_id=case.case_id,
            analyst_label=label,
            model_score=case.fraud_probability,
            transaction_id=case.transaction_id,
            analyst_id=analyst_id,
            model_version=report.get("model_version"),
            fraud_pattern=case.fraud_pattern,
            features_hash=report.get("_meta", {}).get("feature_vector_hash_full"),
        )

    def stats(self) -> dict:
        window = self._buffer[-self._window:]
        if not window:
            return {"total": 0}
        fp_rate = sum(1 for e in window if e.feedback_type == "false_positive") / len(window)
        fn_rate = sum(1 for e in window if e.feedback_type == "false_negative") / len(window)
        acc     = sum(1 for e in window if e.was_correct) / len(window)
        return {
            "total":             len(self._buffer),
            "window_size":       len(window),
            "false_positive_rate": round(fp_rate, 4),
            "false_negative_rate": round(fn_rate, 4),
            "accuracy":          round(acc, 4),
            "retrain_triggered": self._should_retrain(window),
        }

    def load_dataset(self) -> list[dict]:
        """Load all feedback entries from the JSONL file."""
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            return [json.loads(l) for l in lines if l]
        except (FileNotFoundError, OSError):
            return []

    # ------------------------------------------------------------------ #
    # Retrain trigger
    # ------------------------------------------------------------------ #

    def _check_retrain(self) -> None:
        window = self._buffer[-self._window:]
        if len(self._buffer) < self._min_samples:
            return
        if self._should_retrain(window):
            stats = self.stats()
            logger.warning(
                "Retraining triggered: fp_rate=%.3f fn_rate=%.3f",
                stats["false_positive_rate"],
                stats["false_negative_rate"],
            )
            self._retrain_cb(stats)

    def _should_retrain(self, window: list[FeedbackEntry]) -> bool:
        if not window:
            return False
        n = len(window)
        fp_rate = sum(1 for e in window if e.feedback_type == "false_positive") / n
        fn_rate = sum(1 for e in window if e.feedback_type == "false_negative") / n
        return fp_rate > self._fp_thresh or fn_rate > self._fn_thresh


def _default_retrain_callback(stats: dict) -> None:
    logger.warning(
        "RETRAIN TRIGGER — fp_rate=%.3f fn_rate=%.3f accuracy=%.3f. "
        "Schedule a retraining run.",
        stats.get("false_positive_rate", 0),
        stats.get("false_negative_rate", 0),
        stats.get("accuracy", 0),
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    global _collector
    if _collector is None:
        _collector = FeedbackCollector()
    return _collector
