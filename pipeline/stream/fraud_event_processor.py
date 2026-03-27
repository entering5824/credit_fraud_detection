"""
Fraud Event Processor — core business logic for stream processing.

Receives a raw transaction event (dict), enqueues it in the AgentLoop
for full investigation, and optionally fires an alert callback when the
result exceeds a configurable risk threshold.

This module is intentionally transport-agnostic: it is called by
kafka_consumer.py but can also be called directly from any other source
(HTTP webhook, file watcher, test harness).

Alert callback contract
-----------------------
The callback receives a dict with at least:
    task_id              str
    transaction_id       str
    fraud_probability    float
    risk_level           str
    recommended_action   str
    fraud_pattern        str
    escalated            bool
    case_id              str | None
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ALERT_THRESHOLD = 0.85
_DEFAULT_INVESTIGATION_TIMEOUT = 15.0


class FraudEventProcessor:
    """
    Processes transaction events from any streaming source.

    Parameters
    ----------
    agent_loop      : AgentLoop instance (created lazily if None).
    alert_callback  : called when fraud_probability >= alert_threshold.
    alert_threshold : default 0.85.
    request_type    : "investigate" | "analyze" — controls depth of analysis.
    investigation_timeout_seconds : how long to wait for agent result.
    """

    def __init__(
        self,
        agent_loop=None,
        alert_callback: Optional[Callable[[dict], None]] = None,
        alert_threshold: float = _DEFAULT_ALERT_THRESHOLD,
        request_type: str = "investigate",
        investigation_timeout_seconds: float = _DEFAULT_INVESTIGATION_TIMEOUT,
    ) -> None:
        self._loop = agent_loop
        self._alert_callback = alert_callback or _log_alert
        self._alert_threshold = alert_threshold
        self._request_type = request_type
        self._timeout = investigation_timeout_seconds

        self.stats = {
            "events_received": 0,
            "events_processed": 0,
            "alerts_fired": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def process(self, event: dict[str, Any]) -> Optional[dict]:
        """
        Process a single transaction event.

        Returns the investigation report or None on error.
        The call blocks until the agent produces a result (up to *timeout* s).
        """
        self.stats["events_received"] += 1

        try:
            features, tx_id = _extract_features(event)
        except (KeyError, ValueError) as exc:
            logger.warning("Malformed event — skipping: %s", exc)
            self.stats["errors"] += 1
            return None

        loop = self._get_loop()
        task_id = loop.submit_task(
            features=features,
            transaction_id=tx_id,
            request_type=self._request_type,
            include_behavior=True,
            include_explanation=True,
        )

        report = loop.get_result(task_id, timeout=self._timeout)
        if report is None:
            logger.warning("Investigation timed out for tx=%s task=%s", tx_id, task_id)
            self.stats["errors"] += 1
            return None

        self.stats["events_processed"] += 1
        self._maybe_alert(report)
        return report

    def process_batch(self, events: list[dict[str, Any]]) -> list[dict]:
        """Process multiple events, returning reports (None entries filtered out)."""
        results = []
        for event in events:
            r = self.process(event)
            if r is not None:
                results.append(r)
        return results

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get_loop(self):
        if self._loop is None:
            from src.agent_runtime.agent_loop import AgentLoop
            self._loop = AgentLoop(workers=2)
            self._loop.start()
        return self._loop

    def _maybe_alert(self, report: dict) -> None:
        prob = float(report.get("fraud_probability", 0.0))
        if prob >= self._alert_threshold or report.get("escalated", False):
            self.stats["alerts_fired"] += 1
            try:
                self._alert_callback(report)
            except Exception as exc:
                logger.error("Alert callback raised: %s", exc)
            # Dispatch to configured alert channels (Slack/PD/Email/SIEM)
            try:
                from src.alerts.alert_dispatcher import get_dispatcher
                get_dispatcher().dispatch(report)
            except Exception as exc:
                logger.error("AlertDispatcher raised: %s", exc)


def _log_alert(report: dict) -> None:
    """Default alert: write a structured WARNING log."""
    logger.warning(
        "FRAUD ALERT | tx=%s prob=%.3f level=%s pattern=%s action=%s case=%s",
        report.get("transaction_id"),
        report.get("fraud_probability", 0),
        report.get("risk_level", "?"),
        report.get("fraud_pattern", "?"),
        report.get("recommended_action", "?"),
        report.get("case_id", "none"),
    )


# ---------------------------------------------------------------------------
# Helper: extract normalised feature dict from raw event schema
# ---------------------------------------------------------------------------

def _extract_features(event: dict) -> tuple[dict, Optional[str]]:
    """
    Normalise a raw Kafka/webhook event to (features, transaction_id).

    Supports two common event schemas:

    Schema A (flat):
        {"transaction_id": "...", "Amount": 120.0, "V1": -1.2, ...}

    Schema B (nested):
        {"transaction_id": "...", "features": {"Amount": 120.0, "V1": -1.2, ...}}
    """
    tx_id = event.get("transaction_id") or event.get("tx_id")

    if "features" in event and isinstance(event["features"], dict):
        return event["features"], tx_id

    # Flat schema: everything except known metadata keys is a feature
    _META_KEYS = {"transaction_id", "tx_id", "user_id", "merchant_id",
                  "timestamp", "label", "fraud", "source"}
    features = {k: v for k, v in event.items() if k not in _META_KEYS}
    if not features:
        raise ValueError("No feature keys found in event")
    return features, tx_id
