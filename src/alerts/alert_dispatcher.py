"""
Alert Dispatcher — multi-channel fraud alert distribution.

Channels
--------
  Slack        POST to incoming webhook URL
  PagerDuty    POST to Events API v2
  Email        SMTP (via smtplib)
  SIEM         Generic HTTP webhook (JSON payload)
  Log          Python logger (always-on fallback)

Configuration via environment variables or direct constructor kwargs:

  SLACK_WEBHOOK_URL
  PAGERDUTY_INTEGRATION_KEY
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / ALERT_EMAIL_TO
  SIEM_WEBHOOK_URL / SIEM_WEBHOOK_TOKEN

Alert payload (standardised across all channels):
  {
    "title":          "FRAUD ALERT — critical risk detected",
    "transaction_id": "tx_abc123",
    "risk_level":     "critical",
    "fraud_probability": 0.94,
    "fraud_pattern":  "account_takeover",
    "policy_action":  "BLOCK",
    "confidence":     0.88,
    "case_id":        "case_uuid",
    "case_link":      "https://console/cases/case_uuid",
    "timestamp":      "2026-03-18T10:22:00Z",
    "model_version":  "1.2.0",
  }

Usage
-----
    from src.alerts.alert_dispatcher import AlertDispatcher

    dispatcher = AlertDispatcher()
    dispatcher.dispatch(report)          # sends to all configured channels

    # Or fire specific channels
    dispatcher.send_slack(report)
    dispatcher.send_pagerduty(report)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Optional
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

_CONSOLE_BASE_URL = os.getenv("ANALYST_CONSOLE_URL", "http://localhost:8000")


def _build_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Normalise investigation report into a compact alert payload."""
    prob = float(report.get("fraud_probability", 0))
    case_id = report.get("case_id", "")
    return {
        "title":          f"FRAUD ALERT — {report.get('risk_level', 'unknown').upper()} risk detected",
        "transaction_id": report.get("transaction_id") or report.get("task_id", "—"),
        "risk_level":     report.get("risk_level",     "unknown"),
        "fraud_probability": round(prob, 3),
        "risk_score":     report.get("risk_score",     round(prob * 100, 1)),
        "fraud_pattern":  report.get("fraud_pattern",  "unknown"),
        "policy_action":  report.get("policy_action",  "—"),
        "confidence":     report.get("confidence_score", 0),
        "case_id":        case_id,
        "case_link":      f"{_CONSOLE_BASE_URL}/cases/{case_id}" if case_id else "—",
        "model_version":  report.get("model_version",  "—"),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Channel senders
# ---------------------------------------------------------------------------

def _send_slack(payload: dict, webhook_url: str) -> None:
    p = payload
    text = (
        f"*{p['title']}*\n"
        f"> Transaction: `{p['transaction_id']}`\n"
        f"> Risk: *{p['risk_level'].upper()}* ({p['fraud_probability']:.3f})\n"
        f"> Pattern: {p['fraud_pattern']}  |  Action: *{p['policy_action']}*\n"
        f"> Confidence: {p['confidence']:.2f}  |  Model: {p['model_version']}\n"
        f"> Case: {p['case_link']}"
    )
    body = json.dumps({"text": text}).encode()
    req = urllib_request.Request(webhook_url, data=body, headers={"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 204):
            raise RuntimeError(f"Slack webhook returned {resp.status}")
    logger.debug("Slack alert sent for tx=%s", p["transaction_id"])


def _send_pagerduty(payload: dict, integration_key: str) -> None:
    p = payload
    body = {
        "routing_key":  integration_key,
        "event_action": "trigger",
        "dedup_key":    p["transaction_id"],
        "payload": {
            "summary":   p["title"],
            "severity":  "critical" if p["risk_level"] in ("critical",) else "warning",
            "timestamp": p["timestamp"],
            "custom_details": {
                "fraud_probability": p["fraud_probability"],
                "fraud_pattern":     p["fraud_pattern"],
                "policy_action":     p["policy_action"],
                "case_link":         p["case_link"],
                "model_version":     p["model_version"],
            },
        },
    }
    data = json.dumps(body).encode()
    req = urllib_request.Request(
        "https://events.pagerduty.com/v2/enqueue",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 202):
            raise RuntimeError(f"PagerDuty returned {resp.status}")
    logger.debug("PagerDuty alert sent for tx=%s", p["transaction_id"])


def _send_email(
    payload: dict,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_address: str,
) -> None:
    p = payload
    subject = p["title"]
    body = (
        f"Transaction ID: {p['transaction_id']}\n"
        f"Risk Level:     {p['risk_level'].upper()}\n"
        f"Probability:    {p['fraud_probability']:.3f}\n"
        f"Fraud Pattern:  {p['fraud_pattern']}\n"
        f"Policy Action:  {p['policy_action']}\n"
        f"Confidence:     {p['confidence']:.2f}\n"
        f"Case Link:      {p['case_link']}\n"
        f"Model Version:  {p['model_version']}\n"
        f"Timestamp:      {p['timestamp']}\n"
    )
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = to_address
    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
        s.ehlo()
        if smtp_password:
            s.starttls()
            s.login(smtp_user, smtp_password)
        s.sendmail(smtp_user, [to_address], msg.as_string())
    logger.debug("Email alert sent for tx=%s to %s", p["transaction_id"], to_address)


def _send_siem(payload: dict, webhook_url: str, token: Optional[str] = None) -> None:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps({"event": "fraud_alert", "data": payload}).encode()
    req = urllib_request.Request(webhook_url, data=body, headers=headers)
    with urllib_request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 201, 204):
            raise RuntimeError(f"SIEM webhook returned {resp.status}")
    logger.debug("SIEM alert sent for tx=%s", payload["transaction_id"])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """
    Multi-channel fraud alert dispatcher.

    All channel credentials are read from env vars by default.
    Pass explicit kwargs to override for testing.
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str]   = None,
        pagerduty_key: Optional[str]       = None,
        smtp_host: Optional[str]           = None,
        smtp_port: int                     = 587,
        smtp_user: Optional[str]           = None,
        smtp_password: Optional[str]       = None,
        email_to: Optional[str]            = None,
        siem_url: Optional[str]            = None,
        siem_token: Optional[str]          = None,
        min_risk_level: str                = "high",
    ) -> None:
        self._slack   = slack_webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self._pd_key  = pagerduty_key     or os.getenv("PAGERDUTY_INTEGRATION_KEY")
        self._smtp_h  = smtp_host         or os.getenv("SMTP_HOST")
        self._smtp_p  = smtp_port
        self._smtp_u  = smtp_user         or os.getenv("SMTP_USER", "")
        self._smtp_pw = smtp_password     or os.getenv("SMTP_PASSWORD", "")
        self._email   = email_to          or os.getenv("ALERT_EMAIL_TO")
        self._siem    = siem_url          or os.getenv("SIEM_WEBHOOK_URL")
        self._siem_tk = siem_token        or os.getenv("SIEM_WEBHOOK_TOKEN")
        self._min_rl  = min_risk_level
        self._stats   = {"dispatched": 0, "errors": 0}

    _RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def _above_threshold(self, report: dict) -> bool:
        level  = report.get("risk_level", "low")
        return self._RISK_ORDER.get(level, 0) >= self._RISK_ORDER.get(self._min_rl, 2)

    def dispatch(self, report: dict[str, Any]) -> dict[str, list]:
        """
        Dispatch alert to all configured channels.
        Returns {"sent": [...], "errors": [...]} for observability.
        """
        if not self._above_threshold(report):
            return {"sent": [], "errors": []}

        payload = _build_payload(report)
        sent, errors = [], []

        for name, fn in [
            ("slack",      self.send_slack),
            ("pagerduty",  self.send_pagerduty),
            ("email",      self.send_email),
            ("siem",       self.send_siem),
        ]:
            try:
                fn(report)
                sent.append(name)
            except Exception as exc:
                errors.append({"channel": name, "error": str(exc)})
                logger.warning("Alert channel %s failed: %s", name, exc)

        logger.info(
            "Alert dispatched for tx=%s risk=%s sent=%s errors=%d",
            payload["transaction_id"], payload["risk_level"], sent, len(errors),
        )
        self._stats["dispatched"] += 1
        if errors:
            self._stats["errors"] += len(errors)
        return {"sent": sent, "errors": errors}

    def send_slack(self, report: dict) -> None:
        if self._slack:
            _send_slack(_build_payload(report), self._slack)

    def send_pagerduty(self, report: dict) -> None:
        if self._pd_key:
            _send_pagerduty(_build_payload(report), self._pd_key)

    def send_email(self, report: dict) -> None:
        if self._smtp_h and self._email:
            _send_email(
                _build_payload(report),
                self._smtp_h, self._smtp_p,
                self._smtp_u, self._smtp_pw,
                self._email,
            )

    def send_siem(self, report: dict) -> None:
        if self._siem:
            _send_siem(_build_payload(report), self._siem, self._siem_tk)

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_dispatcher: Optional[AlertDispatcher] = None


def get_dispatcher() -> AlertDispatcher:
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = AlertDispatcher()
    return _default_dispatcher
