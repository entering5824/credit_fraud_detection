"""Alert Dispatcher — multi-channel fraud alert distribution (Slack/PagerDuty/Email/SIEM)."""

from src.alerts.alert_dispatcher import AlertDispatcher, get_dispatcher

__all__ = ["AlertDispatcher", "get_dispatcher"]
