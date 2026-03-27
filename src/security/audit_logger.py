"""
Audit Logger — immutable append-only audit trail for analyst actions.

Every action that touches a fraud case is recorded:
  • case status changes
  • analyst notes
  • session access
  • alert dispatches
  • model version promotions

Audit entry structure:
  {
    "audit_id":    "<uuid>",
    "timestamp":   "2026-03-18T10:22:00Z",
    "analyst_id":  "analyst@bank.com",
    "action":      "UPDATE_CASE_STATUS",
    "case_id":     "case_uuid",
    "session_id":  "session_uuid",
    "detail":      "Status changed from open → confirmed_fraud",
    "ip_address":  "10.0.0.1",
    "user_agent":  "Mozilla/5.0..."
  }

Storage: JSONL file at results/audit/audit_log.jsonl (append-only).
For production, swap _AuditBackend to write to Postgres or a SIEM.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.paths import get_paths

logger = logging.getLogger(__name__)

# Audit action catalogue
AUDIT_ACTIONS = {
    "VIEW_SESSION",
    "VIEW_CASE",
    "LIST_CASES",
    "UPDATE_CASE_STATUS",
    "ADD_CASE_NOTE",
    "ASSIGN_CASE",
    "DISPATCH_ALERT",
    "PROMOTE_MODEL",
    "ROLLBACK_MODEL",
    "RUN_INVESTIGATION",
    "ACCESS_REPORT",
    "DELETE_SESSION",
}


class AuditEntry:
    __slots__ = (
        "audit_id", "timestamp", "analyst_id", "action",
        "case_id", "session_id", "detail", "ip_address", "user_agent", "extra",
    )

    def __init__(
        self,
        action: str,
        analyst_id: str = "system",
        case_id: Optional[str] = None,
        session_id: Optional[str] = None,
        detail: str = "",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        self.audit_id   = str(uuid.uuid4())
        self.timestamp  = datetime.now(timezone.utc).isoformat()
        self.analyst_id = analyst_id
        self.action     = action
        self.case_id    = case_id
        self.session_id = session_id
        self.detail     = detail
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.extra      = extra or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id":   self.audit_id,
            "timestamp":  self.timestamp,
            "analyst_id": self.analyst_id,
            "action":     self.action,
            "case_id":    self.case_id,
            "session_id": self.session_id,
            "detail":     self.detail,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            **self.extra,
        }


class AuditLogger:
    """
    Appends audit entries to a JSONL file.

    Parameters
    ----------
    log_path : override for the audit log file path
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        if log_path is None:
            audit_dir = get_paths().results_dir / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            log_path = audit_dir / "audit_log.jsonl"
        self._path = log_path

    def log(
        self,
        action: str,
        analyst_id: str = "system",
        case_id: Optional[str] = None,
        session_id: Optional[str] = None,
        detail: str = "",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        **extra: Any,
    ) -> AuditEntry:
        """Append one audit entry."""
        entry = AuditEntry(
            action=action,
            analyst_id=analyst_id,
            case_id=case_id,
            session_id=session_id,
            detail=detail,
            ip_address=ip_address,
            user_agent=user_agent,
            extra=extra,
        )
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as exc:
            logger.error("Audit log write failed: %s", exc)
        return entry

    def tail(self, n: int = 50) -> list[dict]:
        """Read the last *n* audit entries."""
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            return [json.loads(line) for line in lines[-n:]]
        except (FileNotFoundError, OSError):
            return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit(
    action: str,
    analyst_id: str = "system",
    **kwargs: Any,
) -> AuditEntry:
    return get_audit_logger().log(action, analyst_id=analyst_id, **kwargs)
