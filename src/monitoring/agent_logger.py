"""
Agent observability logger.

Logs tool calls, tool results, execution times, and final investigation decisions
to both the standard Python logger and an optional JSONL file for audit trails.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent_orchestrator import InvestigationState
    from src.tools.base import ToolResult

_logger = logging.getLogger("fraud_agent")


def _get_default_log_path() -> Optional[Path]:
    try:
        from src.core.paths import get_paths
        paths = get_paths()
        return paths.results_monitoring_dir / "agent_trace.jsonl"
    except Exception:
        return None


class AgentLogger:
    """
    Structured logger for the fraud investigation agent.

    Every tool call and every final investigation decision is appended as a
    JSON line to `log_path` (default: results/monitoring/agent_trace.jsonl).
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        if log_path is None:
            log_path = _get_default_log_path()
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_tool_call(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        result: "ToolResult",
    ) -> None:
        entry = {
            "event": "tool_call",
            "ts": time.time(),
            "tool": tool_name,
            "success": result.success,
            "execution_time_ms": round(result.execution_time_ms, 2),
            "error": result.error,
            "output_keys": list(result.data.keys()) if result.success else [],
        }
        self._emit(entry)

    def log_investigation(self, state: "InvestigationState") -> None:
        report = state.final_report
        entry = {
            "event": "investigation_complete",
            "ts": time.time(),
            "transaction_id": state.transaction_id,
            "request_type": state.request_type,
            "fraud_probability": state.score_result.get("fraud_probability"),
            "risk_level": state.score_result.get("risk_level"),
            "fraud_pattern": report.get("fraud_pattern"),
            "confidence_score": report.get("confidence_score"),
            "recommended_action": report.get("recommended_action"),
            "tool_calls": state.tool_calls,
            "agent_decisions": state.agent_decisions,
            "execution_time_ms": report.get("_meta", {}).get("execution_time_ms"),
        }
        self._emit(entry)

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {"event": event_type, "ts": time.time(), **data}
        self._emit(entry)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, entry: dict[str, Any]) -> None:
        _logger.debug(json.dumps(entry, default=str))
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                pass
