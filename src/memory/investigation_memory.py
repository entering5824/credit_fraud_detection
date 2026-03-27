"""
Investigation session memory.

Stores full InvestigationSession objects (not just reports) so multi-step
conversations and audit trails are possible.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class InvestigationSession:
    """Complete record of one investigation session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    transactions_investigated: list[dict[str, Any]] = field(default_factory=list)
    agent_decisions: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    final_reports: list[dict[str, Any]] = field(default_factory=list)

    def add_investigation(
        self,
        features: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        agent_decisions: list[str],
        final_report: dict[str, Any],
    ) -> None:
        self.transactions_investigated.append(dict(features))
        self.tool_results.extend(tool_calls)
        self.agent_decisions.extend(agent_decisions)
        self.final_reports.append(dict(final_report))

    def latest_report(self) -> Optional[dict[str, Any]]:
        return self.final_reports[-1] if self.final_reports else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "transaction_count": len(self.transactions_investigated),
            "agent_decisions": self.agent_decisions,
            "tool_results": self.tool_results,
            "final_reports": self.final_reports,
        }


# ---------------------------------------------------------------------------
# In-memory session store (LRU-capped)
# ---------------------------------------------------------------------------

_MAX_SESSIONS = 500
_sessions: OrderedDict[str, InvestigationSession] = OrderedDict()


def create_session() -> InvestigationSession:
    session = InvestigationSession()
    _evict_if_full()
    _sessions[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[InvestigationSession]:
    return _sessions.get(session_id)


def store_investigation(
    session_id: str,
    report: dict[str, Any],
    features: Optional[dict[str, Any]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    agent_decisions: Optional[list[str]] = None,
) -> None:
    """Append an investigation result to the session (create session if missing)."""
    if session_id not in _sessions:
        session = InvestigationSession(session_id=session_id)
        _evict_if_full()
        _sessions[session_id] = session
    session = _sessions[session_id]
    session.add_investigation(
        features=features or {},
        tool_calls=tool_calls or [],
        agent_decisions=agent_decisions or [],
        final_report=report,
    )


def get_last_investigation(session_id: str) -> Optional[dict[str, Any]]:
    session = _sessions.get(session_id)
    return session.latest_report() if session else None


def clear_investigation_memory(session_id: Optional[str] = None) -> None:
    if session_id is None:
        _sessions.clear()
    elif session_id in _sessions:
        del _sessions[session_id]


def _evict_if_full() -> None:
    while len(_sessions) >= _MAX_SESSIONS:
        _sessions.popitem(last=False)
