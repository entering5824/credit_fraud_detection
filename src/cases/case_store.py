"""
Case Store — persistent storage abstraction for fraud investigation cases.

Production deployments should swap InMemoryCaseStore for a Postgres/Redis
implementation by calling set_case_store() at startup.

Case lifecycle:
    open → under_review → confirmed_fraud | false_positive | dismissed
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------

VALID_STATUSES = {"open", "under_review", "confirmed_fraud", "false_positive", "dismissed"}


@dataclass
class FraudCase:
    """A single fraud investigation case."""

    case_id:         str
    transaction_id:  Optional[str]
    risk_score:      float
    fraud_pattern:   str
    fraud_probability: float
    risk_level:      str
    recommended_action: str
    agent_report:    dict            # full Investigation Risk Report
    analyst_notes:   str = ""
    status:          str = "open"
    created_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional enrichment
    session_id:      Optional[str] = None
    assigned_to:     Optional[str] = None
    tags:            list[str]     = field(default_factory=list)

    def update_status(self, new_status: str, note: str = "") -> None:
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}")
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if note:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            self.analyst_notes += f"\n[{ts}] {note}"

    def to_dict(self) -> dict:
        return {
            "case_id":           self.case_id,
            "transaction_id":    self.transaction_id,
            "status":            self.status,
            "risk_score":        self.risk_score,
            "fraud_probability": self.fraud_probability,
            "risk_level":        self.risk_level,
            "fraud_pattern":     self.fraud_pattern,
            "recommended_action": self.recommended_action,
            "analyst_notes":     self.analyst_notes,
            "created_at":        self.created_at,
            "updated_at":        self.updated_at,
            "session_id":        self.session_id,
            "assigned_to":       self.assigned_to,
            "tags":              self.tags,
            "agent_report":      self.agent_report,
        }


# ---------------------------------------------------------------------------
# Abstract store
# ---------------------------------------------------------------------------

class CaseStore(ABC):
    @abstractmethod
    def save(self, case: FraudCase) -> None: ...

    @abstractmethod
    def get(self, case_id: str) -> Optional[FraudCase]: ...

    @abstractmethod
    def list_cases(
        self,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 50,
    ) -> list[FraudCase]: ...

    @abstractmethod
    def delete(self, case_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# In-memory implementation (LRU-capped)
# ---------------------------------------------------------------------------

class InMemoryCaseStore(CaseStore):
    """In-memory case store, LRU-capped at *max_cases*."""

    def __init__(self, max_cases: int = 5000) -> None:
        self._store: OrderedDict[str, FraudCase] = OrderedDict()
        self._max = max_cases

    def save(self, case: FraudCase) -> None:
        if case.case_id in self._store:
            self._store.move_to_end(case.case_id)
        elif len(self._store) >= self._max:
            self._store.popitem(last=False)
        self._store[case.case_id] = case

    def get(self, case_id: str) -> Optional[FraudCase]:
        return self._store.get(case_id)

    def list_cases(
        self,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 50,
    ) -> list[FraudCase]:
        cases = list(reversed(list(self._store.values())))
        if status:
            cases = [c for c in cases if c.status == status]
        if risk_level:
            cases = [c for c in cases if c.risk_level == risk_level]
        return cases[:limit]

    def delete(self, case_id: str) -> bool:
        if case_id in self._store:
            del self._store[case_id]
            return True
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_store: CaseStore = InMemoryCaseStore()


def get_case_store() -> CaseStore:
    return _default_store


def set_case_store(store: CaseStore) -> None:
    global _default_store
    _default_store = store
