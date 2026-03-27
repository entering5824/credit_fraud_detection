"""
Case Manager — orchestrates the creation and lifecycle of fraud cases.

Called by the agent API after every high-risk investigation to produce
a FraudCase record that analysts can review, annotate, and close.

Auto-open policy
----------------
A case is automatically created when:
  • fraud_probability >= AUTO_CASE_THRESHOLD  (default 0.50)
  • OR the report contains escalated=True
  • OR the recommended_action is "flag for manual review"

Cases are deduplicated by transaction_id: if an open case already exists
for the same transaction, it is updated rather than duplicated.
"""

from __future__ import annotations

import uuid
from typing import Optional

from src.cases.case_store import FraudCase, CaseStore, get_case_store

AUTO_CASE_THRESHOLD = 0.50   # fraud_probability threshold to auto-open a case


class CaseManager:
    """
    Creates and manages FraudCase records.

    Parameters
    ----------
    store : CaseStore implementation (default: module-level singleton)
    threshold : minimum fraud_probability to auto-open a case
    """

    def __init__(
        self,
        store: Optional[CaseStore] = None,
        threshold: float = AUTO_CASE_THRESHOLD,
    ) -> None:
        self._store = store or get_case_store()
        self._threshold = threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def maybe_open_case(self, report: dict, session_id: Optional[str] = None) -> Optional[FraudCase]:
        """
        Inspect an investigation report and auto-open a case if warranted.
        Returns the created/updated FraudCase, or None if no case was warranted.
        """
        prob     = float(report.get("fraud_probability", 0.0))
        action   = report.get("recommended_action", "")
        escalated = report.get("escalated", False)

        should_open = (
            prob >= self._threshold
            or action == "flag for manual review"
            or escalated
        )
        if not should_open:
            return None

        tx_id = report.get("transaction_id")

        # Dedup: reuse open case for same transaction
        if tx_id:
            existing = self._find_open_for_tx(tx_id)
            if existing:
                existing.agent_report = report
                existing.risk_score = float(report.get("risk_score", prob * 100))
                existing.fraud_pattern = report.get("fraud_pattern", existing.fraud_pattern)
                existing.update_status(existing.status, note="Report updated by agent.")
                self._store.save(existing)
                return existing

        case = FraudCase(
            case_id          = str(uuid.uuid4()),
            transaction_id   = tx_id,
            risk_score       = float(report.get("risk_score", prob * 100)),
            fraud_probability = prob,
            risk_level        = report.get("risk_level", "unknown"),
            fraud_pattern     = report.get("fraud_pattern", "unknown"),
            recommended_action = action,
            agent_report      = report,
            session_id        = session_id,
        )
        self._store.save(case)
        return case

    def update_status(self, case_id: str, new_status: str, note: str = "") -> Optional[FraudCase]:
        """Move a case through its lifecycle. Returns updated case or None."""
        case = self._store.get(case_id)
        if case is None:
            return None
        case.update_status(new_status, note=note)
        self._store.save(case)
        return case

    def add_note(self, case_id: str, note: str) -> Optional[FraudCase]:
        case = self._store.get(case_id)
        if case is None:
            return None
        case.analyst_notes += f"\n{note}"
        self._store.save(case)
        return case

    def assign(self, case_id: str, analyst: str) -> Optional[FraudCase]:
        case = self._store.get(case_id)
        if case is None:
            return None
        case.assigned_to = analyst
        self._store.save(case)
        return case

    def get_case(self, case_id: str) -> Optional[FraudCase]:
        return self._store.get(case_id)

    def list_cases(
        self,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 50,
    ) -> list[FraudCase]:
        return self._store.list_cases(status=status, risk_level=risk_level, limit=limit)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _find_open_for_tx(self, transaction_id: str) -> Optional[FraudCase]:
        open_cases = self._store.list_cases(status="open", limit=500)
        for c in open_cases:
            if c.transaction_id == transaction_id:
                return c
        return None
