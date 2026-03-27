"""
Case Management API routes.

GET  /cases                  – List cases (filterable by status/risk_level).
GET  /cases/{case_id}        – Get a single case.
POST /cases/{case_id}/status – Update case status.
POST /cases/{case_id}/note   – Add analyst note.
POST /cases/{case_id}/assign – Assign case to an analyst.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.cases.case_manager import CaseManager
from src.cases.case_store import VALID_STATUSES
from src.learning.feedback_collector import get_feedback_collector
from src.security.audit_logger import audit

router = APIRouter(prefix="/cases", tags=["cases"])
_manager = CaseManager()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UpdateStatusRequest(BaseModel):
    status: str = Field(..., description=f"One of: {VALID_STATUSES}")
    note: str   = Field("", description="Optional analyst note attached to this transition.")


class AddNoteRequest(BaseModel):
    note: str = Field(..., min_length=1)


class AssignRequest(BaseModel):
    analyst: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
def list_cases(
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    List fraud investigation cases.

    Filters: ?status=open&risk_level=critical&limit=20
    """
    if status and status not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Valid: {VALID_STATUSES}")
    cases = _manager.list_cases(status=status, risk_level=risk_level, limit=limit)
    return [c.to_dict() for c in cases]


@router.get("/{case_id}")
def get_case(case_id: str) -> Dict[str, Any]:
    case = _manager.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    audit("VIEW_CASE", case_id=case_id)
    return case.to_dict()


@router.post("/{case_id}/status")
def update_case_status(case_id: str, req: UpdateStatusRequest) -> Dict[str, Any]:
    """Advance the case through its lifecycle."""
    case = _manager.update_status(case_id, req.status, note=req.note)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    audit("UPDATE_CASE_STATUS", case_id=case_id,
          detail=f"Status → {req.status}. Note: {req.note}")
    # Collect analyst feedback for closed cases (feeds retraining loop)
    if req.status in ("confirmed_fraud", "false_positive"):
        try:
            get_feedback_collector().record_from_case(case)
        except Exception:
            pass
    return case.to_dict()


@router.post("/{case_id}/note")
def add_case_note(case_id: str, req: AddNoteRequest) -> Dict[str, Any]:
    case = _manager.add_note(case_id, req.note)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    return case.to_dict()


@router.post("/{case_id}/assign")
def assign_case(case_id: str, req: AssignRequest) -> Dict[str, Any]:
    case = _manager.assign(case_id, req.analyst)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    return case.to_dict()
