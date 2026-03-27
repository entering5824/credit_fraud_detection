"""
Agent API routes.

POST /agent/analyze      – Lightweight: score only (no SHAP / behavior).
POST /agent/investigate  – Full investigation: scoring + explanation + behavior + optional drift.
POST /agent/explain      – Explain why a transaction was flagged (scoring + SHAP + behavior).
GET  /agent/session/{id} – Retrieve a stored investigation session.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.fraud_agent import run_investigation
from src.agents.multi.supervisor import InvestigationSupervisor
from src.memory.investigation_memory import get_session

router = APIRouter(prefix="/agent", tags=["agent"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AgentAnalyzeRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Look up features from feature store.")
    features: Optional[Dict[str, Any]] = Field(None, description="Transaction features (V1..V28, Amount).")
    threshold: Optional[float] = Field(None, description="Override decision threshold.")
    session_id: Optional[str] = Field(None, description="Persist result in this session.")


class AgentInvestigateRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Look up features from feature store.")
    features: Optional[Dict[str, Any]] = Field(None, description="Transaction features.")
    include_behavior: bool = Field(True, description="Run behavior analysis.")
    include_explanation: bool = Field(True, description="Include SHAP explanation.")
    include_drift: bool = Field(False, description="Run drift check (requires baseline CSV).")
    include_history: bool = Field(False, description="Include transaction history (requires user_id).")
    user_id: Optional[str] = Field(None, description="User/synthetic_user_id for history.")
    threshold: Optional[float] = Field(None, description="Override decision threshold.")
    top_k: int = Field(5, ge=1, le=20, description="Top SHAP features.")
    session_id: Optional[str] = Field(None, description="Persist result in this session.")


class AgentExplainRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Look up features from feature store.")
    features: Optional[Dict[str, Any]] = Field(None, description="Transaction features.")
    threshold: Optional[float] = Field(None, description="Override decision threshold.")
    top_k: int = Field(5, ge=1, le=20, description="Top SHAP features.")
    session_id: Optional[str] = Field(None, description="Persist result in this session.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_input(req_id: Optional[str], feats: Optional[dict]) -> None:
    if req_id is None and feats is None:
        raise HTTPException(status_code=400, detail="Provide transaction_id or features")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze")
def agent_analyze(req: AgentAnalyzeRequest) -> Dict[str, Any]:
    """
    Lightweight scoring only – no SHAP or behavior analysis.
    Returns risk score, risk level, and recommended action from ML model alone.
    """
    _require_input(req.transaction_id, req.features)
    return run_investigation(
        transaction_id=req.transaction_id,
        features=req.features,
        request_type="analyze",
        include_behavior=False,
        include_explanation=False,
        include_drift=False,
        include_history=False,
        threshold=req.threshold,
        session_id=req.session_id,
    )


@router.post("/investigate")
def agent_investigate(req: AgentInvestigateRequest) -> Dict[str, Any]:
    """
    Full fraud investigation: risk score, key signals, model explanation, recommended action.
    """
    _require_input(req.transaction_id, req.features)
    return run_investigation(
        transaction_id=req.transaction_id,
        features=req.features,
        request_type="investigate",
        include_behavior=req.include_behavior,
        include_explanation=req.include_explanation,
        include_drift=req.include_drift,
        include_history=req.include_history,
        user_id=req.user_id,
        threshold=req.threshold,
        top_k=req.top_k,
        session_id=req.session_id,
    )


@router.post("/explain")
def agent_explain(req: AgentExplainRequest) -> Dict[str, Any]:
    """
    Explain why a transaction was flagged: risk score, suspicious signals,
    model explanation (SHAP), and behavioral anomalies.
    """
    _require_input(req.transaction_id, req.features)
    return run_investigation(
        transaction_id=req.transaction_id,
        features=req.features,
        request_type="explain",
        include_behavior=True,
        include_explanation=True,
        include_drift=False,
        include_history=False,
        threshold=req.threshold,
        top_k=req.top_k,
        session_id=req.session_id,
    )


class MultiInvestigateRequest(BaseModel):
    transaction_id: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    merchant_id: Optional[str] = None
    threshold: Optional[float] = None
    mode: str = Field("sequential", description="sequential | parallel")


@router.post("/multi-investigate")
def agent_multi_investigate(req: MultiInvestigateRequest) -> Dict[str, Any]:
    """
    Full multi-agent investigation: Scoring → Behavior → Graph → Triage.
    Runs specialized agents in sequential or parallel mode.
    """
    _require_input(req.transaction_id, req.features)
    features = req.features or {}
    supervisor = InvestigationSupervisor(mode=req.mode)
    return supervisor.investigate(
        features=features,
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        merchant_id=req.merchant_id,
        threshold=req.threshold or 0.5,
    )


@router.get("/session/{session_id}")
def agent_get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve a stored investigation session by ID."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.to_dict()
