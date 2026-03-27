"""
Entry point for the fraud investigation agent.

Handles request resolution (feature store lookup), calls the orchestrator,
persists the session, and returns the structured Transaction Risk Report.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import HTTPException

from src.agents.agent_orchestrator import AgentOrchestrator
from src.cases.case_manager import CaseManager
from src.feature_store.in_memory import get as fs_get
from src.memory.investigation_memory import store_investigation
from src.policy.risk_policy_engine import get_policy_engine
from src.schemas.feature_schema import validate_feature_dict

_case_manager  = CaseManager()
_policy_engine = get_policy_engine()


def run_investigation(
    transaction_id: Optional[str] = None,
    features: Optional[dict[str, Any]] = None,
    request_type: str = "investigate",
    include_behavior: bool = True,
    include_explanation: bool = True,
    include_drift: bool = False,
    include_history: bool = False,
    user_id: Any = None,
    threshold: Optional[float] = None,
    top_k: int = 5,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the fraud investigation agent and return a Transaction Risk Report.
    Either transaction_id or features must be provided.
    """
    # Resolve features
    if transaction_id is not None:
        resolved = fs_get(transaction_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Transaction not found in feature store")
        features = resolved
    if features is None:
        raise HTTPException(status_code=400, detail="Provide transaction_id or features")
    validate_feature_dict(features)

    if session_id is None:
        session_id = str(uuid.uuid4())

    orchestrator = AgentOrchestrator()
    report = orchestrator.run(
        features=features,
        request_type=request_type,
        threshold=threshold,
        include_behavior=include_behavior,
        include_explanation=include_explanation,
        include_drift=include_drift,
        include_history=include_history,
        user_id=user_id,
        top_k=top_k,
        transaction_id=transaction_id,
    )

    # Apply policy engine — enriches report with policy_action, policy_rule, etc.
    report = _policy_engine.enrich_report(report)

    meta = report.get("_meta", {})
    store_investigation(
        session_id=session_id,
        report=report,
        features=features,
        tool_calls=meta.get("tool_calls", []),
        agent_decisions=meta.get("agent_decisions", []),
    )
    report["session_id"] = session_id

    # Auto-open fraud case for high-risk reports
    case = _case_manager.maybe_open_case(report, session_id=session_id)
    if case is not None:
        report["case_id"] = case.case_id

    return report
