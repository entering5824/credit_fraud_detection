"""Observe -> Analyze -> Use Tools -> Reason -> Produce Result (stateful)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.agents.planners.base_planner import BasePlanner, PlannerInput
from src.agents.planners.rule_based_planner import RuleBasedPlanner
from src.agents.reasoning.risk_reasoner import RiskReasoner
from src.core.thresholds import load_threshold_config
from src.monitoring.agent_logger import AgentLogger
from src.monitoring.metrics import record_tool_call, record_prediction, record_partial_report, record_latency
from src.monitoring.tracing import start_span
from src.tools import TOOL_REGISTRY
from src.tools.base import ToolResult

# Confidence penalty per failed secondary tool (SHAP, behavior, drift)
_CONFIDENCE_PENALTY_PER_FAILED_TOOL = 0.10


# ---------------------------------------------------------------------------
# Investigation state
# ---------------------------------------------------------------------------

@dataclass
class InvestigationState:
    """
    Mutable record of everything that happened during one investigation run.
    Designed to be easy to persist, debug, and extend.
    """

    transaction_id: Optional[str] = None
    features: dict[str, Any] = field(default_factory=dict)
    request_type: str = "investigate"
    threshold: float = 0.5

    # Tool results keyed by tool name
    score_result: dict[str, Any] = field(default_factory=dict)
    explanation_result: Optional[dict[str, Any]] = None
    behavior_result: Optional[dict[str, Any]] = None
    drift_result: Optional[dict[str, Any]] = None
    history_result: Optional[dict[str, Any]] = None

    # Orchestration trace
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_decisions: list[str] = field(default_factory=list)

    # Degraded-mode tracking
    failed_tools: list[str] = field(default_factory=list)
    is_partial_report: bool = False

    # Final outputs
    final_report: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Report builder (uses reasoning layer)
# ---------------------------------------------------------------------------

def build_report(state: InvestigationState) -> dict[str, Any]:
    """Build the normalised Transaction Risk Report from InvestigationState."""
    reasoner = RiskReasoner()
    reasoning = reasoner.reason(
        score_result=state.score_result,
        behavior_result=state.behavior_result,
        explanation_result=state.explanation_result,
        drift_result=state.drift_result,
        threshold=state.threshold,
    )

    score = state.score_result
    prob = float(score.get("fraud_probability", 0.0))
    risk_level = score.get("risk_level", "unknown")

    # Apply confidence penalty for each failed secondary tool
    confidence = reasoning.confidence_score
    if state.failed_tools:
        penalty = len(state.failed_tools) * _CONFIDENCE_PENALTY_PER_FAILED_TOOL
        confidence = max(0.0, round(confidence - penalty, 3))

    # Keep full SHA-256 hash in the report (truncated shown separately in UI)
    feature_hash_full = score.get("feature_vector_hash", "")
    feature_hash_short = feature_hash_full[:16] if feature_hash_full else ""

    report: dict[str, Any] = {
        "transaction_id": state.transaction_id,
        "fraud_probability": prob,
        "risk_score": score.get("risk_score", round(prob * 100, 2)),
        "risk_level": risk_level,
        "prediction": score.get("prediction", 0),
        "threshold_used": score.get("threshold_used", state.threshold),
        "model_version": score.get("model_version"),
        # Short hash for UI; full hash in _meta for audit
        "feature_vector_hash": feature_hash_short,
        "fraud_pattern": reasoning.fraud_pattern,
        "confidence_score": confidence,
        "key_risk_signals": reasoning.key_risk_signals,
        "ranked_signals": reasoning.ranked_signals,
        "behavioral_anomalies": (
            state.behavior_result.get("flags", []) if state.behavior_result else []
        ),
        "model_explanation": {
            "top_features_contributing": (
                state.explanation_result.get("top_features_contributing", [])
                if state.explanation_result else []
            ),
            "narrative": (
                state.explanation_result.get("narrative", "")
                if state.explanation_result else ""
            ),
        },
        "recommended_action": reasoning.recommended_action,
        "analyst_summary": reasoning.analyst_summary,
        "partial_report": state.is_partial_report,
    }

    if state.drift_result is not None:
        report["drift_analysis"] = {
            "is_drifted": state.drift_result.get("is_drifted", False),
            "dataset_psi": state.drift_result.get("dataset_psi", 0.0),
            "fraud_rate_shift": state.drift_result.get("fraud_rate_shift", 0.0),
            "top_drifted_features": state.drift_result.get("top_drifted_features", []),
        }

    elapsed = round((time.time() - state.started_at) * 1000, 1)
    report["_meta"] = {
        "tool_calls": state.tool_calls,
        "agent_decisions": state.agent_decisions,
        "failed_tools": state.failed_tools,
        "feature_vector_hash_full": feature_hash_full,
        "execution_time_ms": elapsed,
    }
    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Runs the fraud investigation loop.

    Observe → Score → Plan → Run tools (statefully) → Reason → Report

    Partial-report policy
    ---------------------
    If a secondary tool (SHAP, behavior, drift) fails, we proceed with a
    degraded report rather than blocking the entire investigation.
    Failed tools are recorded in `_meta.failed_tools` and the confidence
    score is penalised by CONFIDENCE_PENALTY_PER_FAILED_TOOL per failure.
    `partial_report: true` is set so callers can surface a warning to analysts.
    """

    def __init__(
        self,
        tools_registry: dict[str, Any] | None = None,
        planner: BasePlanner | None = None,
    ) -> None:
        self.tools = tools_registry or TOOL_REGISTRY
        self.planner: BasePlanner = planner or RuleBasedPlanner()
        self.logger = AgentLogger()

    def _run_tool(
        self,
        state: InvestigationState,
        tool_name: str,
        **kwargs: Any,
    ) -> ToolResult:
        with start_span(f"tool.{tool_name}", attributes={"transaction_id": str(state.transaction_id)}):
            tool = self.tools.get(tool_name)
            if tool is None:
                result = ToolResult(success=False, data={}, error=f"Tool '{tool_name}' not found")
            else:
                result = tool.run(**kwargs)

        state.tool_calls.append(
            {
                "tool": tool_name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "error": result.error,
            }
        )
        if not result.success:
            state.failed_tools.append(tool_name)
            state.is_partial_report = True
            state.agent_decisions.append(
                f"tool_failed={tool_name}, proceeding with degraded report"
            )
        record_tool_call(tool_name, result.success)
        self.logger.log_tool_call(tool_name, kwargs, result)
        return result

    def run(
        self,
        features: dict[str, Any],
        request_type: str = "investigate",
        threshold: float | None = None,
        include_behavior: bool = True,
        include_explanation: bool = True,
        include_drift: bool = False,
        include_history: bool = False,
        user_id: Any = None,
        top_k: int = 5,
        transaction_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute the agent loop and return a normalised Transaction Risk Report."""
        if threshold is None:
            threshold = load_threshold_config().optimal_threshold

        state = InvestigationState(
            transaction_id=transaction_id,
            features=features,
            request_type=request_type,
            threshold=threshold,
        )
        state.agent_decisions.append(f"request_type={request_type}, threshold={threshold}")

        with start_span("agent.investigation", attributes={"request_type": request_type}):
            pass  # top-level span opened; individual steps traced via _run_tool

        # ---- 1. Observe & Score ----
        score_result = self._run_tool(state, "fraud_scoring", features=features, threshold=threshold)
        if not score_result.success:
            # Scoring is mandatory; cannot proceed without it
            return {"error": f"Fraud scoring failed: {score_result.error}", "partial_report": True}
        state.score_result = score_result.data
        prob = float(state.score_result["fraud_probability"])

        # ---- 2. Plan ----
        pi = PlannerInput(
            request_type=request_type,
            fraud_probability=prob,
            threshold=threshold,
            include_behavior=include_behavior,
            include_explanation=include_explanation,
            include_drift=include_drift,
            include_history=include_history,
            user_id=user_id,
        )
        plan = self.planner.plan(pi)
        state.agent_decisions.append(f"planned_tools={plan.tools}")

        # ---- 3. Run planned tools (graceful degradation on failure) ----
        for tool_name in plan.tools:
            extra = plan.kwargs_overrides.get(tool_name, {})
            if tool_name == "feature_explanation":
                res = self._run_tool(
                    state, tool_name,
                    features=features, top_k=top_k, threshold=threshold,
                    **extra,
                )
                if res.success:
                    state.explanation_result = res.data
            elif tool_name == "behavior_analysis":
                res = self._run_tool(state, tool_name, features=features, **extra)
                if res.success:
                    state.behavior_result = res.data
            elif tool_name == "drift_monitoring":
                res = self._run_tool(state, tool_name, features=features, **extra)
                if res.success:
                    state.drift_result = res.data
            elif tool_name == "transaction_history" and user_id is not None:
                res = self._run_tool(state, tool_name, user_id=str(user_id), **extra)
                if res.success:
                    state.history_result = res.data

        # ---- 4. Reason & Build report ----
        report = build_report(state)
        state.final_report = report
        self.logger.log_investigation(state)

        # Prometheus metrics
        record_prediction(
            fraud_probability=float(state.score_result.get("fraud_probability", 0)),
            prediction=int(state.score_result.get("prediction", 0)),
        )
        if state.is_partial_report:
            record_partial_report()
        elapsed = float(report.get("_meta", {}).get("execution_time_ms", 0))
        record_latency(request_type, elapsed)

        return report
