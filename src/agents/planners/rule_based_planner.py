"""Rule-based planner: deterministic tool selection without an LLM."""

from __future__ import annotations

from src.agents.planners.base_planner import BasePlanner, PlannerInput, ToolPlan


class RuleBasedPlanner(BasePlanner):
    """
    Selects tools based on explicit risk rules and request flags.

    Decision rules:
    - "score" / "analyze" requests: no follow-up tools (fast path).
    - "explain" requests: always run explanation + behavior.
    - "investigate" requests: run explanation + behavior when risk ≥ threshold.
    - Drift is opt-in; history only runs when user_id is provided and opt-in.
    """

    def plan(self, pi: PlannerInput) -> ToolPlan:
        tools: list[str] = []

        if pi.request_type in {"score", "analyze"}:
            return ToolPlan(tools=tools)

        high_risk = pi.fraud_probability >= pi.threshold
        is_explain = pi.request_type == "explain"

        if pi.include_explanation or high_risk or is_explain:
            tools.append("feature_explanation")

        if pi.include_behavior or high_risk or is_explain:
            tools.append("behavior_analysis")

        if pi.include_drift:
            tools.append("drift_monitoring")

        if pi.include_history and pi.user_id is not None:
            tools.append("transaction_history")

        return ToolPlan(tools=tools)
