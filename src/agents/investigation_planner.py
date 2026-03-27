"""Backward-compatible shim: delegate to the pluggable planner strategy."""

from __future__ import annotations

from typing import Any

from src.agents.planners.base_planner import PlannerInput
from src.agents.planners.rule_based_planner import RuleBasedPlanner

_default_planner = RuleBasedPlanner()


def plan_tools(
    request_type: str,
    fraud_probability: float,
    threshold: float,
    include_behavior: bool = True,
    include_explanation: bool = True,
    include_drift: bool = False,
    include_history: bool = False,
    user_id: Any = None,
) -> list[str]:
    """Return ordered list of tool names (backward-compatible interface)."""
    pi = PlannerInput(
        request_type=request_type,
        fraud_probability=fraud_probability,
        threshold=threshold,
        include_behavior=include_behavior,
        include_explanation=include_explanation,
        include_drift=include_drift,
        include_history=include_history,
        user_id=user_id,
    )
    return _default_planner.plan(pi).tools
