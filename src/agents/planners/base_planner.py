"""Abstract base class for investigation planners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannerInput:
    """Everything the planner knows when it constructs the tool plan."""

    request_type: str            # "investigate" | "explain" | "score" | "analyze"
    fraud_probability: float     # result from the initial score
    threshold: float
    include_behavior: bool = True
    include_explanation: bool = True
    include_drift: bool = False
    include_history: bool = False
    user_id: Any = None


@dataclass
class ToolPlan:
    """Ordered list of tool names to execute, and their per-tool kwargs overrides."""

    tools: list[str] = field(default_factory=list)
    kwargs_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


class BasePlanner(ABC):
    """Pluggable strategy that decides which tools to call after the initial score."""

    @abstractmethod
    def plan(self, planner_input: PlannerInput) -> ToolPlan:
        """Return an ordered ToolPlan given the current investigation context."""
