"""Base tool interface for the fraud investigation agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    """Structured result returned by every tool so the orchestrator can reason about failures."""

    success: bool
    data: dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def unwrap(self) -> dict[str, Any]:
        """Return data, or raise on failure."""
        if not self.success:
            raise RuntimeError(f"Tool failed: {self.error}")
        return self.data


@dataclass
class Tool:
    """
    Single-responsibility tool with name, description, schemas, timeout, and execute.

    Fields
    ------
    name            Unique tool identifier used by the registry and orchestrator.
    description     Human-readable purpose (also used as a prompt hint for LLM planners).
    input_schema    JSON-Schema dict describing accepted kwargs.
    output_schema   JSON-Schema dict describing the returned dict keys.
    execute         Callable(**kwargs) -> dict[str, Any]. Must not raise; wrap errors internally.
    timeout_seconds Max seconds the tool is allowed to run (enforced by the registry runner).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    execute: Callable[..., dict[str, Any]]
    timeout_seconds: int = 10

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult (never raises)."""
        t0 = time.perf_counter()
        try:
            data = self.execute(**kwargs)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(success=True, data=data, execution_time_ms=elapsed)
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                success=False,
                data={},
                error=str(exc),
                execution_time_ms=elapsed,
            )
