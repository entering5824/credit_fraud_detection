"""
Agent Executor — runs an InvestigationTask through the full orchestrator pipeline.

The executor is stateless: it accepts a task, delegates to AgentOrchestrator,
and returns the final investigation report.  It is the bridge between the
runtime layer (task queue, agent loop) and the core agent logic.

Design decisions
----------------
* Stateless: one orchestrator per execution → no shared mutable state.
* Resilient: exceptions from orchestrator produce a structured error report
  rather than crashing the loop.
* Observable: emits Prometheus metrics and OTel spans per execution.
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Optional

from src.agent_runtime.task_queue import InvestigationTask
from src.agents.agent_orchestrator import AgentOrchestrator
from src.monitoring.metrics import record_request
from src.monitoring.tracing import start_span


class AgentExecutor:
    """
    Executes a single InvestigationTask using AgentOrchestrator.

    Parameters
    ----------
    tools_registry : optional override – useful for testing
    """

    def __init__(self, tools_registry: dict | None = None) -> None:
        self._tools_registry = tools_registry

    def execute(self, task: InvestigationTask) -> dict[str, Any]:
        """
        Run the investigation pipeline for *task*.

        Returns the normalised Transaction Risk Report produced by
        AgentOrchestrator.build_report().  Never raises — failures are
        captured in the returned dict.
        """
        start = time.time()
        record_request(endpoint="runtime", request_type=task.request_type)

        with start_span(
            "executor.run",
            attributes={
                "task_id":        task.task_id,
                "request_type":   task.request_type,
                "transaction_id": str(task.transaction_id),
                "priority":       str(task.priority),
            },
        ):
            try:
                orchestrator = AgentOrchestrator(tools_registry=self._tools_registry)

                opts = task.options or {}
                report = orchestrator.run(
                    features=task.features,
                    request_type=task.request_type,
                    transaction_id=task.task_id if task.transaction_id is None else task.transaction_id,
                    include_behavior=opts.get("include_behavior", True),
                    include_explanation=opts.get("include_explanation", True),
                    include_drift=opts.get("include_drift", False),
                    include_history=opts.get("include_history", False),
                    user_id=opts.get("user_id"),
                    threshold=opts.get("threshold"),
                    top_k=opts.get("top_k", 5),
                )
                report["task_id"] = task.task_id

            except Exception as exc:
                elapsed = round((time.time() - start) * 1000, 1)
                report = {
                    "task_id": task.task_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "partial_report": True,
                    "execution_time_ms": elapsed,
                }

        return report
