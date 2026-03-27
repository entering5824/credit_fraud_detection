"""
Agent Loop — the autonomous Observe → Plan → Act → Evaluate runtime.

The loop runs in a background thread and continuously drains the
AgentTaskQueue, delegates each InvestigationTask to AgentExecutor,
evaluates the result (apply escalation rules, invoke callbacks), and
can be monitored via its ``status`` property.

                     ┌─────────────────────────────┐
    submit_task()    │          AGENT LOOP          │
     ──────────────► │                              │
                     │   while running:             │
                     │     task = queue.get()       │  ← OBSERVE
                     │     plan = planner.plan()    │  ← PLAN
                     │     report = executor.run()  │  ← ACT
                     │     evaluate(report)         │  ← EVALUATE
                     │     callback(report)         │
                     └─────────────────────────────┘

Usage
-----
    from src.agent_runtime.agent_loop import AgentLoop

    loop = AgentLoop()
    loop.start()

    task_id = loop.submit_task(features={"Amount": 150.0, ...})

    # fire-and-forget — result written to callback or polled via get_result()
    result = loop.get_result(task_id, timeout=5.0)

    loop.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

from src.agent_runtime.agent_executor import AgentExecutor
from src.agent_runtime.task_queue import AgentTaskQueue, InvestigationTask, TaskPriority

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Escalation policy
# ---------------------------------------------------------------------------

def _default_escalation_policy(report: dict) -> bool:
    """Return True if the investigation should be escalated to a human analyst."""
    prob = report.get("fraud_probability", 0.0)
    pattern = report.get("fraud_pattern", "")
    partial = report.get("partial_report", False)
    return prob >= 0.90 or pattern in {"velocity_fraud", "account_takeover"} or partial


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Background-thread agent loop.

    Parameters
    ----------
    workers         Number of parallel executor threads (default 2).
    max_queue_size  Hard limit on enqueued tasks (back-pressure).
    tools_registry  Injected tool registry (for testing / swapping).
    escalation_fn   Custom escalation policy callable.
    """

    def __init__(
        self,
        workers: int = 2,
        max_queue_size: int = 1000,
        tools_registry: dict | None = None,
        escalation_fn: Optional[Callable[[dict], bool]] = None,
    ) -> None:
        self._queue   = AgentTaskQueue(maxsize=max_queue_size)
        self._executor = AgentExecutor(tools_registry=tools_registry)
        self._escalation_fn = escalation_fn or _default_escalation_policy

        self._workers = workers
        self._threads: list[threading.Thread] = []
        self._running = False

        # task_id → result (capped at 2000 entries)
        self._results: dict[str, dict] = {}
        self._results_lock = threading.Lock()
        self._MAX_RESULTS = 2000

        # per-loop statistics
        self.stats = {
            "processed": 0,
            "escalated": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return
        self._running = True
        for i in range(self._workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"agent-worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        logger.info("AgentLoop started (%d workers)", self._workers)

    def stop(self, wait: bool = True) -> None:
        """Signal workers to stop."""
        self._running = False
        # Unblock all waiting threads with sentinel tasks
        for _ in self._threads:
            try:
                self._queue._q.put_nowait(
                    InvestigationTask(priority=int(TaskPriority.CRITICAL) - 1)
                )
            except queue.Full:
                pass
        if wait:
            for t in self._threads:
                t.join(timeout=5.0)
        logger.info("AgentLoop stopped")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def submit_task(
        self,
        features: dict[str, Any],
        *,
        fraud_probability: float = 0.0,
        transaction_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_type: str = "investigate",
        callback: Optional[Callable[[dict], None]] = None,
        **options: Any,
    ) -> str:
        """
        Enqueue a new investigation task and return its task_id.
        The result can be retrieved later via get_result().
        """
        task = InvestigationTask.from_features(
            features,
            fraud_probability=fraud_probability,
            transaction_id=transaction_id,
            session_id=session_id,
            request_type=request_type,
            callback=callback,
            **options,
        )
        self._queue.submit(task)
        return task.task_id

    def get_result(
        self,
        task_id: str,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> Optional[dict[str, Any]]:
        """
        Block until the result for *task_id* is available or *timeout* elapses.
        Returns None on timeout.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._results_lock:
                if task_id in self._results:
                    return self._results[task_id]
            time.sleep(poll_interval)
        return None

    @property
    def queue_stats(self) -> dict:
        return {**self._queue.stats, "loop": self.stats}

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue

            # Sentinel check — empty task with no features
            if not task.features and not task.transaction_id:
                self._queue.task_done()
                break

            self._process(task)
            self._queue.task_done()

    def _process(self, task: InvestigationTask) -> None:
        """OBSERVE → PLAN → ACT → EVALUATE for a single task."""

        # --- OBSERVE (already done: task contains features + context)

        # --- ACT
        try:
            report = self._executor.execute(task)
        except Exception as exc:
            logger.exception("Executor raised for task %s: %s", task.task_id, exc)
            report = {
                "task_id": task.task_id,
                "error": str(exc),
                "partial_report": True,
            }
            self.stats["errors"] += 1

        # --- EVALUATE
        if self._escalation_fn(report):
            report["escalated"] = True
            self.stats["escalated"] += 1
            logger.warning(
                "Task %s escalated — prob=%.2f pattern=%s",
                task.task_id,
                report.get("fraud_probability", 0),
                report.get("fraud_pattern", "?"),
            )
        else:
            report["escalated"] = False

        self.stats["processed"] += 1

        # Store result
        with self._results_lock:
            if len(self._results) >= self._MAX_RESULTS:
                oldest = next(iter(self._results))
                del self._results[oldest]
            self._results[task.task_id] = report

        # Invoke optional callback (e.g. write to Kafka, webhook)
        if task.callback is not None:
            try:
                task.callback(report)
            except Exception as exc:
                logger.error("Task %s callback raised: %s", task.task_id, exc)
