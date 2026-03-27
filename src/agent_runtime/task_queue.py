"""
Task Queue for the Fraud Investigation Agent Runtime.

Provides a priority queue of InvestigationTask objects that the agent_loop
drains and dispatches to agent_executor.  Tasks with higher priority
(critical risk, explicit analyst escalation) are processed first.

Priority scale (lower = higher priority):
    0  CRITICAL  – fraud_probability >= 0.85
    1  HIGH      – fraud_probability >= 0.50
    2  NORMAL    – everything else
    3  BACKGROUND – batch / async jobs
"""

from __future__ import annotations

import queue
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional


class TaskPriority(IntEnum):
    CRITICAL   = 0
    HIGH       = 1
    NORMAL     = 2
    BACKGROUND = 3


@dataclass(order=True)
class InvestigationTask:
    """
    A single unit of work for the agent runtime.

    Ordering is by (priority, created_at) so tasks of the same priority
    are processed FIFO.
    """
    priority: int = field(default=TaskPriority.NORMAL)
    created_at: float = field(default_factory=time.time)

    # Non-comparable payload fields
    task_id: str         = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    transaction_id: Optional[str] = field(default=None, compare=False)
    features: dict       = field(default_factory=dict,  compare=False)
    request_type: str    = field(default="investigate", compare=False)
    session_id: Optional[str] = field(default=None, compare=False)
    callback: Optional[Callable[[dict], None]] = field(default=None, compare=False, repr=False)

    # Optional overrides forwarded to orchestrator
    options: dict        = field(default_factory=dict, compare=False)

    @classmethod
    def from_features(
        cls,
        features: dict,
        *,
        fraud_probability: float = 0.0,
        transaction_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_type: str = "investigate",
        callback: Optional[Callable[[dict], None]] = None,
        **options: Any,
    ) -> "InvestigationTask":
        if fraud_probability >= 0.85:
            priority = TaskPriority.CRITICAL
        elif fraud_probability >= 0.50:
            priority = TaskPriority.HIGH
        else:
            priority = TaskPriority.NORMAL
        return cls(
            priority=int(priority),
            features=features,
            transaction_id=transaction_id,
            session_id=session_id,
            request_type=request_type,
            callback=callback,
            options=options,
        )


class AgentTaskQueue:
    """
    Thread-safe priority queue of InvestigationTask objects.

    Uses a heap via queue.PriorityQueue so the highest-priority task
    (lowest numeric priority) is always popped first.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=maxsize)
        self._submitted: int = 0
        self._completed: int = 0

    def submit(self, task: InvestigationTask) -> str:
        """Enqueue a task. Returns task_id."""
        self._q.put_nowait(task)
        self._submitted += 1
        return task.task_id

    def get(self, block: bool = True, timeout: Optional[float] = None) -> InvestigationTask:
        """Pop the highest-priority task."""
        return self._q.get(block=block, timeout=timeout)

    def task_done(self) -> None:
        self._q.task_done()
        self._completed += 1

    def qsize(self) -> int:
        return self._q.qsize()

    @property
    def stats(self) -> dict:
        return {
            "queued":    self.qsize(),
            "submitted": self._submitted,
            "completed": self._completed,
        }
