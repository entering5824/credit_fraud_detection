"""
Investigation Supervisor Agent — coordinates the specialized agent team.

Architecture
------------
                     InvestigationSupervisor
                            │
              ┌─────────────┼──────────────┐
              │             │              │
    FraudScoringAgent  BehaviorAgent  GraphAgent
              │             │              │
              └─────────────┼──────────────┘
                            │
                     CaseTriageAgent
                            │
                  Final Investigation Report

Execution modes
---------------
SEQUENTIAL   Agents run one after another (simpler, lower overhead).
PARALLEL     Behavior + Graph agents run concurrently after scoring (faster).

The supervisor decides whether to run secondary agents based on the
fraud_probability returned by FraudScoringAgent:
  • prob < 0.25  → skip secondary agents (fast approval path)
  • prob >= 0.25 → run all secondary agents + CaseTriageAgent

Usage
-----
    from src.agents.multi.supervisor import InvestigationSupervisor

    supervisor = InvestigationSupervisor(mode="parallel")
    report = supervisor.investigate(
        features={"Amount": 5000.0, "V1": -3.2, ...},
        user_id="user_42",
        transaction_id="tx_001",
    )
    print(report["triage"]["recommended_action"])
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Optional

from src.agents.multi.specialized_agents import (
    AgentContext,
    AgentResult,
    BehaviorAnalysisAgent,
    CaseTriageAgent,
    FraudScoringAgent,
    GraphInvestigationAgent,
)

_SKIP_SECONDARY_BELOW = 0.25   # skip behavior+graph if prob < this


class InvestigationSupervisor:
    """
    Coordinates all specialized agents for a multi-perspective fraud investigation.

    Parameters
    ----------
    mode            : "sequential" | "parallel"
    tools_registry  : inject fake tools (for testing)
    skip_threshold  : skip secondary agents if prob < this value
    """

    def __init__(
        self,
        mode: str = "sequential",
        tools_registry: Optional[dict] = None,
        skip_threshold: float = _SKIP_SECONDARY_BELOW,
    ) -> None:
        self._mode   = mode
        self._tools  = tools_registry
        self._skip_t = skip_threshold

        self._scoring_agent  = FraudScoringAgent()
        self._behavior_agent = BehaviorAnalysisAgent()
        self._graph_agent    = GraphInvestigationAgent()
        self._triage_agent   = CaseTriageAgent()

    def investigate(
        self,
        features: dict[str, Any],
        transaction_id: Optional[str] = None,
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Run the full multi-agent investigation pipeline.

        Returns a dict with:
          transaction_id, fraud_probability, risk_level,
          triage (from CaseTriageAgent), agent_trace
        """
        t0 = time.time()
        context = AgentContext(
            transaction_id=transaction_id,
            features=features,
            user_id=user_id,
            merchant_id=merchant_id,
            threshold=threshold,
            tools_registry=self._tools,
        )

        trace: list[dict] = []

        # Step 1: scoring (always runs)
        r = self._scoring_agent.run(context)
        trace.append(r.to_dict())

        prob = float(context.scores.get("fraud_probability", 0.0))

        # Step 2: secondary agents (conditional)
        if prob >= self._skip_t:
            if self._mode == "parallel":
                secondary_results = self._run_parallel(context)
            else:
                secondary_results = self._run_sequential(context)
            trace.extend([r.to_dict() for r in secondary_results])

        # Step 3: triage synthesis
        triage_result = self._triage_agent.run(context)
        trace.append(triage_result.to_dict())

        elapsed = round((time.time() - t0) * 1000, 2)

        return {
            "transaction_id":    transaction_id,
            "fraud_probability": prob,
            "risk_score":        round(prob * 100, 2),
            "risk_level":        context.scores.get("risk_level", "unknown"),
            "prediction":        context.scores.get("prediction", 0),
            "model_version":     context.scores.get("model_version"),
            "triage":            context.triage,
            "behavioral_signals": context.behavior.get("flags", []),
            "graph_patterns":    [s.get("pattern") for s in context.graph_signals],
            "errors":            context.errors,
            "agent_trace":       trace,
            "_meta": {
                "mode":            self._mode,
                "execution_time_ms": elapsed,
                "agents_run":      [r["agent"] for r in trace],
            },
        }

    # ------------------------------------------------------------------ #
    # Execution strategies
    # ------------------------------------------------------------------ #

    def _run_sequential(self, context: AgentContext) -> list[AgentResult]:
        results = []
        for agent in [self._behavior_agent, self._graph_agent]:
            results.append(agent.run(context))
        return results

    def _run_parallel(self, context: AgentContext) -> list[AgentResult]:
        """Run behavior and graph agents in parallel threads."""
        results: list[AgentResult] = [None, None]  # type: ignore[list-item]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_b = pool.submit(self._behavior_agent.run, context)
            future_g = pool.submit(self._graph_agent.run, context)
            results[0] = future_b.result(timeout=10)
            results[1] = future_g.result(timeout=10)
        return results
