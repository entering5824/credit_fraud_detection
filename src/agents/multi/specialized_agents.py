"""
Specialized Agents — each agent focuses on one investigation dimension.

Agent roles
-----------
FraudScoringAgent       ML model scoring only (fast path).
BehaviorAnalysisAgent   User behavioural signals (velocity, spike, novelty).
GraphInvestigationAgent Graph-level multi-transaction pattern detection.
CaseTriageAgent         Synthesises all signals → final triage decision.

Each agent exposes a single method:
    run(context: AgentContext) -> AgentResult

They are stateless: all shared state flows through AgentContext.
The InvestigationSupervisor (see supervisor.py) orchestrates them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Shared context and result
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """Shared investigation context passed between specialized agents."""
    transaction_id: Optional[str] = None
    features:       dict[str, Any] = field(default_factory=dict)
    user_id:        Optional[str]  = None
    merchant_id:    Optional[str]  = None
    threshold:      float          = 0.5
    tools_registry: Optional[dict] = None

    # Accumulated results from previous agents
    scores:     dict[str, Any] = field(default_factory=dict)
    behavior:   dict[str, Any] = field(default_factory=dict)
    graph_signals: list[dict]  = field(default_factory=list)
    triage:     dict[str, Any] = field(default_factory=dict)
    errors:     list[str]      = field(default_factory=list)


@dataclass
class AgentResult:
    agent_name: str
    success:    bool
    data:       dict[str, Any]
    error:      Optional[str] = None
    latency_ms: float         = 0.0

    def to_dict(self) -> dict:
        return {
            "agent":      self.agent_name,
            "success":    self.success,
            "latency_ms": self.latency_ms,
            "error":      self.error,
            "data":       self.data,
        }


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SpecializedAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult: ...

    def _run_tool(
        self, context: AgentContext, tool_name: str, **kwargs: Any
    ) -> dict[str, Any]:
        registry = context.tools_registry
        if registry is None:
            from src.tools import TOOL_REGISTRY
            registry = TOOL_REGISTRY
        tool = registry.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' not found")
        result = tool.run(**kwargs)
        if not result.success:
            raise RuntimeError(result.error)
        return result.data


# ---------------------------------------------------------------------------
# 1. Fraud Scoring Agent
# ---------------------------------------------------------------------------

class FraudScoringAgent(SpecializedAgent):
    """
    Runs the ML fraud scoring model and populates context.scores.
    """
    name = "fraud_scoring_agent"

    def run(self, context: AgentContext) -> AgentResult:
        import time
        t0 = time.time()
        try:
            data = self._run_tool(
                context, "fraud_scoring",
                features=context.features,
                threshold=context.threshold,
            )
            context.scores = data
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=data,
                latency_ms=round((time.time() - t0) * 1000, 2),
            )
        except Exception as exc:
            context.errors.append(f"{self.name}: {exc}")
            return AgentResult(self.name, False, {}, str(exc),
                               round((time.time() - t0) * 1000, 2))


# ---------------------------------------------------------------------------
# 2. Behavior Analysis Agent
# ---------------------------------------------------------------------------

class BehaviorAnalysisAgent(SpecializedAgent):
    """
    Runs behavioral feature computation and populates context.behavior.
    """
    name = "behavior_analysis_agent"

    def run(self, context: AgentContext) -> AgentResult:
        import time
        t0 = time.time()
        try:
            data = self._run_tool(
                context, "behavior_analysis",
                features=context.features,
            )
            context.behavior = data
            return AgentResult(self.name, True, data,
                               latency_ms=round((time.time() - t0) * 1000, 2))
        except Exception as exc:
            context.errors.append(f"{self.name}: {exc}")
            return AgentResult(self.name, False, {}, str(exc),
                               round((time.time() - t0) * 1000, 2))


# ---------------------------------------------------------------------------
# 3. Graph Investigation Agent
# ---------------------------------------------------------------------------

class GraphInvestigationAgent(SpecializedAgent):
    """
    Runs graph-level fraud pattern detection and populates context.graph_signals.
    """
    name = "graph_investigation_agent"

    def run(self, context: AgentContext) -> AgentResult:
        import time
        t0 = time.time()
        user_id = context.user_id
        if not user_id:
            return AgentResult(self.name, True, {"skipped": "no user_id"})
        try:
            data = self._run_tool(
                context, "graph_analysis",
                user_id=user_id,
                amount=float(context.features.get("Amount", 0)),
                merchant_id=context.merchant_id,
            )
            context.graph_signals = data.get("patterns", [])
            return AgentResult(self.name, True, data,
                               latency_ms=round((time.time() - t0) * 1000, 2))
        except Exception as exc:
            context.errors.append(f"{self.name}: {exc}")
            return AgentResult(self.name, False, {}, str(exc),
                               round((time.time() - t0) * 1000, 2))


# ---------------------------------------------------------------------------
# 4. Case Triage Agent
# ---------------------------------------------------------------------------

class CaseTriageAgent(SpecializedAgent):
    """
    Synthesises all signals from context into a triage decision:
      risk_level, fraud_pattern, recommended_action, confidence_score,
      analyst_summary, and signals used.
    """
    name = "case_triage_agent"

    def run(self, context: AgentContext) -> AgentResult:
        import time
        t0 = time.time()
        try:
            from src.agents.reasoning.risk_reasoner import RiskReasoner
            from src.policy.risk_policy_engine import get_policy_engine

            reasoner = RiskReasoner()
            fake_report_for_reasoner = {
                "fraud_probability": context.scores.get("fraud_probability", 0.0),
                "risk_level":        context.scores.get("risk_level", "unknown"),
                "threshold_used":    context.threshold,
            }
            reasoning = reasoner.reason(
                score_result=fake_report_for_reasoner,
                behavior_result=context.behavior or None,
                threshold=context.threshold,
            )

            # Enrich with graph signals
            graph_patterns = [s.get("pattern", "") for s in context.graph_signals]
            key_signals = list(reasoning.key_risk_signals) + [
                f"graph:{p}" for p in graph_patterns[:3]
            ]

            triage = {
                "fraud_pattern":     reasoning.fraud_pattern,
                "confidence_score":  reasoning.confidence_score,
                "key_risk_signals":  key_signals,
                "recommended_action": reasoning.recommended_action,
                "analyst_summary":   reasoning.analyst_summary,
                "graph_patterns":    graph_patterns,
            }

            # Apply policy engine
            combined_report = {**context.scores, **triage}
            policy = get_policy_engine()
            combined_report = policy.enrich_report(combined_report)
            triage["policy_action"]    = combined_report.get("policy_action")
            triage["policy_triggered"] = combined_report.get("policy_triggered", False)
            triage["recommended_action"] = combined_report.get("recommended_action", reasoning.recommended_action)

            context.triage = triage
            return AgentResult(self.name, True, triage,
                               latency_ms=round((time.time() - t0) * 1000, 2))
        except Exception as exc:
            context.errors.append(f"{self.name}: {exc}")
            return AgentResult(self.name, False, {}, str(exc),
                               round((time.time() - t0) * 1000, 2))
