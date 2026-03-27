"""Multi-agent architecture — specialized agents + supervisor coordinator."""

from src.agents.multi.specialized_agents import (
    AgentContext, AgentResult, SpecializedAgent,
    FraudScoringAgent, BehaviorAnalysisAgent,
    GraphInvestigationAgent, CaseTriageAgent,
)
from src.agents.multi.supervisor import InvestigationSupervisor

__all__ = [
    "AgentContext", "AgentResult", "SpecializedAgent",
    "FraudScoringAgent", "BehaviorAnalysisAgent",
    "GraphInvestigationAgent", "CaseTriageAgent",
    "InvestigationSupervisor",
]
