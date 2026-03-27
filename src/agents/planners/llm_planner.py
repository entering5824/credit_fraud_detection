"""
LLM-based investigation planner.

Replaces the rule-based planner with an LLM that reads the initial risk
signal and decides which tools to invoke, in which order, and synthesises
a reasoning summary.

Compatibility
-------------
Works with any OpenAI-compatible API endpoint (OpenAI, Azure OpenAI,
Ollama, LM Studio, etc.).  Set the following environment variables:

    OPENAI_API_KEY   – required for openai.com; can be a dummy for local models
    OPENAI_BASE_URL  – override endpoint (e.g. http://localhost:11434/v1)
    OPENAI_MODEL     – model name (default: gpt-4o-mini)

If the openai package is not installed or the API call fails, the planner
falls back transparently to RuleBasedPlanner so the system stays operational.

Design
------
The LLM receives a compact JSON prompt describing:
  • fraud_probability and risk_level from the initial score
  • available tools and their descriptions
  • request_type and user preferences

It returns a JSON object with:
  • tools: list of tool names to run (ordered)
  • reasoning: one-sentence explanation of the plan

Example LLM response:
    {
      "tools": ["feature_explanation", "behavior_analysis"],
      "reasoning": "High probability (0.91) with new merchant — running SHAP + behavior to identify account-takeover signals."
    }
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from src.agents.planners.base_planner import BasePlanner, PlannerInput, ToolPlan
from src.agents.planners.rule_based_planner import RuleBasedPlanner

logger = logging.getLogger(__name__)

_AVAILABLE_TOOLS = {
    "feature_explanation": "SHAP explanation — identifies which features drive the fraud score.",
    "behavior_analysis":   "Behavioural signals — spending spike, merchant novelty, velocity.",
    "drift_monitoring":    "Data drift — PSI per feature vs. training baseline.",
    "transaction_history": "Retrieve past transactions for the user.",
    "graph_analysis":      "Graph pattern detection — velocity bursts, card-testing clusters.",
}

_SYSTEM_PROMPT = """\
You are a fraud investigation planner for an AI agent system.
Your job is to choose which analysis tools to run on a flagged transaction
based on the initial ML risk score and the analyst's preferences.

Available tools:
{tool_list}

Rules:
- Return ONLY valid JSON with the keys "tools" (list) and "reasoning" (string).
- "tools" must be a subset of the available tool names shown above.
- Do not include tools the analyst has disabled.
- Order tools from most to least important.
- Keep reasoning under 2 sentences.
"""

_USER_PROMPT = """\
Transaction risk signal:
  fraud_probability : {fraud_probability:.3f}
  risk_level        : {risk_level}
  request_type      : {request_type}

Analyst preferences:
  include_explanation : {include_explanation}
  include_behavior    : {include_behavior}
  include_drift       : {include_drift}
  include_history     : {include_history}
  user_id_provided    : {user_id_provided}

Which tools should the agent run?  Respond with JSON only.
"""


class LLMPlanner(BasePlanner):
    """
    OpenAI-compatible LLM planner with automatic rule-based fallback.

    Parameters
    ----------
    model           LLM model name (overrides OPENAI_MODEL env var).
    base_url        API base URL (overrides OPENAI_BASE_URL).
    api_key         API key (overrides OPENAI_API_KEY).
    temperature     LLM temperature (default 0 for determinism).
    timeout         HTTP timeout in seconds (default 10).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 10.0,
    ) -> None:
        self._model       = model       or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._base_url    = base_url    or os.getenv("OPENAI_BASE_URL")
        self._api_key     = api_key     or os.getenv("OPENAI_API_KEY", "EMPTY")
        self._temperature = temperature
        self._timeout     = timeout
        self._fallback    = RuleBasedPlanner()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai
            kwargs: dict = {"api_key": self._api_key, "timeout": self._timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
            return self._client
        except ImportError:
            return None

    def _call_llm(self, planner_input: PlannerInput) -> Optional[dict]:
        client = self._get_client()
        if client is None:
            return None

        # Build allowed tools list based on analyst preferences
        allowed = {}
        if planner_input.include_explanation:
            allowed["feature_explanation"] = _AVAILABLE_TOOLS["feature_explanation"]
        if planner_input.include_behavior:
            allowed["behavior_analysis"] = _AVAILABLE_TOOLS["behavior_analysis"]
        if planner_input.include_drift:
            allowed["drift_monitoring"] = _AVAILABLE_TOOLS["drift_monitoring"]
        if planner_input.include_history and planner_input.user_id:
            allowed["transaction_history"] = _AVAILABLE_TOOLS["transaction_history"]
        allowed["graph_analysis"] = _AVAILABLE_TOOLS["graph_analysis"]

        tool_list = "\n".join(f"  - {k}: {v}" for k, v in allowed.items())
        risk_level = (
            "critical" if planner_input.fraud_probability >= 0.85 else
            "high"     if planner_input.fraud_probability >= 0.50 else
            "medium"   if planner_input.fraud_probability >= 0.25 else
            "low"
        )

        system = _SYSTEM_PROMPT.format(tool_list=tool_list)
        user = _USER_PROMPT.format(
            fraud_probability=planner_input.fraud_probability,
            risk_level=risk_level,
            request_type=planner_input.request_type,
            include_explanation=planner_input.include_explanation,
            include_behavior=planner_input.include_behavior,
            include_drift=planner_input.include_drift,
            include_history=planner_input.include_history,
            user_id_provided=planner_input.user_id is not None,
        )

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=self._temperature,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as exc:
            logger.warning("LLMPlanner API call failed: %s", exc)
            return None

    def plan(self, planner_input: PlannerInput) -> ToolPlan:
        """
        Ask the LLM to plan the tool sequence.
        Falls back to RuleBasedPlanner on any error.
        """
        if planner_input.request_type == "analyze":
            # Fast path — always rule-based (no LLM latency on the hot path)
            return self._fallback.plan(planner_input)

        llm_response = self._call_llm(planner_input)

        if llm_response is None:
            logger.info("LLMPlanner falling back to rule-based planner")
            return self._fallback.plan(planner_input)

        tools = [
            t for t in llm_response.get("tools", [])
            if t in _AVAILABLE_TOOLS
        ]
        reasoning = llm_response.get("reasoning", "LLM plan")
        logger.info("LLMPlanner selected tools=%s reasoning=%r", tools, reasoning)

        return ToolPlan(tools=tools, kwargs_overrides={"_llm_reasoning": reasoning})
