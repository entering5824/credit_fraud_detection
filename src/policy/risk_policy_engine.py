"""
Risk Policy Engine — configurable rule-based decision layer.

Sits between RiskReasoner and the final investigation report.
Applies operator-defined policies that can override the ML recommendation.

Policy rules
------------
Rules are evaluated in priority order (lowest number = highest priority).
The first rule that matches determines the final action and stops evaluation.

Built-in actions
----------------
BLOCK              Immediately block the card/transaction.
STEP_UP_AUTH       Require additional authentication (SMS OTP, biometric).
FLAG_FOR_REVIEW    Create a manual review case.
MONITOR            Low-risk: log and watch.
APPROVE            Pass through immediately.

Example policy (YAML-style representation):

  - name: "block_critical_velocity"
    priority: 1
    conditions:
      risk_score:      {gte: 90}
      fraud_pattern:   {in: ["velocity_fraud", "card_testing_attack"]}
    action: BLOCK
    reason: "Critical risk + velocity/card-testing pattern"

  - name: "step_up_high_risk"
    priority: 2
    conditions:
      risk_score:      {gte: 70}
    action: STEP_UP_AUTH
    reason: "High fraud probability — require additional authentication"

  - name: "approve_low_risk"
    priority: 99
    conditions:
      risk_score:      {lt: 25}
    action: APPROVE
    reason: "Low risk — auto-approve"

Usage
-----
    from src.policy.risk_policy_engine import PolicyEngine, PolicyRule, Condition

    engine = PolicyEngine()
    engine.add_rule(PolicyRule(
        name="block_critical",
        priority=1,
        conditions=[
            Condition("risk_score", "gte", 90),
            Condition("fraud_pattern", "in", ["velocity_fraud"]),
        ],
        action="BLOCK",
        reason="Critical velocity fraud",
    ))

    decision = engine.evaluate(report)
    # decision.action, decision.rule_name, decision.reason, decision.policy_triggered
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"BLOCK", "STEP_UP_AUTH", "FLAG_FOR_REVIEW", "MONITOR", "APPROVE"}


@dataclass
class Condition:
    """
    A single predicate on one field of the investigation report.

    Operators: gte, gt, lte, lt, eq, neq, in, not_in, is_true, is_false
    """
    field:    str
    operator: str
    value:    Any

    def evaluate(self, report: dict[str, Any]) -> bool:
        actual = report.get(self.field)
        op = self.operator
        v  = self.value
        try:
            if op == "gte":      return float(actual) >= float(v)
            if op == "gt":       return float(actual) >  float(v)
            if op == "lte":      return float(actual) <= float(v)
            if op == "lt":       return float(actual) <  float(v)
            if op == "eq":       return actual == v
            if op == "neq":      return actual != v
            if op == "in":       return actual in v
            if op == "not_in":   return actual not in v
            if op == "is_true":  return bool(actual) is True
            if op == "is_false": return bool(actual) is False
        except (TypeError, ValueError):
            return False
        return False


@dataclass
class PolicyRule:
    """
    An ordered rule: if all conditions match, apply action.
    """
    name:       str
    priority:   int
    conditions: list[Condition]
    action:     str
    reason:     str     = ""
    enabled:    bool    = True

    def matches(self, report: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        return all(c.evaluate(report) for c in self.conditions)


@dataclass
class PolicyDecision:
    """The output of the policy engine for one investigation report."""
    action:           str
    rule_name:        Optional[str]
    reason:           str
    policy_triggered: bool

    def to_dict(self) -> dict:
        return {
            "action":           self.action,
            "rule_name":        self.rule_name,
            "reason":           self.reason,
            "policy_triggered": self.policy_triggered,
        }


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Evaluates an ordered list of PolicyRules against an investigation report.

    Rules are sorted by `priority` (ascending: 1 = highest priority).
    Evaluation stops at the first matching rule.
    If no rule matches, the engine falls back to the ML-recommended action.
    """

    _DEFAULT_RULES: list[PolicyRule] = [
        PolicyRule(
            name="block_critical_velocity",
            priority=1,
            conditions=[
                Condition("risk_score",    "gte", 90),
                Condition("fraud_pattern", "in",  ["velocity_fraud", "card_testing_attack", "testing_attack"]),
            ],
            action="BLOCK",
            reason="Critical risk score + velocity/card-testing pattern — immediate block.",
        ),
        PolicyRule(
            name="block_critical_takeover",
            priority=2,
            conditions=[
                Condition("risk_score",    "gte", 90),
                Condition("fraud_pattern", "in",  ["account_takeover"]),
            ],
            action="BLOCK",
            reason="Critical risk score + account-takeover pattern — block and escalate.",
        ),
        PolicyRule(
            name="step_up_high_risk",
            priority=3,
            conditions=[
                Condition("risk_score", "gte", 70),
            ],
            action="STEP_UP_AUTH",
            reason="High fraud probability — require additional authentication.",
        ),
        PolicyRule(
            name="flag_medium_risk",
            priority=4,
            conditions=[
                Condition("risk_score", "gte", 50),
            ],
            action="FLAG_FOR_REVIEW",
            reason="Medium-high risk — route to manual review queue.",
        ),
        PolicyRule(
            name="monitor_low_risk",
            priority=5,
            conditions=[
                Condition("risk_score", "gte", 25),
            ],
            action="MONITOR",
            reason="Low-medium risk — monitor for 24 h.",
        ),
        PolicyRule(
            name="approve_very_low_risk",
            priority=99,
            conditions=[
                Condition("risk_score", "lt", 25),
            ],
            action="APPROVE",
            reason="Low risk — auto-approve.",
        ),
    ]

    def __init__(self, rules: Optional[list[PolicyRule]] = None) -> None:
        self._rules: list[PolicyRule] = sorted(
            rules if rules is not None else list(self._DEFAULT_RULES),
            key=lambda r: r.priority,
        )

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def evaluate(self, report: dict[str, Any]) -> PolicyDecision:
        """
        Evaluate all rules and return the first matching PolicyDecision.
        Falls back to the ML `recommended_action` if no rule matches.
        """
        for rule in self._rules:
            if rule.matches(report):
                return PolicyDecision(
                    action=rule.action,
                    rule_name=rule.name,
                    reason=rule.reason,
                    policy_triggered=True,
                )

        # Fallback: pass through the ML recommendation
        ml_action = report.get("recommended_action", "MONITOR")
        return PolicyDecision(
            action=_ml_to_policy_action(ml_action),
            rule_name=None,
            reason="No policy rule matched — using ML recommendation.",
            policy_triggered=False,
        )

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule and re-sort by priority."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def list_rules(self) -> list[dict]:
        return [
            {
                "name": r.name, "priority": r.priority,
                "action": r.action, "enabled": r.enabled, "reason": r.reason,
            }
            for r in self._rules
        ]

    def enrich_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate the engine and merge the PolicyDecision into the report.
        Adds `policy_action`, `policy_rule`, `policy_reason`, `policy_triggered`.
        """
        decision = self.evaluate(report)
        report = dict(report)
        report["policy_action"]    = decision.action
        report["policy_rule"]      = decision.rule_name
        report["policy_reason"]    = decision.reason
        report["policy_triggered"] = decision.policy_triggered
        # Override recommended_action with policy decision for consistency
        report["recommended_action"] = _policy_to_display(decision.action)
        return report


# ---------------------------------------------------------------------------
# Action mapping helpers
# ---------------------------------------------------------------------------

_ML_TO_POLICY = {
    "approve":              "APPROVE",
    "monitor":              "MONITOR",
    "flag for manual review": "FLAG_FOR_REVIEW",
}
_POLICY_TO_DISPLAY = {
    "BLOCK":          "block transaction",
    "STEP_UP_AUTH":   "require step-up authentication",
    "FLAG_FOR_REVIEW":"flag for manual review",
    "MONITOR":        "monitor",
    "APPROVE":        "approve",
}


def _ml_to_policy_action(ml: str) -> str:
    return _ML_TO_POLICY.get(ml.lower(), "MONITOR")


def _policy_to_display(action: str) -> str:
    return _POLICY_TO_DISPLAY.get(action, action.lower())


# ---------------------------------------------------------------------------
# Module-level default engine
# ---------------------------------------------------------------------------

_default_engine = PolicyEngine()


def get_policy_engine() -> PolicyEngine:
    return _default_engine
