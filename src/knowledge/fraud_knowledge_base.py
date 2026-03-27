"""
Fraud Knowledge Base — structured access to fraud patterns, investigation
guidelines, and escalation rules.

The knowledge base loads from `docs/knowledge/fraud_patterns.md` and
exposes a queryable Python interface.  An LLM planner can inject this
context into its prompt to ground decisions in domain expertise.

Usage
-----
    from src.knowledge.fraud_knowledge_base import FraudKnowledgeBase

    kb = FraudKnowledgeBase()
    pattern = kb.get_pattern("account_takeover")
    print(pattern["description"])

    context = kb.build_llm_context(fraud_probability=0.91, detected_signals=["new_merchant", "spike"])
    # → compact string to inject into LLM system prompt
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Static pattern catalogue (mirrors docs/knowledge/fraud_patterns.md)
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, dict] = {
    "velocity_fraud": {
        "name":         "Velocity Fraud",
        "description":  "Many transactions in a short window, typically within one hour.",
        "signals":      ["transactions_last_1h >= 5", "transaction_velocity_1h spike"],
        "risk_level":   "high",
        "action":       "Block card, escalate to fraud ops",
        "fp_risk":      "low",
    },
    "account_takeover": {
        "name":         "Account Takeover",
        "description":  "Attacker uses a legitimate account to transact at new merchants.",
        "signals":      ["is_new_merchant_for_user == 1", "spending_spike_ratio >= 3"],
        "risk_level":   "critical",
        "action":       "Lock account, escalate to fraud ops",
        "fp_risk":      "medium",
    },
    "testing_attack": {
        "name":         "Card Testing Attack",
        "description":  "Probing card validity with tiny amounts across multiple merchants.",
        "signals":      ["Amount <= 5", "3+ distinct merchants in 1h"],
        "risk_level":   "critical",
        "action":       "Block card immediately",
        "fp_risk":      "very low",
    },
    "large_anomalous_purchase": {
        "name":         "Large Anomalous Purchase",
        "description":  "Single high-value transaction far above historical average.",
        "signals":      ["spending_spike_ratio >= 4", "Amount >= 500"],
        "risk_level":   "high",
        "action":       "Flag for manual review, optional 2FA",
        "fp_risk":      "high",
    },
    "velocity_burst": {
        "name":         "Velocity Burst (Graph)",
        "description":  "Graph-level: >= 5 transactions in 1 h detected in historical graph.",
        "signals":      ["graph.velocity_1h >= 5"],
        "risk_level":   "high",
        "action":       "Temporary card hold",
        "fp_risk":      "medium",
    },
    "merchant_cluster": {
        "name":         "Merchant Cluster Fraud",
        "description":  "Merchant with high historical fraud rate (>= 15%).",
        "signals":      ["merchant_fraud_rate >= 0.15"],
        "risk_level":   "high",
        "action":       "Flag for manual review",
        "fp_risk":      "medium",
    },
    "dormant_account": {
        "name":         "Dormant Account Reactivation",
        "description":  "Account inactive 30+ days, suddenly transacts at new merchant.",
        "signals":      ["silence_days >= 30"],
        "risk_level":   "medium",
        "action":       "Send OTP, monitor next 24 h",
        "fp_risk":      "medium",
    },
    "high_degree_anomaly": {
        "name":         "High Degree Anomaly",
        "description":  "User transacted at 5+ distinct merchants in 24 h.",
        "signals":      ["distinct_merchants_24h >= 5"],
        "risk_level":   "high",
        "action":       "Monitor, optional 2FA",
        "fp_risk":      "medium",
    },
    "unknown": {
        "name":         "Unknown Pattern",
        "description":  "Could not match a known fraud pattern.",
        "signals":      [],
        "risk_level":   "variable",
        "action":       "Monitor",
        "fp_risk":      "variable",
    },
}

_TRIAGE_RULES = [
    {"risk_level": "critical", "prob_min": 0.85, "action": "Immediate review",  "sla": "15 minutes"},
    {"risk_level": "high",     "prob_min": 0.50, "action": "Same-day review",   "sla": "4 hours"},
    {"risk_level": "medium",   "prob_min": 0.25, "action": "Weekly review",     "sla": "3 days"},
    {"risk_level": "low",      "prob_min": 0.00, "action": "Auto-approve",      "sla": "—"},
]


class FraudKnowledgeBase:
    """
    Structured knowledge base for fraud investigation.

    Parameters
    ----------
    knowledge_file : Path to the Markdown knowledge document.
                     Loaded lazily for LLM context injection.
    """

    def __init__(self, knowledge_file: Optional[Path] = None) -> None:
        self._file = knowledge_file or (
            Path(__file__).parents[2] / "docs" / "knowledge" / "fraud_patterns.md"
        )
        self._md_cache: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Pattern queries
    # ------------------------------------------------------------------ #

    def get_pattern(self, pattern_name: str) -> dict:
        """Return the full pattern entry, or the 'unknown' entry."""
        return _PATTERNS.get(pattern_name, _PATTERNS["unknown"])

    def list_patterns(self) -> list[str]:
        return list(_PATTERNS.keys())

    def get_action(self, pattern_name: str) -> str:
        return self.get_pattern(pattern_name).get("action", "Monitor")

    def get_triage(self, fraud_probability: float) -> dict:
        for rule in _TRIAGE_RULES:
            if fraud_probability >= rule["prob_min"]:
                return rule
        return _TRIAGE_RULES[-1]

    # ------------------------------------------------------------------ #
    # LLM context builder
    # ------------------------------------------------------------------ #

    def build_llm_context(
        self,
        fraud_probability: float,
        detected_signals: Optional[list[str]] = None,
        fraud_pattern: Optional[str] = None,
    ) -> str:
        """
        Build a compact context string suitable for injecting into an LLM prompt.

        Includes: triage rule, matched pattern description, relevant signals.
        """
        triage = self.get_triage(fraud_probability)
        pattern = self.get_pattern(fraud_pattern or "unknown")
        signals_str = (
            "\n".join(f"  - {s}" for s in (detected_signals or []))
            or "  (none detected)"
        )

        return textwrap.dedent(f"""
            ## Investigation Context (from Knowledge Base)
            Fraud probability : {fraud_probability:.3f}
            Triage rule       : {triage["action"]} (SLA: {triage["sla"]})
            Pattern matched   : {pattern["name"]}
            Pattern description: {pattern["description"]}
            Recommended action: {pattern["action"]}
            False positive risk: {pattern.get("fp_risk", "unknown")}
            Detected signals  :
            {signals_str}
        """).strip()

    # ------------------------------------------------------------------ #
    # Raw Markdown (for LLM context injection)
    # ------------------------------------------------------------------ #

    def full_text(self) -> str:
        """Return the full Markdown knowledge document."""
        if self._md_cache is None:
            try:
                self._md_cache = self._file.read_text(encoding="utf-8")
            except FileNotFoundError:
                self._md_cache = "(Knowledge base file not found)"
        return self._md_cache
