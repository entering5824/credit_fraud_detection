"""
Risk reasoning layer: classify fraud patterns, rank signals, compute confidence.

Fraud pattern taxonomy
----------------------
velocity_fraud          Rapid repeated transactions in a short window.
account_takeover        New merchant + unusual amount + high velocity.
testing_attack          Many low-value transactions probing card validity.
large_anomalous_purchase  Single high-value transaction with no behavioral context.
unknown                 Could not match a known pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Pattern thresholds (tunable)
# ---------------------------------------------------------------------------

_VELOCITY_HIGH = 5          # transactions_last_1h >= this → velocity_fraud candidate
_SPIKE_HIGH = 3.0           # spending_spike_ratio >= this
_SMALL_AMOUNT_MAX = 5.0     # testing attacks often probe with very small amounts
_LARGE_AMOUNT_MIN = 500.0   # large anomalous purchase threshold
_CONFIDENCE_SHAP_WEIGHT = 0.4
_CONFIDENCE_BEHAVIOR_WEIGHT = 0.3
_CONFIDENCE_PROB_WEIGHT = 0.3


@dataclass
class ReasoningResult:
    """Output of the risk reasoning step."""

    fraud_pattern: str                    # taxonomy label
    confidence_score: float               # 0.0 – 1.0
    key_risk_signals: list[str] = field(default_factory=list)
    ranked_signals: list[dict[str, Any]] = field(default_factory=list)
    recommended_action: str = "monitor"
    analyst_summary: str = ""


def _classify_pattern(
    fraud_probability: float,
    threshold: float,
    behavior: Optional[dict],
    explanation: Optional[dict],
    amount: float = 0.0,
) -> str:
    """Return the most likely fraud pattern label."""
    if behavior is None:
        return "unknown"

    velocity = float(behavior.get("transaction_velocity_1h") or 0)
    spike = float(behavior.get("spending_spike_ratio") or 1)
    is_new_merchant = behavior.get("is_new_merchant_for_user") == 1

    # Priority order: most specific first
    if velocity >= _VELOCITY_HIGH and amount <= _SMALL_AMOUNT_MAX and amount > 0:
        return "testing_attack"
    if velocity >= _VELOCITY_HIGH:
        return "velocity_fraud"
    if is_new_merchant and velocity >= 2 and spike >= 2:
        return "account_takeover"
    if spike >= _SPIKE_HIGH and amount >= _LARGE_AMOUNT_MIN:
        return "large_anomalous_purchase"
    if fraud_probability >= threshold:
        return "unknown"
    return "unknown"


def _confidence(
    fraud_probability: float,
    behavior: Optional[dict],
    explanation: Optional[dict],
) -> float:
    """
    Heuristic confidence score (0–1):
    - High probability → high contribution from prob component
    - Behavior flags → behavior component
    - SHAP top feature magnitude → shap component
    """
    prob_score = float(fraud_probability)

    beh_score = 0.0
    if behavior:
        flags = behavior.get("flags", [])
        beh_score = min(len(flags) / 4.0, 1.0)

    shap_score = 0.0
    if explanation:
        top = explanation.get("top_features_contributing", [])
        if top:
            abs_shap = [abs(f.get("shap_value", 0)) for f in top[:3]]
            shap_score = min(sum(abs_shap) / 3.0, 1.0)

    confidence = (
        _CONFIDENCE_PROB_WEIGHT * prob_score
        + _CONFIDENCE_BEHAVIOR_WEIGHT * beh_score
        + _CONFIDENCE_SHAP_WEIGHT * shap_score
    )
    return round(min(confidence, 1.0), 3)


def _rank_signals(
    behavior: Optional[dict],
    explanation: Optional[dict],
) -> list[dict[str, Any]]:
    """Return risk signals ranked by severity (behavior flags + top SHAP features)."""
    ranked: list[dict[str, Any]] = []

    if behavior:
        for flag in behavior.get("flags", []):
            ranked.append({"signal": flag, "source": "behavior", "weight": 1.0})

    if explanation:
        for feat in explanation.get("top_features_contributing", [])[:5]:
            ranked.append(
                {
                    "signal": f"{feat['feature']} ({feat['direction']})",
                    "source": "shap",
                    "weight": round(abs(feat.get("shap_value", 0)), 4),
                }
            )

    ranked.sort(key=lambda x: x["weight"], reverse=True)
    return ranked


def _build_key_signals(behavior: Optional[dict], explanation: Optional[dict]) -> list[str]:
    signals: list[str] = []
    if behavior:
        signals.extend(behavior.get("flags", []))
    if explanation and explanation.get("narrative"):
        for line in explanation["narrative"].split("\n"):
            line = line.strip().lstrip("- ")
            if line and "flagged" not in line.lower() and line not in signals:
                signals.append(line)
    return signals[:10]


def _recommended_action(prob: float, threshold: float, pattern: str) -> str:
    if prob >= threshold:
        return "flag for manual review"
    if prob >= 0.3 or pattern != "unknown":
        return "monitor"
    return "approve"


def _analyst_summary(
    fraud_probability: float,
    risk_level: str,
    pattern: str,
    key_signals: list[str],
    recommended_action: str,
) -> str:
    signals_text = "; ".join(key_signals[:4]) if key_signals else "no specific signals detected"
    return (
        f"Transaction assessed as {risk_level} risk "
        f"(fraud probability: {fraud_probability:.1%}). "
        f"Possible pattern: {pattern.replace('_', ' ')}. "
        f"Key signals: {signals_text}. "
        f"Recommended action: {recommended_action}."
    )


class RiskReasoner:
    """Combines tool outputs into a final reasoned report."""

    def reason(
        self,
        score_result: dict[str, Any],
        behavior_result: Optional[dict[str, Any]] = None,
        explanation_result: Optional[dict[str, Any]] = None,
        drift_result: Optional[dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> ReasoningResult:
        prob = float(score_result.get("fraud_probability", 0.0))
        risk_level = score_result.get("risk_level", "unknown")
        amount = float(score_result.get("feature_value_amount", 0.0))

        pattern = _classify_pattern(
            fraud_probability=prob,
            threshold=threshold,
            behavior=behavior_result,
            explanation=explanation_result,
            amount=amount,
        )
        confidence = _confidence(prob, behavior_result, explanation_result)
        ranked = _rank_signals(behavior_result, explanation_result)
        key_signals = _build_key_signals(behavior_result, explanation_result)
        action = _recommended_action(prob, threshold, pattern)
        summary = _analyst_summary(prob, risk_level, pattern, key_signals, action)

        return ReasoningResult(
            fraud_pattern=pattern,
            confidence_score=confidence,
            key_risk_signals=key_signals,
            ranked_signals=ranked,
            recommended_action=action,
            analyst_summary=summary,
        )
