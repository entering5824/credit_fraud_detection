"""Unit tests: RiskReasoner — pattern classification, confidence, signal ranking."""

from __future__ import annotations

import pytest
from src.agents.reasoning.risk_reasoner import RiskReasoner, _classify_pattern, _confidence


THRESHOLD = 0.5
HIGH_PROB = 0.92
LOW_PROB = 0.07


def _score(prob: float) -> dict:
    risk_level = "critical" if prob >= 0.85 else "high" if prob >= THRESHOLD else "low"
    return {
        "fraud_probability": prob,
        "risk_level": risk_level,
        "threshold_used": THRESHOLD,
    }


def _behavior(velocity_1h: int = 0, spike_ratio: float = 1.0, new_merchant: int = 0) -> dict:
    flags = []
    if spike_ratio > 2.0:
        flags.append("spending spike detected")
    if new_merchant:
        flags.append("new merchant interaction")
    if velocity_1h >= 5:
        flags.append("high transaction velocity")
    return {
        "transaction_velocity_1h": velocity_1h,
        "spending_spike_ratio": spike_ratio,
        "is_new_merchant_for_user": new_merchant,
        "flags": flags,
    }


def _explanation(top_shap_abs: float = 0.4) -> dict:
    return {
        "top_features_contributing": [
            {"feature": "Amount", "shap_value": top_shap_abs, "feature_value": 999.0, "direction": "increases_risk"},
        ],
        "narrative": "- unusually high transaction amount (relative to learned patterns)",
    }


class TestPatternClassification:
    def test_velocity_fraud(self):
        pattern = _classify_pattern(HIGH_PROB, THRESHOLD, _behavior(velocity_1h=6, spike_ratio=2.5), None, amount=50.0)
        assert pattern == "velocity_fraud"

    def test_testing_attack_low_amount_high_velocity(self):
        pattern = _classify_pattern(HIGH_PROB, THRESHOLD, _behavior(velocity_1h=7), None, amount=1.0)
        assert pattern == "testing_attack"

    def test_account_takeover(self):
        b = _behavior(velocity_1h=3, spike_ratio=3.0, new_merchant=1)
        pattern = _classify_pattern(HIGH_PROB, THRESHOLD, b, None, amount=200.0)
        assert pattern == "account_takeover"

    def test_large_anomalous_purchase(self):
        b = _behavior(velocity_1h=0, spike_ratio=4.0)
        pattern = _classify_pattern(HIGH_PROB, THRESHOLD, b, None, amount=1500.0)
        assert pattern == "large_anomalous_purchase"

    def test_unknown_for_low_risk(self):
        pattern = _classify_pattern(LOW_PROB, THRESHOLD, _behavior(), None, amount=20.0)
        assert pattern == "unknown"

    def test_none_behavior_returns_unknown(self):
        assert _classify_pattern(HIGH_PROB, THRESHOLD, None, None) == "unknown"


class TestConfidenceScore:
    def test_high_risk_high_confidence(self):
        score = _confidence(HIGH_PROB, _behavior(spike_ratio=3.0, new_merchant=1, velocity_1h=6), _explanation(0.5))
        assert score > 0.5

    def test_low_risk_low_confidence(self):
        score = _confidence(LOW_PROB, _behavior(), None)
        assert score < 0.3

    def test_capped_at_one(self):
        score = _confidence(1.0, _behavior(spike_ratio=10, new_merchant=1, velocity_1h=10), _explanation(5.0))
        assert score <= 1.0

    def test_zero_when_no_evidence(self):
        score = _confidence(0.0, None, None)
        assert score == 0.0


class TestReasoningResult:
    def setup_method(self):
        self.reasoner = RiskReasoner()

    def test_full_investigation(self):
        result = self.reasoner.reason(
            score_result=_score(HIGH_PROB),
            behavior_result=_behavior(spike_ratio=3.5, new_merchant=1),
            explanation_result=_explanation(0.4),
            threshold=THRESHOLD,
        )
        assert result.confidence_score > 0
        assert len(result.key_risk_signals) > 0
        assert result.recommended_action == "flag for manual review"
        assert "critical" in result.analyst_summary or "high" in result.analyst_summary

    def test_low_risk_approved(self):
        result = self.reasoner.reason(score_result=_score(LOW_PROB), threshold=THRESHOLD)
        assert result.recommended_action == "approve"

    def test_medium_risk_monitored(self):
        result = self.reasoner.reason(score_result=_score(0.35), threshold=THRESHOLD)
        assert result.recommended_action == "monitor"

    def test_ranked_signals_ordered_by_weight(self):
        result = self.reasoner.reason(
            score_result=_score(HIGH_PROB),
            behavior_result=_behavior(spike_ratio=3.0),
            explanation_result=_explanation(0.6),
            threshold=THRESHOLD,
        )
        weights = [s["weight"] for s in result.ranked_signals]
        assert weights == sorted(weights, reverse=True)
