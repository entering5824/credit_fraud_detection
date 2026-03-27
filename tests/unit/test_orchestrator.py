"""Unit tests: AgentOrchestrator state transitions and partial-report policy."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.agents.agent_orchestrator import AgentOrchestrator, InvestigationState, build_report
from src.tools.base import Tool, ToolResult
from src.tools.registry import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features():
    feat = {f"V{i}": 0.0 for i in range(1, 29)}
    feat["Amount"] = 100.0
    feat["Time"] = 0.0
    return feat


def _fake_score_tool(prob: float = 0.92) -> Tool:
    def execute(features, threshold=0.5):
        return {
            "fraud_probability": prob,
            "risk_score": prob * 100,
            "risk_level": "critical" if prob > 0.85 else "low",
            "prediction": 1 if prob >= threshold else 0,
            "threshold_used": threshold,
            "model_version": "1.0.0",
            "feature_vector_hash": "abcdef1234567890",
        }
    return Tool(name="fraud_scoring", description="", input_schema={}, output_schema={}, execute=execute)


def _fake_explanation_tool() -> Tool:
    def execute(features, top_k=5, threshold=0.5):
        return {
            "top_features_contributing": [
                {"feature": "Amount", "shap_value": 0.3, "feature_value": 100.0, "direction": "increases_risk"}
            ],
            "narrative": "Transaction flagged due to:\n- unusually high transaction amount",
            "fraud_probability": 0.92,
            "prediction": 1,
        }
    return Tool(name="feature_explanation", description="", input_schema={}, output_schema={}, execute=execute)


def _fake_behavior_tool(spike: float = 3.0) -> Tool:
    def execute(features, **kw):
        return {
            "spending_spike_ratio": spike,
            "merchant_frequency_score": 0.1,
            "transaction_velocity_1h": 0,
            "is_new_merchant_for_user": 1,
            "signals": {},
            "flags": ["spending spike detected", "new merchant interaction"],
        }
    return Tool(name="behavior_analysis", description="", input_schema={}, output_schema={}, execute=execute)


def _failing_tool(name: str) -> Tool:
    def execute(**kw):
        raise RuntimeError("simulated tool failure")
    return Tool(name=name, description="", input_schema={}, output_schema={}, execute=execute)


def _build_registry(*tools: Tool) -> dict:
    return {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInvestigationState:
    def test_initial_state(self):
        state = InvestigationState(transaction_id="tx_001")
        assert state.transaction_id == "tx_001"
        assert state.is_partial_report is False
        assert state.failed_tools == []

    def test_failed_tools_tracked(self):
        state = InvestigationState()
        state.failed_tools.append("behavior_analysis")
        state.is_partial_report = True
        assert "behavior_analysis" in state.failed_tools
        assert state.is_partial_report


class TestOrchestratorRun:
    """High-risk transaction — all secondary tools run."""

    def setup_method(self):
        self.features = _make_features()
        self.registry = _build_registry(
            _fake_score_tool(0.92),
            _fake_explanation_tool(),
            _fake_behavior_tool(),
        )
        self.orchestrator = AgentOrchestrator(tools_registry=self.registry)

    def test_report_contains_required_keys(self):
        report = self.orchestrator.run(self.features, request_type="investigate")
        required = {
            "fraud_probability", "risk_score", "risk_level", "prediction",
            "threshold_used", "fraud_pattern", "confidence_score",
            "key_risk_signals", "recommended_action", "analyst_summary",
            "partial_report", "_meta",
        }
        assert required.issubset(set(report.keys()))

    def test_high_risk_flags_for_review(self):
        report = self.orchestrator.run(self.features, request_type="investigate")
        assert report["recommended_action"] == "flag for manual review"

    def test_model_explanation_populated(self):
        report = self.orchestrator.run(self.features, request_type="investigate")
        assert len(report["model_explanation"]["top_features_contributing"]) > 0

    def test_meta_tool_calls_logged(self):
        report = self.orchestrator.run(self.features, request_type="investigate")
        tool_names = [tc["tool"] for tc in report["_meta"]["tool_calls"]]
        assert "fraud_scoring" in tool_names

    def test_analyze_request_type_skips_secondary_tools(self):
        report = self.orchestrator.run(self.features, request_type="analyze")
        tool_names = [tc["tool"] for tc in report["_meta"]["tool_calls"]]
        # Only scoring should run for "analyze"
        assert tool_names == ["fraud_scoring"]


class TestOrchestratorDegradedMode:
    """Verify partial-report policy when a tool fails."""

    def setup_method(self):
        self.features = _make_features()

    def test_failing_explanation_tool_gives_partial_report(self):
        registry = _build_registry(
            _fake_score_tool(0.92),
            _failing_tool("feature_explanation"),
            _fake_behavior_tool(),
        )
        orchestrator = AgentOrchestrator(tools_registry=registry)
        report = orchestrator.run(self.features, request_type="investigate")
        assert report["partial_report"] is True
        assert "feature_explanation" in report["_meta"]["failed_tools"]

    def test_confidence_penalised_on_failure(self):
        registry_ok = _build_registry(
            _fake_score_tool(0.92),
            _fake_explanation_tool(),
            _fake_behavior_tool(),
        )
        registry_bad = _build_registry(
            _fake_score_tool(0.92),
            _failing_tool("feature_explanation"),
            _fake_behavior_tool(),
        )
        good_report = AgentOrchestrator(tools_registry=registry_ok).run(
            self.features, request_type="investigate"
        )
        bad_report = AgentOrchestrator(tools_registry=registry_bad).run(
            self.features, request_type="investigate"
        )
        assert bad_report["confidence_score"] < good_report["confidence_score"]

    def test_scoring_failure_returns_error_dict(self):
        registry = _build_registry(_failing_tool("fraud_scoring"))
        orchestrator = AgentOrchestrator(tools_registry=registry)
        report = orchestrator.run(self.features, request_type="investigate")
        assert "error" in report
        assert report.get("partial_report") is True
