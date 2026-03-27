"""
Integration tests: agent API endpoints.

Fake tools are injected by replacing the TOOL_REGISTRY in the orchestrator module
so no real model files are required.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.tools.base import Tool, ToolResult


# ---------------------------------------------------------------------------
# Fake tool factories
# ---------------------------------------------------------------------------

def _fake_score_tool(prob: float, threshold: float = 0.5) -> Tool:
    risk = "critical" if prob >= 0.85 else "high" if prob >= threshold else "low"
    def execute(features, threshold=threshold):
        return {
            "fraud_probability": prob,
            "risk_score": round(prob * 100, 2),
            "risk_level": risk,
            "prediction": 1 if prob >= threshold else 0,
            "threshold_used": threshold,
            "model_version": "test-1.0",
            "feature_vector_hash": "deadbeefcafebabe",
        }
    return Tool(name="fraud_scoring", description="fake", input_schema={}, output_schema={}, execute=execute)


def _fake_explanation_tool(prob: float = 0.92) -> Tool:
    def execute(features, top_k=5, threshold=0.5):
        return {
            "top_features_contributing": [
                {"feature": "Amount", "shap_value": 0.35, "feature_value": 5000.0, "direction": "increases_risk"},
                {"feature": "V1", "shap_value": -0.20, "feature_value": -3.0, "direction": "decreases_risk"},
            ],
            "narrative": "Transaction flagged due to:\n- unusually high transaction amount",
            "fraud_probability": prob,
            "prediction": 1,
        }
    return Tool(name="feature_explanation", description="fake", input_schema={}, output_schema={}, execute=execute)


def _fake_behavior_tool() -> Tool:
    def execute(features, **kw):
        return {
            "spending_spike_ratio": 3.5,
            "merchant_frequency_score": 0.1,
            "transaction_velocity_1h": 2.0,
            "transaction_velocity_24h": 0.0,
            "is_new_merchant_for_user": 1,
            "user_avg_amount": 80.0,
            "signals": {},
            "flags": ["spending spike detected", "new merchant interaction"],
        }
    return Tool(name="behavior_analysis", description="fake", input_schema={}, output_schema={}, execute=execute)


def _noop_tool(name: str) -> Tool:
    return Tool(name=name, description="noop", input_schema={}, output_schema={}, execute=lambda **kw: {})


def _high_risk_registry() -> dict:
    return {
        "fraud_scoring": _fake_score_tool(0.92),
        "feature_explanation": _fake_explanation_tool(0.92),
        "behavior_analysis": _fake_behavior_tool(),
        "drift_monitoring": _noop_tool("drift_monitoring"),
        "transaction_history": _noop_tool("transaction_history"),
    }


def _low_risk_registry() -> dict:
    return {
        "fraud_scoring": _fake_score_tool(0.07),
        "feature_explanation": _fake_explanation_tool(0.07),
        "behavior_analysis": _noop_tool("behavior_analysis"),
        "drift_monitoring": _noop_tool("drift_monitoring"),
        "transaction_history": _noop_tool("transaction_history"),
    }


def _base_features() -> dict:
    feat = {f"V{i}": 0.1 for i in range(1, 29)}
    feat["Amount"] = 5000.0
    feat["Time"] = 3600.0
    return feat


# ---------------------------------------------------------------------------
# TestClient factory — injects fake registry
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    # Keep patch active for the entire fixture lifetime so every request sees fake tools
    with patch("src.agents.agent_orchestrator.TOOL_REGISTRY", _high_risk_registry()):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def low_risk_client():
    from src.api.main import app
    with patch("src.agents.agent_orchestrator.TOOL_REGISTRY", _low_risk_registry()):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /agent/analyze
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    def test_returns_200(self, client):
        r = client.post("/agent/analyze", json={"features": _base_features()})
        assert r.status_code == 200

    def test_contains_required_fields(self, client):
        r = client.post("/agent/analyze", json={"features": _base_features()})
        body = r.json()
        for field in ("fraud_probability", "risk_level", "prediction", "recommended_action"):
            assert field in body, f"Missing field: {field}"

    def test_no_shap_in_analyze(self, client):
        r = client.post("/agent/analyze", json={"features": _base_features()})
        expl = r.json().get("model_explanation", {})
        assert expl.get("top_features_contributing", []) == []

    def test_meta_tool_calls_present(self, client):
        r = client.post("/agent/analyze", json={"features": _base_features()})
        assert "_meta" in r.json()
        assert "tool_calls" in r.json()["_meta"]

    def test_analyze_only_runs_scoring(self, client):
        r = client.post("/agent/analyze", json={"features": _base_features()})
        tool_names = [tc["tool"] for tc in r.json()["_meta"]["tool_calls"]]
        assert tool_names == ["fraud_scoring"]

    def test_missing_features_returns_400(self, client):
        r = client.post("/agent/analyze", json={})
        assert r.status_code == 400

    def test_partial_report_flag_present(self, client):
        r = client.post("/agent/analyze", json={"features": _base_features()})
        assert "partial_report" in r.json()


# ---------------------------------------------------------------------------
# POST /agent/investigate
# ---------------------------------------------------------------------------

class TestInvestigateEndpoint:
    def test_returns_200(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        assert r.status_code == 200

    def test_high_risk_recommended_for_review(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        # Policy engine maps high/critical risk → block or step-up auth or flag
        action = r.json()["recommended_action"]
        assert action in ("flag for manual review", "block transaction",
                          "require step-up authentication"), f"Unexpected action: {action}"

    def test_low_risk_approved(self, low_risk_client):
        r = low_risk_client.post("/agent/investigate", json={"features": _base_features()})
        assert r.json()["recommended_action"] == "approve"

    def test_session_id_returned(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        assert "session_id" in r.json()

    def test_fraud_pattern_present(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        assert "fraud_pattern" in r.json()

    def test_confidence_score_in_range(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        cs = r.json()["confidence_score"]
        assert 0.0 <= cs <= 1.0

    def test_risk_score_is_probability_times_100(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        body = r.json()
        assert abs(body["risk_score"] - body["fraud_probability"] * 100) < 1.0

    def test_behavioral_anomalies_populated(self, client):
        r = client.post("/agent/investigate", json={"features": _base_features()})
        assert isinstance(r.json().get("behavioral_anomalies"), list)


# ---------------------------------------------------------------------------
# POST /agent/explain
# ---------------------------------------------------------------------------

class TestExplainEndpoint:
    def test_returns_200(self, client):
        r = client.post("/agent/explain", json={"features": _base_features()})
        assert r.status_code == 200

    def test_model_explanation_populated(self, client):
        r = client.post("/agent/explain", json={"features": _base_features()})
        expl = r.json().get("model_explanation", {})
        assert len(expl.get("top_features_contributing", [])) > 0

    def test_analyst_summary_nonempty(self, client):
        r = client.post("/agent/explain", json={"features": _base_features()})
        assert len(r.json().get("analyst_summary", "")) > 0

    def test_missing_input_returns_400(self, client):
        r = client.post("/agent/explain", json={})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /agent/session/{id}
# ---------------------------------------------------------------------------

class TestSessionEndpoint:
    def test_session_retrievable_after_investigate(self, client):
        inv_r = client.post("/agent/investigate", json={"features": _base_features()})
        sid = inv_r.json().get("session_id")
        assert sid is not None
        sess_r = client.get(f"/agent/session/{sid}")
        assert sess_r.status_code == 200
        body = sess_r.json()
        assert body["session_id"] == sid
        assert body["transaction_count"] >= 1

    def test_unknown_session_returns_404(self, client):
        r = client.get("/agent/session/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_session_contains_final_report(self, client):
        inv_r = client.post("/agent/investigate", json={"features": _base_features()})
        sid = inv_r.json()["session_id"]
        sess_r = client.get(f"/agent/session/{sid}")
        assert len(sess_r.json()["final_reports"]) >= 1
