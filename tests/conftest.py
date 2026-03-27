"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Synthetic feature dictionaries
# ---------------------------------------------------------------------------

def _make_features(amount: float = 120.0, **overrides) -> dict:
    feat = {f"V{i}": float(i) * 0.1 for i in range(1, 29)}
    feat["Amount"] = amount
    feat["Time"] = 3600.0
    feat.update(overrides)
    return feat


@pytest.fixture()
def sample_features():
    return _make_features()


@pytest.fixture()
def high_risk_features():
    """Features biased toward fraud (high Amount, shifted V1)."""
    return _make_features(amount=5000.0, V1=-5.0, V2=3.0)


# ---------------------------------------------------------------------------
# Fake ML model + scaler
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.08, 0.92]])
    return model


@pytest.fixture()
def fake_scaler():
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x
    return scaler


@pytest.fixture()
def patched_inference(fake_model, fake_scaler):
    """
    Patch the inference path so no real model files are needed.
    Returns fraud_probability=0.92, alert=1 for any features.
    """
    result = {"fraud_probability": 0.92, "alert": 1, "threshold_used": 0.5}
    with patch("src.models.inference.score", return_value=result) as mock_score:
        yield mock_score


@pytest.fixture()
def patched_low_risk_inference():
    result = {"fraud_probability": 0.07, "alert": 0, "threshold_used": 0.5}
    with patch("src.models.inference.score", return_value=result) as mock_score:
        yield mock_score


@pytest.fixture()
def patched_explain():
    from src.explainability.shap_explainer import Explanation
    ex = Explanation(
        fraud_probability=0.92,
        prediction=1,
        top_features_contributing=[
            {"feature": "Amount", "shap_value": 0.35, "feature_value": 5000.0, "direction": "increases_risk"},
            {"feature": "V1", "shap_value": -0.22, "feature_value": -5.0, "direction": "decreases_risk"},
        ],
        narrative="Transaction flagged due to:\n- unusually high transaction amount (relative to learned patterns)",
    )
    with patch("src.explainability.shap_explainer.explain_transaction", return_value=ex) as mock:
        yield mock
