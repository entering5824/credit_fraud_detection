"""Feature explanation tool: SHAP top contributors and narrative."""

from __future__ import annotations

from src.explainability.shap_explainer import explain_transaction
from src.tools.base import Tool
from src.tools.registry import register_tool

FEATURE_EXPLANATION_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {"type": "object", "description": "Transaction features."},
        "top_k": {
            "type": "integer",
            "description": "Number of top contributing features to return.",
            "default": 5,
        },
        "threshold": {
            "type": "number",
            "description": "Decision threshold for prediction label.",
            "default": 0.5,
        },
    },
    "required": ["features"],
}

FEATURE_EXPLANATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "top_features_contributing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "shap_value": {"type": "number"},
                    "feature_value": {"type": "number"},
                    "direction": {"type": "string", "enum": ["increases_risk", "decreases_risk"]},
                },
            },
        },
        "narrative": {"type": "string"},
        "fraud_probability": {"type": "number"},
        "prediction": {"type": "integer"},
    },
}


def _execute(features: dict, top_k: int = 5, threshold: float = 0.5) -> dict:
    ex = explain_transaction(features, threshold=threshold, top_k=top_k)
    return {
        "top_features_contributing": ex.top_features_contributing,
        "narrative": ex.narrative,
        "fraud_probability": ex.fraud_probability,
        "prediction": ex.prediction,
    }


feature_explanation_tool = register_tool(
    Tool(
        name="feature_explanation",
        description=(
            "Uses SHAP to explain the model prediction for a single transaction. "
            "Returns the top contributing features (with direction) and a human-readable narrative."
        ),
        input_schema=FEATURE_EXPLANATION_INPUT_SCHEMA,
        output_schema=FEATURE_EXPLANATION_OUTPUT_SCHEMA,
        execute=_execute,
        timeout_seconds=30,
    )
)
