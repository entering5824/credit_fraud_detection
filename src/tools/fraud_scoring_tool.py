"""Fraud scoring tool: ML fraud probability with enriched metadata."""

from __future__ import annotations

import hashlib
import json

from src.core.thresholds import load_threshold_config
from src.models.inference import score as score_transaction
from src.tools.base import Tool
from src.tools.registry import register_tool

MODEL_VERSION = "1.0.0"  # bumped when a new model artifact is deployed

FRAUD_SCORING_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "object",
            "description": "Transaction features (V1..V28, Amount, optional engineered).",
        },
        "threshold": {
            "type": "number",
            "description": "Optional decision threshold override.",
        },
    },
    "required": ["features"],
}

FRAUD_SCORING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "fraud_probability": {"type": "number"},
        "risk_score": {"type": "number", "description": "0–100 scale (100 × probability)."},
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
            "description": "Human-readable risk tier.",
        },
        "prediction": {"type": "integer", "enum": [0, 1]},
        "threshold_used": {"type": "number"},
        "model_version": {"type": "string"},
        "feature_vector_hash": {
            "type": "string",
            "description": "SHA-256 of the sorted feature dict for reproducibility.",
        },
    },
}


def _risk_level(prob: float, threshold: float) -> str:
    if prob < 0.2:
        return "low"
    if prob < threshold:
        return "medium"
    if prob < 0.85:
        return "high"
    return "critical"


def _hash_features(features: dict) -> str:
    canonical = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _execute(features: dict, threshold: float | None = None) -> dict:
    if threshold is None:
        threshold = load_threshold_config().optimal_threshold
    res = score_transaction(features, threshold=threshold)
    prob = float(res["fraud_probability"])
    t = float(res["threshold_used"])
    return {
        "fraud_probability": prob,
        "risk_score": round(prob * 100.0, 2),
        "risk_level": _risk_level(prob, t),
        "prediction": int(res["alert"]),
        "threshold_used": t,
        "model_version": MODEL_VERSION,
        "feature_vector_hash": _hash_features(features),
    }


fraud_scoring_tool = register_tool(
    Tool(
        name="fraud_scoring",
        description=(
            "Uses the trained ML model to score a transaction. "
            "Returns fraud probability, risk level (low/medium/high/critical), "
            "model version, and a feature hash for reproducibility."
        ),
        input_schema=FRAUD_SCORING_INPUT_SCHEMA,
        output_schema=FRAUD_SCORING_OUTPUT_SCHEMA,
        execute=_execute,
        timeout_seconds=10,
    )
)
