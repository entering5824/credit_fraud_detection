from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline.feature_store import get as fs_get  # legacy integration
from src.explainability.shap_explainer import explain_transaction
from src.models.inference import score as score_transaction
from src.schemas.feature_schema import validate_feature_dict
from src.core.thresholds import load_threshold_config
from src.monitoring.performance_monitoring import PerfRecord, append_perf_record
from src.core.paths import get_paths
import time


app = FastAPI(title="Fraud Scoring API", version="2.0.0")


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Transaction features (V1..V28, Amount, optional engineered).")
    model_name: str = Field("xgboost", description="Registered model name (default: xgboost).")
    model_version: Optional[str] = Field(None, description="Model version in registry (default: latest).")
    threshold: Optional[float] = Field(None, description="Override decision threshold.")
    top_k: int = Field(5, ge=1, le=20, description="Top features to include in explanation.")


class PredictResponse(BaseModel):
    fraud_probability: float
    risk_score: float
    prediction: int
    threshold_used: float
    explanation: Dict[str, Any]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Canonical real-time fraud scoring endpoint.
    """
    # Validate minimal schema (keeps API behavior predictable)
    validate_feature_dict(req.features)

    # Probability + alert (fast path)
    # If caller doesn't override threshold, use centralized config default.
    effective_threshold = req.threshold
    if effective_threshold is None:
        effective_threshold = load_threshold_config().optimal_threshold

    t0 = time.perf_counter()
    base = score_transaction(req.features, threshold=effective_threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    threshold_used = float(base["threshold_used"])
    fraud_probability = float(base["fraud_probability"])
    prediction = int(base["alert"])

    # SHAP explanation (tree models only for now; legacy fallback supported)
    try:
        ex = explain_transaction(
            req.features,
            model_name=req.model_name,
            version=req.model_version,
            threshold=threshold_used,
            top_k=req.top_k,
        )
        explanation = {
            "top_features_contributing": ex.top_features_contributing,
            "narrative": ex.narrative,
        }
    except Exception as e:
        explanation = {"error": str(e), "top_features_contributing": [], "narrative": ""}

    resp = PredictResponse(
        fraud_probability=fraud_probability,
        risk_score=float(100.0 * fraud_probability),
        prediction=prediction,
        threshold_used=threshold_used,
        explanation=explanation,
    )
    try:
        paths = get_paths()
        append_perf_record(
            paths.results_monitoring_dir / "api_perf.jsonl",
            PerfRecord(
                ts=time.time(),
                endpoint="/predict",
                latency_ms=float(latency_ms),
                fraud_probability=float(resp.fraud_probability),
                prediction=int(resp.prediction),
            ),
        )
    except Exception:
        pass
    return resp


# ---- Legacy endpoints (kept for backward compatibility) ----


class ScoreByIdRequest(BaseModel):
    transaction_id: str


class ScoreByFeaturesRequest(BaseModel):
    features: Dict[str, Any]


@app.post("/score")
def score_by_id(request: ScoreByIdRequest):
    features = fs_get(request.transaction_id)
    if not features:
        raise HTTPException(status_code=404, detail="Transaction not found in feature store")
    return score_transaction(features)


@app.post("/score/features")
def score_by_features(request: ScoreByFeaturesRequest):
    return score_transaction(request.features)


@app.get("/health")
def health():
    return {"status": "ok"}

