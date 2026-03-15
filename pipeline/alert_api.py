"""
Alert API: score transaction by transaction_id (from feature store) or by raw features.
Kafka -> Feature Store -> Fraud Model -> Alert API
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from pipeline.feature_store import get as fs_get
from pipeline.scorer import score as score_transaction

app = FastAPI(title="Fraud Alert API", version="1.0.0")


class ScoreByIdRequest(BaseModel):
    transaction_id: str


class ScoreByFeaturesRequest(BaseModel):
    features: Dict[str, Any]


@app.post("/score")
def score(request: ScoreByIdRequest):
    """Score by transaction_id (features must already be in feature store)."""
    features = fs_get(request.transaction_id)
    if not features:
        raise HTTPException(status_code=404, detail="Transaction not found in feature store")
    return score_transaction(features)


@app.post("/score/features")
def score_by_features(request: ScoreByFeaturesRequest):
    """Score by raw features (V1..V28, Amount). No Kafka/feature store required."""
    return score_transaction(request.features)


@app.get("/health")
def health():
    return {"status": "ok"}
