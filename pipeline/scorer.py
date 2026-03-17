"""
Legacy shim (kept for backward compatibility).

New canonical implementation lives in `src.models.inference`.
"""

from src.models.inference import score as _score_new


def score(features: dict) -> dict:
    """
    Backward compatible wrapper.

    Historical response used `fraud_score`; the new API uses `fraud_probability`.
    """
    res = _score_new(features)
    return {
        "fraud_score": res["fraud_probability"],
        "fraud_probability": res["fraud_probability"],
        "alert": res["alert"],
        "threshold_used": res["threshold_used"],
    }
