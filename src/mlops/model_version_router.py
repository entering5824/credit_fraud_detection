"""
Model Version Router — routes scoring requests across production/shadow/A-B models.

Deployment modes
----------------
SINGLE      Production model only (default).
SHADOW      Production scores are returned; shadow runs in parallel for comparison.
AB_TEST     Traffic split between two model versions by percentage.

Shadow deployment pattern
-------------------------
  FraudEvent → ModelRouter
                ├─ production (v1.0) → score used in report
                └─ shadow    (v1.1) → score logged for offline comparison only

A/B test pattern
----------------
  FraudEvent → ModelRouter
                ├─ version A  (80% traffic) → score returned
                └─ version B  (20% traffic) → score returned

Report enrichment
-----------------
Every score result from the router includes:
  model_version
  model_stage
  model_latency_ms
  shadow_score        (present only in SHADOW mode)
  ab_variant          (present only in AB_TEST mode)
"""

from __future__ import annotations

import hashlib
import logging
import time
from enum import Enum
from typing import Any, Optional

from src.mlops.model_loader import load
from src.mlops.model_registry import ModelRegistry, ModelVersion, get_registry
from src.schemas.feature_schema import BASE_FEATURES

logger = logging.getLogger(__name__)


class RoutingMode(str, Enum):
    SINGLE   = "single"
    SHADOW   = "shadow"
    AB_TEST  = "ab_test"


class ModelVersionRouter:
    """
    Routes fraud scoring calls to the appropriate model version.

    Parameters
    ----------
    model_name   : name of the model family (default: "xgboost")
    mode         : RoutingMode (SINGLE | SHADOW | AB_TEST)
    ab_split     : fraction of traffic routed to version B (0.0–1.0)
    registry     : override the default registry
    """

    def __init__(
        self,
        model_name: str = "xgboost",
        mode: RoutingMode = RoutingMode.SINGLE,
        ab_split: float = 0.20,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._name    = model_name
        self._mode    = mode
        self._split   = ab_split
        self._reg     = registry or get_registry()

    # ------------------------------------------------------------------ #
    # Public scoring API
    # ------------------------------------------------------------------ #

    def score(
        self,
        features: dict[str, Any],
        threshold: float = 0.5,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Score *features* using the active routing strategy.

        Returns a dict with at minimum:
          fraud_probability, prediction, threshold_used,
          model_version, model_stage, model_latency_ms
        """
        if self._mode == RoutingMode.SHADOW:
            return self._shadow_score(features, threshold)
        if self._mode == RoutingMode.AB_TEST:
            return self._ab_score(features, threshold, request_id)
        return self._single_score(features, threshold)

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #

    def _single_score(
        self, features: dict, threshold: float
    ) -> dict[str, Any]:
        prod = self._get_production_version()
        if prod is None:
            raise RuntimeError("No production model registered")
        return self._run_score(prod, features, threshold)

    def _shadow_score(
        self, features: dict, threshold: float
    ) -> dict[str, Any]:
        prod    = self._get_production_version()
        shadow  = self._get_shadow_version()
        if prod is None:
            raise RuntimeError("No production model registered")

        result = self._run_score(prod, features, threshold)

        if shadow is not None:
            try:
                shadow_result = self._run_score(shadow, features, threshold)
                result["shadow_score"]   = shadow_result["fraud_probability"]
                result["shadow_version"] = shadow.version
                logger.debug(
                    "Shadow score: prod=%.3f shadow=%.3f",
                    result["fraud_probability"],
                    shadow_result["fraud_probability"],
                )
            except Exception as exc:
                logger.warning("Shadow model scoring failed: %s", exc)

        return result

    def _ab_score(
        self, features: dict, threshold: float, request_id: Optional[str]
    ) -> dict[str, Any]:
        prod   = self._get_production_version()
        shadow = self._get_shadow_version()
        if prod is None:
            raise RuntimeError("No production model registered")

        # Deterministic bucket assignment via request_id hash
        in_b_bucket = False
        if shadow is not None:
            bucket_key = (request_id or str(features.get("Amount", 0))).encode()
            bucket_val = int(hashlib.md5(bucket_key).hexdigest(), 16) % 100
            in_b_bucket = bucket_val < int(self._split * 100)

        chosen = shadow if (in_b_bucket and shadow is not None) else prod
        result = self._run_score(chosen, features, threshold)
        result["ab_variant"] = "B" if in_b_bucket else "A"
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_score(
        self, mv: ModelVersion, features: dict, threshold: float
    ) -> dict[str, Any]:
        t0 = time.time()
        try:
            model, scaler, feature_names, _ = load(self._name, mv.version, self._reg)
        except Exception:
            # Legacy fallback: load directly from file
            from src.models.model_registry import load_artifacts_from_registry
            model, scaler, feature_names, _ = load_artifacts_from_registry(self._name)

        import numpy as np
        row = np.array([[float(features.get(k, 0.0)) for k in (feature_names or list(BASE_FEATURES))]])
        if scaler is not None:
            row = scaler.transform(row)

        prob = float(model.predict_proba(row)[0][1])
        elapsed = round((time.time() - t0) * 1000, 2)

        return {
            "fraud_probability": prob,
            "prediction":        int(prob >= threshold),
            "threshold_used":    threshold,
            "model_version":     mv.version,
            "model_stage":       mv.stage,
            "model_latency_ms":  elapsed,
        }

    def _get_production_version(self) -> Optional[ModelVersion]:
        return self._reg.get_production(self._name)

    def _get_shadow_version(self) -> Optional[ModelVersion]:
        return self._reg.get_shadow(self._name)


# ---------------------------------------------------------------------------
# Module-level default router
# ---------------------------------------------------------------------------

_default_router: Optional[ModelVersionRouter] = None


def get_router(
    model_name: str = "xgboost",
    mode: RoutingMode = RoutingMode.SINGLE,
) -> ModelVersionRouter:
    global _default_router
    if _default_router is None:
        _default_router = ModelVersionRouter(model_name=model_name, mode=mode)
    return _default_router
