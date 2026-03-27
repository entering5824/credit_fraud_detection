"""
Model Loader — caches loaded model artifacts per (name, version).

Wraps ModelRegistry.load_artifacts() with an in-process LRU cache so each
model version is loaded from disk exactly once per process lifetime.

Also provides warm_start() for use in ASGI lifespan hooks.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

from src.mlops.model_registry import ModelRegistry, ModelVersion, get_registry

logger = logging.getLogger(__name__)

# in-process cache: (name, version) → (model, scaler, features, meta)
_ARTIFACT_CACHE: dict[tuple, tuple] = {}


def load(
    name: str,
    version: str,
    registry: Optional[ModelRegistry] = None,
    force_reload: bool = False,
) -> tuple[Any, Any, list[str], ModelVersion]:
    """
    Load (model, scaler, feature_names, ModelVersion) — cached per version.

    Parameters
    ----------
    name         : model name (e.g. "xgboost")
    version      : version string (e.g. "1.0.0")
    registry     : override the default module-level registry
    force_reload : bypass cache (e.g. after hot-swap)
    """
    reg = registry or get_registry()
    key = (name, version)
    if not force_reload and key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[key]

    logger.info("Loading model %s version=%s from registry", name, version)
    artifacts = reg.load_artifacts(name, version)
    _ARTIFACT_CACHE[key] = artifacts
    return artifacts


def evict(name: str, version: str) -> None:
    """Remove a version from the in-process cache."""
    _ARTIFACT_CACHE.pop((name, version), None)


def evict_all() -> None:
    _ARTIFACT_CACHE.clear()


def warm_start(
    name: str = "xgboost",
    registry: Optional[ModelRegistry] = None,
) -> None:
    """
    Pre-load all non-archived versions of *name* at process startup.
    Call from ASGI lifespan to eliminate cold-start latency.
    """
    reg = registry or get_registry()
    for mv in reg.list_versions(name):
        if mv.stage != "archived":
            try:
                load(name, mv.version, registry=reg)
                logger.info("Warm-started model %s v%s (%s)", name, mv.version, mv.stage)
            except Exception as exc:
                logger.warning("Warm-start failed for %s v%s: %s", name, mv.version, exc)
