"""
MLOps Model Registry — versioned model lifecycle management.

Supports:
  • Multiple model versions stored under models/<name>/<version>/
  • Metadata: stage (staging/production/shadow/archived), accuracy, training date
  • Promotion: staging → production, production → archived
  • Rollback: point production at a previous version instantly

Directory layout
----------------
models/
  xgboost/
    v1.0.0/
      model.pkl
      scaler.pkl
      metadata.json
    v1.1.0/
      model.pkl
      scaler.pkl
      metadata.json
  registry.json      ← active versions per model name

Legacy single-file layout (models/xgboost.pkl) still supported as v0.0.0.
"""

from __future__ import annotations

import json
import pickle
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.paths import get_paths

_STAGES = {"staging", "production", "shadow", "archived"}


# ---------------------------------------------------------------------------
# Metadata model
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    name:         str
    version:      str
    stage:        str          = "staging"
    framework:    str          = "xgboost"
    features:     list[str]    = field(default_factory=list)
    metrics:      dict         = field(default_factory=dict)   # auc, pr_auc, etc.
    artifact_hash: str         = ""                            # SHA-256 of model.pkl
    training_date: str         = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    description:  str          = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    File-backed versioned model registry.

    Parameters
    ----------
    registry_dir : root directory that contains model subdirectories.
                   Defaults to the project models/ directory.
    """

    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self._root = registry_dir or get_paths().models_dir
        self._index_path = self._root / "registry.json"
        self._index: dict[str, dict[str, dict]] = self._load_index()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        version: str,
        model: Any,
        scaler: Any = None,
        features: Optional[list[str]] = None,
        metrics: Optional[dict] = None,
        stage: str = "staging",
        description: str = "",
    ) -> ModelVersion:
        """
        Persist a model artifact and register it in the index.
        """
        ver_dir = self._version_dir(name, version)
        ver_dir.mkdir(parents=True, exist_ok=True)

        # Serialize artifacts
        model_path = ver_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        if scaler is not None:
            with open(ver_dir / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

        artifact_hash = _sha256(model_path)

        mv = ModelVersion(
            name=name,
            version=version,
            stage=stage,
            features=features or [],
            metrics=metrics or {},
            artifact_hash=artifact_hash,
            description=description,
        )
        # Save metadata
        with open(ver_dir / "metadata.json", "w") as f:
            json.dump(mv.to_dict(), f, indent=2)

        self._index.setdefault(name, {})[version] = mv.to_dict()
        self._save_index()
        return mv

    def promote(self, name: str, version: str, target_stage: str) -> ModelVersion:
        """Move a version to a new stage (e.g. staging → production)."""
        if target_stage not in _STAGES:
            raise ValueError(f"Invalid stage '{target_stage}'. Valid: {_STAGES}")
        mv = self._get_version_meta(name, version)
        # Archive current production when promoting a new one
        if target_stage == "production":
            for v, meta in self._index.get(name, {}).items():
                if meta.get("stage") == "production" and v != version:
                    meta["stage"] = "archived"
        mv.stage = target_stage
        self._index[name][version] = mv.to_dict()
        self._save_index()
        # Update metadata file on disk
        meta_path = self._version_dir(name, version) / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "w") as f:
                json.dump(mv.to_dict(), f, indent=2)
        return mv

    def rollback(self, name: str, to_version: str) -> ModelVersion:
        """Shortcut: promote *to_version* straight to production."""
        return self.promote(name, to_version, "production")

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def get_production(self, name: str) -> Optional[ModelVersion]:
        return self._find_by_stage(name, "production")

    def get_shadow(self, name: str) -> Optional[ModelVersion]:
        return self._find_by_stage(name, "shadow")

    def list_versions(self, name: str) -> list[ModelVersion]:
        return [
            ModelVersion.from_dict(meta)
            for meta in self._index.get(name, {}).values()
        ]

    def load_artifacts(
        self, name: str, version: str
    ) -> tuple[Any, Any, list[str], ModelVersion]:
        """Load (model, scaler, feature_names, metadata) for a specific version."""
        mv = self._get_version_meta(name, version)
        ver_dir = self._version_dir(name, version)

        # Try versioned layout first, fall back to legacy
        model_pkl = ver_dir / "model.pkl"
        if not model_pkl.exists():
            return self._load_legacy(name, version, mv)

        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        scaler = None
        scaler_pkl = ver_dir / "scaler.pkl"
        if scaler_pkl.exists():
            with open(scaler_pkl, "rb") as f:
                scaler = pickle.load(f)
        features = mv.features or []
        return model, scaler, features, mv

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _version_dir(self, name: str, version: str) -> Path:
        return self._root / name / version

    def _get_version_meta(self, name: str, version: str) -> ModelVersion:
        meta = self._index.get(name, {}).get(version)
        if meta is None:
            raise KeyError(f"Model '{name}' version '{version}' not found in registry")
        return ModelVersion.from_dict(meta)

    def _find_by_stage(self, name: str, stage: str) -> Optional[ModelVersion]:
        for meta in self._index.get(name, {}).values():
            if meta.get("stage") == stage:
                return ModelVersion.from_dict(meta)
        return None

    def _load_legacy(self, name: str, version: str, mv: ModelVersion):
        """Fall back to top-level models/xgboost.pkl."""
        from src.models.model_registry import load_artifacts_from_registry
        from src.schemas.feature_schema import BASE_FEATURES
        model, scaler, features, _ = load_artifacts_from_registry(name)
        mv.features = mv.features or list(BASE_FEATURES)
        return model, scaler, mv.features, mv

    def _load_index(self) -> dict:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
