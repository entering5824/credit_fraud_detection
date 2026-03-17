from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.core.paths import get_paths


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _default_run_id() -> str:
    # Short, filesystem-friendly id.
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunInfo:
    run_id: str
    name: str
    started_at: str
    ended_at: Optional[str] = None
    status: str = "running"
    host: str = ""
    user: str = ""
    git_commit: Optional[str] = None


class ExperimentTracker:
    """
    Minimal experiment tracker that writes a single `run.json` plus optional artifacts.

    Layout:
      results/experiments/<run_id>/
        run.json
        metrics.json
        params.json
        artifacts/...
    """

    def __init__(self, run_name: str, run_id: Optional[str] = None, root: Optional[Path] = None):
        paths = get_paths(root=root)
        self.base_dir = paths.results_dir / "experiments" / (run_id or _default_run_id())
        self.artifacts_dir = self.base_dir / "artifacts"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.info = RunInfo(
            run_id=self.base_dir.name,
            name=run_name,
            started_at=_utcnow(),
            host=socket.gethostname(),
            user=os.environ.get("USERNAME") or os.environ.get("USER") or "",
            git_commit=_try_git_commit(paths.root),
        )
        self._write_json("run.json", asdict(self.info))
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)
        self._write_json("params.json", self._params)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics.update(metrics)
        self._write_json("metrics.json", self._metrics)

    def log_artifact_path(self, path: Path, name: Optional[str] = None) -> Path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        target = self.artifacts_dir / (name or path.name)
        # Copy bytes (avoid shutil to keep behavior explicit).
        target.write_bytes(path.read_bytes())
        return target

    def end(self, status: str = "finished") -> None:
        self.info.ended_at = _utcnow()
        self.info.status = status
        self._write_json("run.json", asdict(self.info))

    def _write_json(self, rel: str, payload: dict[str, Any]) -> None:
        p = self.base_dir / rel
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)


def _try_git_commit(root: Path) -> Optional[str]:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            ref_path = root / ".git" / ref.replace("ref:", "").strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        return ref
    except Exception:
        return None

