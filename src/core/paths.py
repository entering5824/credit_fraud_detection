from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    data_raw_dir: Path
    data_processed_dir: Path
    models_dir: Path
    results_dir: Path
    results_plots_dir: Path
    results_monitoring_dir: Path

    def ensure_dirs(self) -> None:
        for p in [
            self.data_dir,
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.results_dir,
            self.results_plots_dir,
            self.results_monitoring_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    # `src/core/paths.py` -> `src` -> project root
    return Path(__file__).resolve().parents[2]


def get_paths(root: Path | None = None) -> ProjectPaths:
    root = get_project_root() if root is None else Path(root).resolve()
    data_dir = root / "data"
    results_dir = root / "results"
    return ProjectPaths(
        root=root,
        data_dir=data_dir,
        data_raw_dir=data_dir / "raw",
        data_processed_dir=data_dir / "processed",
        models_dir=root / "models",
        results_dir=results_dir,
        results_plots_dir=results_dir / "plots",
        results_monitoring_dir=results_dir / "monitoring",
    )

