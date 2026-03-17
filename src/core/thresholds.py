from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.cost_sensitive import load_cost_config


@dataclass(frozen=True)
class ThresholdConfig:
    optimal_threshold: float = 0.5
    cost_fn: float = 1000.0
    cost_fp: float = 10.0


def load_threshold_config(config_path: Optional[Path] = None) -> ThresholdConfig:
    cfg = load_cost_config(config_path=config_path)
    return ThresholdConfig(
        optimal_threshold=float(cfg.get("optimal_threshold", 0.5)),
        cost_fn=float(cfg.get("cost_fn", 1000.0)),
        cost_fp=float(cfg.get("cost_fp", 10.0)),
    )

