#!/usr/bin/env python
"""
CLI entrypoint for the low-memory training pipeline.
Run from project root: python scripts/run_training_pipeline.py [options]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing src when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

from src.training.config import LowMemoryTrainingConfig
from src.training.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Low-memory training pipeline")
    parser.add_argument("--max-rows-per-dataset", type=int, default=50_000)
    parser.add_argument("--sample-size", type=int, default=80_000)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--hpo-iter", type=int, default=8)
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=["extract", "sample", "train", "aggregate"],
    )
    args = parser.parse_args()
    cfg = LowMemoryTrainingConfig(
        max_rows_per_dataset=args.max_rows_per_dataset,
        sample_size=args.sample_size,
        n_splits=args.splits,
        hpo=args.hpo,
        hpo_iter=args.hpo_iter,
    )
    stages = args.stages if args.stages else None
    agg = run_pipeline(cfg=cfg, stages=stages)
    if not agg.empty:
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
