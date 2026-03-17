"""Configuration for the low-memory training pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LowMemoryTrainingConfig:
    """
    Settings for running the training pipeline on a low-resource machine.
    """

    # Stage 1: per-dataset row cap (only one dataset in RAM at a time)
    max_rows_per_dataset: int = 50_000

    # Stage 2: total rows to use for training (stratified sample from all parquets)
    sample_size: int = 80_000

    # Stage 3: same as BenchmarkConfig
    n_splits: int = 3
    random_state: int = 42
    hpo: bool = False
    hpo_iter: int = 8

    # Paths (relative to project paths)
    processed_dir_name: str = "processed"
    training_dir_name: str = "training"
    features_prefix: str = "features_"
    index_filename: str = "index.json"
    sample_filename: str = "train_sample.parquet"
    per_dataset_metrics_filename: str = "per_dataset_metrics.csv"
