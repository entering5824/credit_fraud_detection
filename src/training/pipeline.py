"""
Low-memory training pipeline: extract features per dataset to disk, sample, train, optional aggregate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.paths import get_paths
from src.data_pipeline.dataset_loader import (
    default_all_dataset_specs,
    iter_datasets,
    validate_dataset_schema,
)
from src.features.unified_features import build_unified_model_frame
from src.models.train import BenchmarkConfig, run_benchmark_from_arrays

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):  # noqa: D103
        return iterable

from sklearn.model_selection import train_test_split

from src.training.config import LowMemoryTrainingConfig

logger = logging.getLogger(__name__)

# Columns we attach for stratification and later drop before training
META_COLS = ["Class", "dataset_name"]


def _stage_extract(cfg: LowMemoryTrainingConfig, paths) -> list[Path]:
    """
    Stage 1: Load each dataset one-by-one, canonicalize, build features, write parquet.
    Only labeled datasets with Class in {0,1} are written. Returns list of written paths.
    """
    logger.info("[STAGE 1/4] EXTRACT — per-dataset feature build to disk")
    processed_dir = paths.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    specs = default_all_dataset_specs()
    written: list[Path] = []
    index_entries: list[dict] = []

    for spec, df in tqdm(
        iter_datasets(specs=specs, max_rows_per_dataset=cfg.max_rows_per_dataset),
        desc="Extract datasets",
        total=len([s for s in specs if s.path.exists()]),
    ):
        if not spec.labeled:
            continue
        df = df[df["Class"].isin([0, 1])].copy()
        if df.empty:
            continue
        try:
            validate_dataset_schema(df)
        except ValueError:
            continue
        X_df = build_unified_model_frame(df)
        X_df["Class"] = df["Class"].values
        X_df["dataset_name"] = df["dataset_name"].values
        out_path = processed_dir / f"{cfg.features_prefix}{spec.dataset_name}.parquet"
        X_df.to_parquet(out_path, index=False)
        written.append(out_path)
        n_pos = int((df["Class"] == 1).sum())
        index_entries.append({
            "path": str(out_path),
            "dataset_name": spec.dataset_name,
            "rows": len(X_df),
            "n_pos": n_pos,
            "n_neg": len(X_df) - n_pos,
        })

    index_path = paths.results_dir / cfg.training_dir_name / cfg.index_filename
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump({"parquets": index_entries}, f, indent=2)
    logger.info("Wrote %s parquets and index to %s", len(written), index_path)
    return written


def _stage_sample(
    cfg: LowMemoryTrainingConfig,
    paths,
    parquet_paths: list[Path],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Stage 2: Stratified sample from parquets up to sample_size. Returns (sample_df, feature_names).
    """
    logger.info("[STAGE 2/4] SAMPLE — stratified sample from disk")
    if not parquet_paths:
        raise FileNotFoundError("No parquet files found; run extract stage first.")
    index_path = paths.results_dir / cfg.training_dir_name / cfg.index_filename
    with open(index_path) as f:
        index = json.load(f)
    entries = index["parquets"]
    total_rows = sum(e["rows"] for e in entries)
    target = min(cfg.sample_size, total_rows)
    rng = np.random.default_rng(cfg.random_state)
    chunks: list[pd.DataFrame] = []

    for entry in tqdm(entries, desc="Sample from parquets"):
        path = Path(entry["path"])
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        n = len(df)
        if n == 0:
            continue
        n_take = max(1, int(round(target * n / total_rows))) if total_rows else min(n, target)
        n_take = min(n_take, n)
        if n_take >= n:
            chunks.append(df)
            continue
        _, sub = train_test_split(
            df, train_size=n_take, stratify=df["Class"], random_state=cfg.random_state
        )
        chunks.append(sub)

    sample_df = pd.concat(chunks, ignore_index=True)
    if len(sample_df) > target:
        sample_df, _ = train_test_split(
            sample_df,
            train_size=target,
            stratify=sample_df["Class"],
            random_state=cfg.random_state,
        )

    feature_cols = [c for c in sample_df.columns if c not in META_COLS]
    sample_path = paths.results_dir / cfg.training_dir_name / cfg.sample_filename
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(sample_path, index=False)
    logger.info("Sample shape %s written to %s", sample_df.shape, sample_path)
    return sample_df, feature_cols


def _stage_train(
    sample_df: pd.DataFrame,
    feature_names: list[str],
    cfg: LowMemoryTrainingConfig,
) -> pd.DataFrame:
    """
    Stage 3: Run benchmark on the sampled data and register models.
    """
    logger.info("[STAGE 3/4] TRAIN — stratified CV and model registration")
    X = sample_df[feature_names].to_numpy(dtype=float)
    y = sample_df["Class"].astype(int).to_numpy()
    dataset_names = sample_df["dataset_name"].astype(str).to_numpy()
    bench_cfg = BenchmarkConfig(
        n_splits=cfg.n_splits,
        random_state=cfg.random_state,
        max_rows=None,
        add_temporal=True,
        add_behavior=True,
        hpo=cfg.hpo,
        hpo_iter=cfg.hpo_iter,
    )
    agg = run_benchmark_from_arrays(
        X, y, feature_names, bench_cfg, dataset_names=dataset_names
    )
    return agg


def _stage_aggregate(
    cfg: LowMemoryTrainingConfig,
    paths,
    parquet_paths: list[Path],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Stage 4 (optional): Score each dataset parquet with the registered model, compute metrics.
    Requires a single model in registry (e.g. the one just trained). Simplified: we just
    list per-dataset row counts and optionally run evaluation in a follow-up script.
    """
    logger.info("[STAGE 4/4] AGGREGATE — per-dataset summary")
    index_path = paths.results_dir / cfg.training_dir_name / cfg.index_filename
    if not index_path.exists():
        return pd.DataFrame()
    with open(index_path) as f:
        index = json.load(f)
    rows = []
    for entry in tqdm(index["parquets"], desc="Aggregate"):
        rows.append({
            "dataset_name": entry["dataset_name"],
            "rows": entry["rows"],
            "n_pos": entry["n_pos"],
            "n_neg": entry["n_neg"],
        })
    out_df = pd.DataFrame(rows)
    out_path = paths.results_dir / cfg.training_dir_name / cfg.per_dataset_metrics_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote per-dataset summary to %s", out_path)
    return out_df


def run_pipeline(
    cfg: LowMemoryTrainingConfig | None = None,
    stages: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the low-memory training pipeline. Stages: extract, sample, train, aggregate.
    If stages is None, run all. Otherwise run only the given stages (e.g. ["extract", "sample", "train"]).
    Returns the benchmark aggregate DataFrame from the train stage.
    """
    cfg = cfg or LowMemoryTrainingConfig()
    stages = stages or ["extract", "sample", "train", "aggregate"]
    paths = get_paths()
    paths.ensure_dirs()

    parquet_paths: list[Path] = []
    if "extract" in stages:
        parquet_paths = _stage_extract(cfg, paths)
    else:
        processed_dir = paths.data_processed_dir
        if processed_dir.exists():
            parquet_paths = list(processed_dir.glob(f"{cfg.features_prefix}*.parquet"))
        index_path = paths.results_dir / cfg.training_dir_name / cfg.index_filename
        if index_path.exists():
            with open(index_path) as f:
                parquet_paths = [Path(e["path"]) for e in json.load(f).get("parquets", [])]

    sample_df = None
    feature_names: list[str] = []
    if "sample" in stages:
        sample_df, feature_names = _stage_sample(cfg, paths, parquet_paths)
    else:
        sample_path = paths.results_dir / cfg.training_dir_name / cfg.sample_filename
        if sample_path.exists():
            sample_df = pd.read_parquet(sample_path)
            feature_names = [c for c in sample_df.columns if c not in META_COLS]

    agg = pd.DataFrame()
    if "train" in stages and sample_df is not None and feature_names:
        agg = _stage_train(sample_df, feature_names, cfg)
    if "aggregate" in stages:
        _stage_aggregate(cfg, paths, parquet_paths, feature_names)

    return agg


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Low-memory training pipeline")
    parser.add_argument("--max-rows-per-dataset", type=int, default=50_000)
    parser.add_argument("--sample-size", type=int, default=80_000)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--hpo-iter", type=int, default=8)
    parser.add_argument("--stage", action="append", dest="stages", choices=["extract", "sample", "train", "aggregate"])
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
