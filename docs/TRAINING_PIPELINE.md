# Low-Memory Training Pipeline Design

This document describes the architecture of the training pipeline for credit fraud detection under **hardware constraints** (e.g. 8GB RAM, CPU-focused). The design ensures all datasets in `data/` can be used without loading everything into memory, with intermediate results saved to disk and clear progress reporting.

---

## Goals

- Use **all datasets** in `data/` (via existing `dataset_loader` specs).
- Avoid loading all data into RAM at once (**dataset-by-dataset** or **chunk-based** processing).
- **Save intermediates** (feature matrices, sample manifest) to disk.
- **Combine results** after training (aggregated metrics, optional per-dataset evaluation).
- **Progress indicators**: pipeline stages and progress bars for CV/model training.
- **Simple and portfolio-ready**: fits existing `src/` layout and looks professional.

---

## Architecture Overview

The pipeline has **four stages**. Only one dataset (or one chunk) is in memory at a time during extraction; during training, only a **stratified sample** of bounded size is loaded.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: EXTRACT (per-dataset, bounded memory)                           │
│  For each dataset: load → canonicalize → build features → write parquet  │
│  Output: data/processed/features_<name>.parquet + results/training/index │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: SAMPLE (stream over parquets, bounded memory)                 │
│  Read each parquet; stratified sample by (Class, dataset_name); merge    │
│  Output: in-memory DataFrame (cap e.g. 80k rows) or train_sample.parquet  │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 3: TRAIN (single in-memory sample)                                │
│  Stratified CV, existing models (LogReg, RF, XGBoost); save to registry  │
│  Output: models/, results/model_benchmark.csv, results/plots/            │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 4: AGGREGATE (optional, per-dataset eval)                         │
│  For each dataset parquet: load or stream, score, compute metrics         │
│  Output: results/training/per_dataset_metrics.csv + combined summary    │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Stage 1** keeps RAM bounded by the **largest single dataset** (capped by `max_rows_per_dataset`, e.g. 50k).
- **Stage 2** builds a stratified sample by reading parquets one-by-one and sampling; only the final sample is held (or written to a single small parquet).
- **Stage 3** runs the existing benchmark on that sample (same metrics and model registry).
- **Stage 4** optionally evaluates the registered model on each dataset separately and aggregates metrics.

---

## Recommended Directory Structure

```
credit_fraud_detection/
├── data/
│   ├── *.csv                          # raw datasets (unchanged)
│   └── processed/                     # NEW: pipeline intermediates
│       ├── features_creditcard.parquet
│       ├── features_creditcard_2023.parquet
│       ├── features_transactions.parquet
│       └── ...
│
├── results/
│   ├── model_benchmark.csv            # existing
│   ├── plots/                         # existing
│   └── training/                      # NEW: pipeline run metadata
│       ├── index.json                 # list of parquet paths + row/class counts
│       ├── train_sample.parquet       # optional: saved sample for reproducibility
│       ├── per_dataset_metrics.csv    # optional: Stage 4 output
│       └── pipeline_run.json          # last run config + stage timings
│
├── src/
│   ├── data_pipeline/                 # existing; add optional chunked iterator
│   ├── features/                      # existing
│   ├── models/                        # existing; add run_benchmark_from_arrays
│   └── training/                      # NEW
│       ├── __init__.py
│       ├── config.py                  # LowMemoryTrainingConfig
│       └── pipeline.py                # run_pipeline(), stage functions, progress
│
├── docs/
│   └── TRAINING_PIPELINE.md           # this document
│
└── scripts/
    └── run_training_pipeline.py       # CLI entrypoint (optional)
```

No new datasets or feature names are introduced; only existing loader specs, canonical schema, and `build_unified_model_frame` are used.

---

## Key Design Choices

| Concern | Choice | Rationale |
|--------|--------|-----------|
| Memory bound | Per-dataset row cap (`max_rows_per_dataset`) | One dataset in RAM at a time; 50k rows × ~100 cols × 8 bytes ≈ 40MB per dataset. |
| Feature consistency | Same `build_unified_model_frame()` per dataset | Behavioral/temporal features are consistent within each file; schema aligned when merging. |
| Training set size | Stratified sample (e.g. 80k rows) from all parquets | Fits in 8GB; preserves class balance; one unified model. |
| Combining results | Single model + optional per-dataset metrics CSV | One production model; optional breakdown by dataset for analysis. |
| Progress | Stage names + tqdm over datasets/folds/models | Clear pipeline stages and per-step progress bars. |

---

## Usage (Example)

```bash
# Full pipeline: extract → sample → train → optional aggregate
python -m src.training.pipeline --max-rows-per-dataset 50000 --sample-size 80000 --splits 3

# Extract only (writes parquets)
python -m src.training.pipeline --stage extract --max-rows-per-dataset 50000

# Train from existing parquets (skip extract)
python -m src.training.pipeline --stage sample --stage train --sample-size 80000
```

---

## Progress and Logging

- **Stage headers**: `[STAGE 1/4] EXTRACT`, `[STAGE 2/4] SAMPLE`, etc. (via `logging.info`).
- **Extract**: `tqdm` over datasets (`Extract datasets: 0%|...| 8/8`).
- **Sample**: `tqdm` over parquet files (`Sample from parquets: 0%|...| 7/7`).
- **Train**: `tqdm` over models (`Models: 0%|...| 3/3`), then per-model bar over CV folds (`logreg folds: 0%|...| 3/3`). Implemented in `src/models/train.py` and used by `run_benchmark_from_arrays`.
- **Aggregate**: `tqdm` over index entries (`Aggregate: 0%|...| 7/7`).
- Use `logging` at INFO for stage start/end and paths; `tqdm` for iterative progress.

---

## Limitations (8GB RAM)

- **Per-dataset cap**: Very large files (e.g. 300k+ rows) are truncated to `max_rows_per_dataset` so that one dataset still fits in RAM during extract. Increasing the cap on a 8GB machine may cause OOM.
- **Behavioral features**: Built per dataset in memory; no cross-dataset user aggregation.
- **Stratification**: Stratified sampling is by `(Class, dataset_name)`; if a dataset has very few positives, that dataset may contribute few samples to the training sample.
- **Single model**: Training produces one model (per algorithm) on the sample; optional Stage 4 evaluates it on each dataset separately for reporting only.
