# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A production-style machine learning system for detecting fraudulent financial transactions. The project implements a unified data pipeline capable of training across heterogeneous fraud datasets, evaluating models with fraud-specific metrics, and serving predictions through a REST API and interactive dashboard.

---

## Table of Contents

- [Problem](#problem)
- [Approach](#approach)
- [Supported Datasets](#supported-datasets)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [API Service](#api-service)
  - [Dashboard](#dashboard)
  - [Monitoring](#monitoring)
- [License](#license)

---

## Problem

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions often represent less than 1% of all observations. In this setting, **accuracy is a misleading metric**. Evaluation must instead prioritize:

- **PR-AUC** (area under the precision–recall curve)
- **Recall at fixed FPR** (e.g. at 1% and 0.1% false positive rates)
- **Cost-based threshold selection** that balances the operational cost of missed fraud against false alerts

Beyond model quality, production fraud systems face two additional challenges: (1) data comes from **multiple sources with incompatible schemas**, and (2) systems must remain **interpretable and monitored** as transaction distributions evolve over time.

---

## Approach

The system is organized as a modular ML pipeline with six main stages:

1. **Dataset Ingestion** — `src/data_pipeline/dataset_loader.py` normalizes heterogeneous CSV schemas into a canonical format: `Class`, `Time`, `Amount`, and `dataset_name`. This allows multiple datasets to be combined into a single training frame while preserving source provenance.

2. **Feature Engineering** — `src/features/unified_features.py` applies different transformation paths depending on the row type. Rows containing PCA features `V1–V28` (Kaggle-style credit card data) are processed with temporal and behavioral features. Generic transaction rows are processed with clipped numerics and one-hot encoded categoricals. A `feature_source` indicator column marks the transformation path used.

3. **Benchmarking** — Stratified cross-validation is run across Logistic Regression, Random Forest, and XGBoost. Each model is evaluated on PR-AUC, recall at fixed FPR thresholds, and optional cost-sensitive thresholding.

4. **Model Registry** — Trained models and their feature schemas are saved to `models/registry.json`. Inference always loads the frozen feature list from the registry to prevent schema drift.

5. **Inference & Explanation** — A FastAPI service exposes a `POST /predict` endpoint that returns a fraud probability, risk score, prediction label, and a SHAP-based narrative explanation. Batch scoring over CSV files is also available.

6. **Monitoring** — Drift utilities compute PSI and KS statistics per feature and track fraud rate changes over time. A lightweight experiment tracker logs metrics, parameters, and artifacts for each training run.

---

## Supported Datasets

All CSV files are placed in the `data/` directory. Each is mapped to a canonical schema before training. Unlabeled datasets are excluded from training but remain available for inference and demonstration.

| File | Dataset Name | Label Column | Time Column | Note |
|------|-------------|--------------|-------------|------|
| `creditcard.csv` | creditcard | `Class` | `Time` | Kaggle classic; PCA features V1–V28, highly imbalanced |
| `creditcard_2023.csv` | creditcard_2023 | `Class` | `Time` | 2023 version with the same schema |
| `credit_card_fraud_10k.csv` | credit_card_fraud_10k | `is_fraud` | `transaction_hour` | ~10k rows |
| `onlinefraud.csv` | onlinefraud | `isFraud` | `step` | PaySim-style online payments |
| `PS_20174392719_1491204439457_log.csv` | PS_log | `isFraud` | `step` | PaySim simulation log |
| `Synthetic_Financial_datasets_log.csv` | synthetic_log | `isFraud` | `step` | Synthetic financial transactions |
| `transactions.csv` | transactions | `is_fraud` | `transaction_time` | Large file; row limit applied by default |
| `bank_transactions_data_2.csv` | bank_transactions_data_2 | — | `TransactionDate` | **Unlabeled**; inference and demo only |

---

## Architecture

```
Raw CSVs (data/*.csv)
        │
        ▼
Dataset Loader          ← schema normalization, canonicalization
        │
        ▼
Unified Feature Engineering   ← behavioral, temporal, or generic features
        │
        ▼
Model Training & Benchmarking  ← stratified CV, PR-AUC, recall@FPR
        │
        ▼
Model Registry (models/registry.json)
        │
   ┌────┴────┬──────────────────┐
   ▼         ▼                  ▼
REST API   Batch Scoring    Streamlit Dashboard
   │
   ▼
Monitoring & Drift Detection
```

---

## Repository Structure

```
credit_fraud_detection/
├── data/                        # raw datasets (not tracked in git)
│   └── raw/                     # generated demo data
├── src/
│   ├── core/                    # paths and threshold configuration
│   ├── data_pipeline/           # dataset loader, canonicalization, schema validation
│   ├── features/                # unified, behavioral, and temporal feature engineering
│   ├── evaluation/              # PR-AUC, recall@FPR, cost-sensitive thresholds
│   ├── models/                  # training, inference, registry, batch scoring
│   ├── explainability/          # SHAP explainer
│   ├── api/                     # FastAPI prediction service
│   ├── dashboard/               # Streamlit investigation dashboard
│   └── monitoring/              # drift detection, performance monitoring
├── app/                         # Streamlit entrypoint (shim → src.dashboard)
├── results/                     # benchmark CSVs, plots, experiment artifacts
├── models/                      # trained model artifacts (not tracked in git)
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
pip install -r requirements.txt
```

Place at least one labeled dataset (e.g. `creditcard.csv`) in the `data/` directory before training.

---

## Usage

### Training

Run stratified cross-validation benchmarking across all available models:

```bash
python -m src.models.train --max-rows 30000 --splits 3
```

Enable optional hyperparameter search:

```bash
python -m src.models.train --max-rows 30000 --splits 3 --hpo --hpo-iter 8
```

Outputs written to:
- `results/model_benchmark.csv` — per-model metrics summary
- `results/plots/<model>_roc.png` and `results/plots/<model>_pr.png` — evaluation curves
- `models/<model>/<version>/` — model artifacts and feature schema
- `models/registry.json` — model registry

---

### API Service

Start the prediction server:

```bash
uvicorn src.api.main:app --reload
```

**`POST /predict`** — Score a single transaction and return an explanation.

Request:
```json
{
  "features": {
    "Amount": 123.0,
    "V1": 0.0,
    "V2": 0.0,
    "V3": 0.0
  }
}
```

Response:
```json
{
  "fraud_probability": 0.04,
  "risk_score": 4,
  "prediction": 0,
  "explanation": "..."
}
```

---

### Dashboard

Launch the interactive investigation dashboard:

```bash
streamlit run app/streamlit_app.py
```

The dashboard displays flagged transactions sorted by risk score, supports threshold and alert filters, and shows per-transaction SHAP explanations.

---

### Monitoring

Run drift detection between a baseline and a current dataset:

```bash
python -m src.monitoring.drift_detection \
  --baseline data/creditcard.csv \
  --current data/creditcard.csv
```

The script computes PSI and KS statistics per feature and reports fraud rate changes. Output is written to `results/monitoring/drift_report.json`.

---

### Batch Scoring

Score a CSV file and emit a scored output with a summary:

```bash
python -m src.models.batch_scoring --input data/creditcard.csv --output results/scored.csv
```

---

## License

This project is licensed under the [MIT License](LICENSE).
