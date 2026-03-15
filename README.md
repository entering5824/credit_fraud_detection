# Credit Fraud Detection

**Credit card transaction fraud detection** — Ensemble models (Random Forest, AdaBoost, XGBoost) with imbalanced handling, notebooks for EDA and training, and CLI/Streamlit app for inference. Metrics: accuracy, precision, recall, F1, AUC.

---

## Problem Statement

- **Real-world problem**: A tiny fraction of transactions are fraudulent; missing them is costly, but too many false positives hurt user experience. Models must balance recall (catch fraud) and precision (avoid blocking legitimate users).
- **Why it matters**: Reduces financial loss and chargebacks; automated scoring supports real-time or batch review. Imbalanced data (e.g. creditcard.csv) requires class weights or resampling.
- **Constraints**: Data from CSV (e.g. creditcard.csv); features often anonymized (V1–V28, Amount, Time); latency for real-time use; users are analysts (notebooks) and operators (app).

---

## System Architecture

```
data/creditcard.csv → src/data_preprocessing → train/val/test
       → src/models (Random Forest, AdaBoost, XGBoost) → saved models
       → src/evaluate → results/metrics.csv
User → app/fraud_detection_app.py (Streamlit or CLI) → load model → predict (fraud probability / label)
```

- **Notebooks**: 01 EDA and preprocessing, 02 model training, 03 evaluation and comparison.
- **App**: Streamlit (or loop_random_fda) for single or batch prediction.
- **No external DB/API**: Data and models on disk; results in results/.

---

## Key Features

### AI Features

- **Model training**: Random Forest, AdaBoost, XGBoost (sklearn + xgboost); class_weight for imbalanced data; training in notebooks and/or src.
- **Inference**: Probability and binary label; app and CLI for single/batch.
- **Evaluation metrics**: Accuracy, precision, recall, F1, AUC (src/evaluate.py); results in results/metrics.csv.
- **Explainability**: Tree-based models support feature importance (see notebooks/source).

### Application Features

- **Streamlit app**: Upload or input transactions; view prediction and metrics.
- **Notebooks**: Full pipeline from EDA to comparison (Logistic Regression, RF, AdaBoost, XGBoost).
- **Export**: metrics.csv for model comparison.

### Engineering Features

- **Modular**: src/data_preprocessing, src/models, src/evaluate; app entry points.
- **Reproducibility**: Random state and train/val/test split.

---

## Model & Methodology

- **Algorithms**: Random Forest, AdaBoost (Decision Tree base), XGBoost; Logistic Regression in comparison.
- **Loss**: Classification (cross-entropy); class_weight='balanced' or custom for imbalance.
- **Evaluation metrics**: Accuracy, precision, recall, F1, AUC. Example from results/metrics.csv: RF accuracy ≈ 0.999, F1 ≈ 0.80; XGBoost AUC ≈ 0.98.

---

## Results

- **Metrics** (example from results/metrics.csv): Random Forest — accuracy ≈ 0.999, precision ≈ 0.89, recall ≈ 0.73, F1 ≈ 0.80, AUC ≈ 0.96; XGBoost — accuracy ≈ 0.999, F1 ≈ 0.77, AUC ≈ 0.98. Exact values depend on split and tuning.
- **Latency**: Per-transaction inference is fast (tree ensembles).
- *(Run notebooks or training script and evaluate to regenerate results/metrics.csv.)*

---

## Project Structure

```
credit_fraud_detection/
├── data/              # creditcard.csv
├── src/               # data_preprocessing.py, models.py, evaluate.py
├── notebooks/         # 01_EDA_and_Preprocessing, 02_Model_Training, 03_Evaluation_and_Comparison
├── app/               # fraud_detection_app.py, loop_random_fda.py
├── results/           # metrics.csv
├── requirements.txt
└── README.md
```

---

## Installation

### Backend

```bash
pip install -r requirements.txt
```

Place creditcard.csv (or equivalent) in data/ as expected by preprocessing.

---

## Usage

### Notebooks

Open and run 01 → 02 → 03 for EDA, training, and comparison.

### App

```bash
python app/fraud_detection_app.py
```

Use UI for single or batch prediction.

---

## Demo

*(Add screenshot of Streamlit app or metrics table.)*

---

## Deployment

- **Local/VM**: Run Streamlit or wrap model in FastAPI for API deployment.
- **Docker**: Add Dockerfile (install deps, run app or API) if needed.

---

## Future Improvements

- Real-time API with threshold tuning endpoint.
- Model versioning and A/B testing.
- More ensembles (e.g. stacking) and hyperparameter search.
- Drift detection and retraining pipeline.
