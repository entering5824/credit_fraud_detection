from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from src.core.paths import get_paths
from src.data_pipeline.dataset_loader import load_dataset
from src.explainability.shap_explainer import explain_transaction
from src.features.feature_engineering import build_features
from src.models.inference import score as score_transaction


st.set_page_config(page_title="Fraud Investigation Dashboard", layout="wide")
st.title("Fraud Investigation Dashboard")

paths = get_paths()


@st.cache_data(show_spinner=False)
def load_scoring_frame(max_rows: int) -> pd.DataFrame:
    df = load_dataset()
    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    # Add engineered features for investigation context (even if the legacy model doesn't consume them yet).
    feats = build_features(df.drop(columns=["Class"]))
    feats["Class"] = df["Class"].astype(int).to_numpy()
    return feats


with st.sidebar:
    st.header("Controls")
    max_rows = st.number_input("Rows to load", min_value=1000, max_value=100000, value=20000, step=1000)
    threshold = st.slider("Alert threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    top_k = st.slider("Top SHAP features", min_value=3, max_value=10, value=5, step=1)
    show_only_alerts = st.checkbox("Show only alerts", value=True)


df = load_scoring_frame(int(max_rows))

# Score (fast path) using legacy-compatible inference
base_feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_data(show_spinner=False)
def score_all(df_in: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probs = []
    preds = []
    for _, r in df_in.iterrows():
        features = {k: float(r.get(k, 0.0)) for k in base_feature_names}
        s = score_transaction(features, threshold=threshold)
        probs.append(float(s["fraud_probability"]))
        preds.append(int(s["alert"]))
    out = df_in.copy()
    out["fraud_probability"] = probs
    out["prediction"] = preds
    return out


try:
    scored = score_all(df, float(threshold))
except FileNotFoundError as e:
    st.warning(
        "No trained model artifacts found. Train a model first, then restart the dashboard.\n\n"
        f"Details: {e}\n\n"
        "Suggested: run `python -m src.models.train --max-rows 50000` to register a model, "
        "or run your legacy notebooks to produce `models/xgboost.pkl` + `models/scaler.pkl`."
    )
    # Keep the dashboard usable (and importable) even when artifacts are missing.
    scored = df.copy()
    scored["fraud_probability"] = 0.0
    scored["prediction"] = 0

if show_only_alerts:
    view = scored[scored["prediction"] == 1].copy()
else:
    view = scored.copy()

view = view.sort_values("fraud_probability", ascending=False).reset_index(drop=True)

colA, colB = st.columns([2, 1])

with colA:
    st.subheader("Suspicious transactions")
    show_cols = (
        ["fraud_probability", "prediction", "Class", "Amount", "Time"]
        + [c for c in ["transactions_last_1h", "transactions_last_24h", "time_since_last_transaction"] if c in view.columns]
        + [c for c in ["is_new_merchant_for_user", "spending_spike_ratio"] if c in view.columns]
    )
    st.dataframe(view[show_cols].head(500), use_container_width=True, height=520)

with colB:
    st.subheader("Quick stats")
    st.metric("Loaded rows", f"{len(scored):,}")
    st.metric("Alerts", f"{int(scored['prediction'].sum()):,}")
    st.metric("Fraud rate (true)", f"{100.0 * float(scored['Class'].mean()):.3f}%")
    st.metric("Avg score", f"{100.0 * float(scored['fraud_probability'].mean()):.2f}%")


st.divider()
st.subheader("Transaction explanation (SHAP)")

if len(view) == 0:
    st.info("No alerts at this threshold. Lower the threshold or disable 'Show only alerts'.")
else:
    idx = st.number_input(
        "Row index in the table above",
        min_value=0,
        max_value=max(0, len(view) - 1),
        value=0,
        step=1,
    )
    row = view.iloc[int(idx)]

    features = {k: float(row.get(k, 0.0)) for k in base_feature_names}
    ex = explain_transaction(features, threshold=float(threshold), top_k=int(top_k))

    left, right = st.columns([1, 1])
    with left:
        st.metric("Fraud probability", f"{100.0 * ex.fraud_probability:.2f}%")
        st.metric("Prediction", "ALERT" if ex.prediction == 1 else "OK")
        st.write(ex.narrative)

    with right:
        top_df = pd.DataFrame(ex.top_features_contributing)
        if not top_df.empty:
            top_df = top_df.assign(abs_shap=top_df["shap_value"].abs()).sort_values(
                "abs_shap", ascending=False
            )
            st.bar_chart(top_df.set_index("feature")["abs_shap"])
            st.dataframe(top_df.drop(columns=["abs_shap"]), use_container_width=True)
        else:
            st.info("No SHAP explanation available for the current model.")


st.caption(
    "Tip: run `python -m src.models.train` to benchmark and register new models; "
    "this dashboard currently scores using the legacy-compatible XGBoost artifacts (`models/xgboost.pkl`)."
)

