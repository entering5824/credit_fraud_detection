"""
Explainability dashboard: global feature importance + local SHAP for sample transactions.
Run from project root: streamlit run app/explain_dashboard.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.data_preprocessing import scale_features, split_data
from src.explain import global_importance, local_shap_values, FEATURE_COLS

st.set_page_config(page_title="Fraud Explainability", layout="wide")
st.title("Explainability: Feature importance & SHAP")

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv(project_root / "data" / "creditcard.csv")
    X = df[FEATURE_COLS]
    y = df["Class"]
    X_scaled, scaler = scale_features(X, feature_cols=FEATURE_COLS, fit=True)
    _, _, X_test, _, _, y_test = split_data(X_scaled, y, test_size=0.15, val_size=0.15, random_state=42)
    with open(project_root / "models" / "xgboost.pkl", "rb") as f:
        import pickle
        model = pickle.load(f)
    X_test_np = X_test.values if hasattr(X_test, "values") else X_test
    return model, X_test_np, y_test.values if hasattr(y_test, "values") else y_test, scaler

try:
    model, X_test, y_test, scaler = load_data_and_model()
except Exception as e:
    st.error(f"Load error: {e}. Train models first (notebook 02).")
    st.stop()

# Global importance
st.header("Global feature importance")
names, imp = global_importance(model, X_test[:500])
imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
st.bar_chart(imp_df.set_index("feature")["importance"])

# Local SHAP: pick a sample
st.header("Local SHAP (why this transaction?)")
idx = st.slider("Test set index", 0, min(100, len(X_test) - 1), 0)
sample = X_test[idx : idx + 1]
shap_vals = local_shap_values(sample, model=model)[0]
shap_df = pd.DataFrame({"feature": FEATURE_COLS, "SHAP": shap_vals}).sort_values("SHAP", key=abs, ascending=False)
st.bar_chart(shap_df.set_index("feature")["SHAP"])
st.caption(f"True label: {'Fraud' if y_test[idx] == 1 else 'Normal'}")
