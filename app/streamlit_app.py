"""
Fraud Analyst Console — multi-page Streamlit application.

Pages:
  1. Investigations    – browse and manage fraud cases
  2. Transaction Explorer – submit transactions for on-demand investigation
  3. Drift Dashboard   – feature drift monitoring

Run:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Fraud Agent Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Page navigation ──────────────────────────────────────────────────────────
PAGES = {
    "🔍 Investigations":       "app.pages.investigations",
    "🔎 Transaction Explorer": "app.pages.transaction_explorer",
    "📊 Drift Dashboard":      "app.pages.drift_dashboard",
}

with st.sidebar:
    st.markdown("# 🛡️ Fraud Agent Console")
    st.markdown("---")
    page_name = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

# Load and render the selected page module
import importlib
module = importlib.import_module(PAGES[page_name])
module.render()
