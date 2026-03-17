"""
Legacy explainability Streamlit entrypoint (kept for backward compatibility).

The new investigation dashboard includes SHAP explanations and lives at:
  `src.dashboard.streamlit_app`

Run:
  streamlit run app/explain_dashboard.py
"""

import src.dashboard.streamlit_app  # noqa: F401
