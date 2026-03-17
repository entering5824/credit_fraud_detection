"""
Legacy Streamlit entrypoint (kept for backward compatibility).

New canonical dashboard lives in `src.dashboard.streamlit_app`.

Run:
  streamlit run app/streamlit_app.py
"""

# Streamlit expects a file entrypoint; importing runs the app.
import src.dashboard.streamlit_app  # noqa: F401
