"""
Drift Dashboard — Streamlit page.

Visualises feature drift between training distribution and current data.
Uses the simulation environment to generate sample current data when no
real data is available.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

_PSI_WARN     = 0.10
_PSI_CRITICAL = 0.20


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions."""
    eps = 1e-6
    buckets = np.percentile(expected, np.linspace(0, 100, bins + 1))
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    exp_counts = np.histogram(expected, buckets)[0]
    act_counts = np.histogram(actual,   buckets)[0]

    exp_pct = exp_counts / (exp_counts.sum() + eps)
    act_pct = act_counts / (act_counts.sum() + eps)

    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def render() -> None:
    st.title("📊 Feature Drift Dashboard")
    st.caption(
        "Compare the current transaction distribution against the training baseline. "
        "PSI > 0.10 → investigate  |  PSI > 0.20 → retrain required"
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        n_baseline = st.number_input("Baseline samples", 500, 10000, 2000, step=500)
        n_current  = st.number_input("Current samples",  100, 5000,  500,  step=100)
        fraud_frac = st.slider("Fraud fraction in current data", 0.0, 0.5, 0.05, step=0.01)
        seed = st.number_input("Random seed", 0, 9999, 42)
        run_btn = st.button("Run Drift Analysis", type="primary")

    if not run_btn:
        st.info("Configure the analysis in the sidebar and click **Run Drift Analysis**.")
        return

    rng = np.random.default_rng(int(seed))

    # ── Generate baseline and current distributions ───────────────────────
    n_fraud_current = int(n_current * fraud_frac)
    n_legit_current = n_current - n_fraud_current

    # Baseline: normal-ish distribution for each V feature and Amount
    baseline: dict[str, np.ndarray] = {}
    current:  dict[str, np.ndarray] = {}

    # Amount
    baseline["Amount"] = rng.lognormal(3.5, 1.0, int(n_baseline))
    current_legit_amt  = rng.lognormal(3.5, 1.0, n_legit_current)
    current_fraud_amt  = rng.lognormal(5.5, 1.2, n_fraud_current)
    current["Amount"]  = np.concatenate([current_legit_amt, current_fraud_amt])

    # V features: baseline = Normal(0, 0.5), fraud has shifted components
    for i in range(1, 29):
        baseline[f"V{i}"] = rng.normal(0, 0.5, int(n_baseline))
        legit = rng.normal(0, 0.5, n_legit_current)
        fraud = rng.normal(-1.5 if i in {1, 3, 14} else 0, 2.0, n_fraud_current)
        current[f"V{i}"] = np.concatenate([legit, fraud])

    # ── Compute PSI per feature ───────────────────────────────────────────
    features = ["Amount"] + [f"V{i}" for i in range(1, 29)]
    psi_values = {f: _compute_psi(baseline[f], current[f]) for f in features}

    psi_df = pd.DataFrame(
        [{"Feature": f, "PSI": round(v, 4)} for f, v in psi_values.items()]
    ).sort_values("PSI", ascending=False)

    # ── Metrics ───────────────────────────────────────────────────────────
    mean_psi   = psi_df["PSI"].mean()
    n_warn     = (psi_df["PSI"] >= _PSI_WARN).sum()
    n_critical = (psi_df["PSI"] >= _PSI_CRITICAL).sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean PSI",         f"{mean_psi:.4f}")
    m2.metric("Features ≥ 0.10",  int(n_warn),    delta_color="inverse")
    m3.metric("Features ≥ 0.20",  int(n_critical), delta_color="inverse")
    m4.metric("Fraud fraction",   f"{fraud_frac:.1%}")

    if n_critical > 0:
        st.error(f"🔴 {n_critical} feature(s) have PSI ≥ 0.20 — model retrain recommended.")
    elif n_warn > 0:
        st.warning(f"🟠 {n_warn} feature(s) have PSI ≥ 0.10 — investigate data pipeline.")
    else:
        st.success("✅ All features within acceptable drift bounds.")

    # ── Bar chart ─────────────────────────────────────────────────────────
    st.subheader("PSI by Feature")
    chart_df = psi_df.set_index("Feature")

    def _color_psi(val: float) -> str:
        if val >= _PSI_CRITICAL:
            return "🔴"
        if val >= _PSI_WARN:
            return "🟠"
        return "🟢"

    chart_df["status"] = chart_df["PSI"].apply(_color_psi)
    st.bar_chart(chart_df["PSI"])

    # ── Detail table ──────────────────────────────────────────────────────
    with st.expander("Feature PSI Table"):
        styled = psi_df.style.background_gradient(subset=["PSI"], cmap="RdYlGn_r")
        st.dataframe(styled, use_container_width=True)

    # ── Distribution comparison for top drifted feature ──────────────────
    top_feature = psi_df.iloc[0]["Feature"]
    st.subheader(f"Distribution Comparison: {top_feature}")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Baseline")
        hist_data = pd.Series(baseline[top_feature]).rename("baseline")
        st.area_chart(hist_data.value_counts(bins=30, sort=False).sort_index())
    with col2:
        st.caption("Current")
        hist_data = pd.Series(current[top_feature]).rename("current")
        st.area_chart(hist_data.value_counts(bins=30, sort=False).sort_index())


if __name__ == "__main__":
    render()
