"""
Transaction Explorer — Streamlit page.

Lets analysts submit a transaction for on-demand investigation and
view the full agent report: SHAP explanation, behavioral signals,
fraud pattern, confidence score, and recommended action.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import streamlit as st

from src.simulation.fraud_simulator import FraudSimulator

_sim = FraudSimulator(seed=0)

_API_BASE = "http://localhost:8000"


def render() -> None:
    st.title("🔎 Transaction Explorer")
    st.caption(
        "Submit a transaction for a full agent investigation. "
        "Use the sidebar to load a simulated scenario."
    )

    # ── Sidebar: scenario loader ─────────────────────────────────────────
    with st.sidebar:
        st.header("Load a Scenario")
        scenario = st.selectbox(
            "Scenario",
            ["manual", "normal", "velocity_fraud", "account_takeover",
             "testing_attack", "large_anomalous_purchase"],
        )
        load_btn = st.button("Load")

    # ── Feature input ─────────────────────────────────────────────────────
    default_features: dict = {}
    if load_btn and scenario != "manual":
        if scenario == "normal":
            tx = _sim.normal_transaction()
        else:
            tx = _sim.fraud_transaction(pattern=scenario)
        default_features = tx.features
        st.sidebar.success(f"Loaded: {scenario} | label={tx.label}")

    col1, col2 = st.columns([2, 1])
    with col1:
        features_raw = st.text_area(
            "Transaction features (JSON)",
            value=json.dumps(default_features, indent=2) if default_features else
                  json.dumps({"Amount": 120.5, "V1": 0.0, "V2": 0.0}, indent=2),
            height=300,
        )
    with col2:
        st.markdown("**Options**")
        include_explanation = st.checkbox("SHAP explanation", value=True)
        include_behavior    = st.checkbox("Behavior analysis", value=True)
        include_drift       = st.checkbox("Drift check", value=False)
        threshold           = st.slider("Alert threshold", 0.1, 0.99, 0.5, step=0.05)
        top_k               = st.slider("SHAP top-k features", 3, 15, 5)

    investigate_btn = st.button("🚀 Investigate", type="primary")

    if not investigate_btn:
        return

    try:
        features = json.loads(features_raw)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return

    # ── Call agent API ────────────────────────────────────────────────────
    try:
        import httpx
        response = httpx.post(
            f"{_API_BASE}/agent/investigate",
            json={
                "features": features,
                "include_explanation": include_explanation,
                "include_behavior": include_behavior,
                "include_drift": include_drift,
                "threshold": threshold,
                "top_k": top_k,
            },
            timeout=30,
        )
        response.raise_for_status()
        report = response.json()
    except Exception as exc:
        # Fallback: run agent directly (no running API)
        st.warning(f"API unreachable ({exc}). Running agent directly...")
        from src.agents.fraud_agent import run_investigation
        report = run_investigation(
            features=features,
            include_explanation=include_explanation,
            include_behavior=include_behavior,
            include_drift=include_drift,
            threshold=threshold,
            top_k=top_k,
        )

    _render_report(report)


def _render_report(report: dict) -> None:
    """Display the agent investigation report."""
    st.divider()

    # ── Risk summary ──────────────────────────────────────────────────────
    prob   = report.get("fraud_probability", 0)
    level  = report.get("risk_level", "unknown").upper()
    action = report.get("recommended_action", "—")
    pattern= report.get("fraud_pattern", "unknown")
    conf   = report.get("confidence_score", 0.0)
    partial= report.get("partial_report", False)

    color  = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue", "LOW": "green"}.get(level, "gray")

    st.markdown(f"## :{color}[{level} RISK]   —   {action}")
    if partial:
        st.warning("⚠️ Partial report — some tools failed. Confidence penalised.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud Probability", f"{prob:.3f}")
    m2.metric("Risk Score",        f"{report.get('risk_score', prob*100):.1f}")
    m3.metric("Confidence",        f"{conf:.2f}")
    m4.metric("Pattern",           pattern)

    st.markdown(f"**Analyst Summary**: {report.get('analyst_summary', '—')}")

    # ── Key risk signals ──────────────────────────────────────────────────
    if report.get("key_risk_signals"):
        with st.expander("🚦 Key Risk Signals", expanded=True):
            for sig in report["key_risk_signals"]:
                st.write(f"  • {sig}")

    # ── SHAP explanation ──────────────────────────────────────────────────
    expl = report.get("model_explanation", {})
    top_features = expl.get("top_features_contributing", [])
    if top_features:
        with st.expander("🧠 SHAP Model Explanation", expanded=True):
            st.caption(expl.get("narrative", ""))
            import pandas as pd
            df = pd.DataFrame(top_features)
            df["abs_shap"] = df["shap_value"].abs()
            df = df.sort_values("abs_shap", ascending=False)
            st.bar_chart(df.set_index("feature")["shap_value"])
            st.dataframe(df[["feature", "feature_value", "shap_value", "direction"]])

    # ── Behavioral anomalies ──────────────────────────────────────────────
    anomalies = report.get("behavioral_anomalies", [])
    if anomalies:
        with st.expander("⚡ Behavioral Anomalies"):
            for a in anomalies:
                st.write(f"  • {a}")

    # ── Drift analysis ────────────────────────────────────────────────────
    drift = report.get("drift_analysis")
    if drift:
        with st.expander("📊 Drift Analysis"):
            st.metric("Dataset PSI", f"{drift.get('dataset_psi', 0):.3f}")
            st.metric("Fraud Rate Shift", f"{drift.get('fraud_rate_shift', 0):.3f}")
            if drift.get("top_drifted_features"):
                st.write("Top drifted features:")
                for f in drift["top_drifted_features"]:
                    st.write(f"  • {f}")

    # ── Session & Case ────────────────────────────────────────────────────
    st.divider()
    cols = st.columns(2)
    cols[0].markdown(f"**Session ID**: `{report.get('session_id', '—')}`")
    if report.get("case_id"):
        cols[1].markdown(f"**Case ID**: `{report['case_id']}`")

    with st.expander("📋 Raw Report"):
        st.json({k: v for k, v in report.items() if k != "_meta"})


if __name__ == "__main__":
    render()
