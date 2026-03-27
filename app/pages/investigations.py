"""
Investigations Dashboard — Streamlit page.

Displays all active fraud cases from the Case Management Layer.
Analysts can:
  • Filter by status / risk level
  • View full agent investigation report
  • Update case status and add notes
  • Assign cases to themselves

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from src.cases.case_manager import CaseManager
from src.cases.case_store import VALID_STATUSES

_manager = CaseManager()

# ---------------------------------------------------------------------------
# Status colours
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    "open":             "🔴",
    "under_review":     "🟠",
    "confirmed_fraud":  "🚨",
    "false_positive":   "✅",
    "dismissed":        "⚪",
}

_RISK_COLOR = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🟢",
    "unknown":  "⚫",
}


def render() -> None:
    st.title("🔍 Fraud Investigation Cases")
    st.caption("Case management console for fraud analysts.")

    # ── Filters ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status", ["all"] + sorted(VALID_STATUSES), index=0
        )
    with col2:
        risk_filter = st.selectbox(
            "Risk Level", ["all", "critical", "high", "medium", "low"], index=0
        )
    with col3:
        limit = st.number_input("Max cases", min_value=5, max_value=500, value=50, step=10)

    cases = _manager.list_cases(
        status=None if status_filter == "all" else status_filter,
        risk_level=None if risk_filter == "all" else risk_filter,
        limit=int(limit),
    )

    st.markdown(f"**{len(cases)} case(s) found**")

    if not cases:
        st.info("No cases match the current filter.")
        return

    # ── Case list ────────────────────────────────────────────────────────
    for case in cases:
        s_icon = _STATUS_COLOR.get(case.status, "⚫")
        r_icon = _RISK_COLOR.get(case.risk_level, "⚫")
        with st.expander(
            f"{s_icon} {case.case_id[:8]}…  |  {r_icon} {case.risk_level.upper()}"
            f"  |  {case.fraud_pattern}  |  prob: {case.fraud_probability:.2f}",
            expanded=False,
        ):
            _render_case_detail(case)

    # ── Summary metrics ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total",           len(cases))
    m2.metric("Open",            sum(1 for c in cases if c.status == "open"))
    m3.metric("Confirmed Fraud", sum(1 for c in cases if c.status == "confirmed_fraud"))
    m4.metric("Avg Probability", f"{sum(c.fraud_probability for c in cases) / max(len(cases),1):.2f}")


def _render_case_detail(case) -> None:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**Transaction ID**: `{case.transaction_id or '—'}`")
        st.markdown(f"**Session ID**: `{case.session_id or '—'}`")
        st.markdown(f"**Created**: {case.created_at[:19]}")
        st.markdown(f"**Updated**: {case.updated_at[:19]}")
        if case.assigned_to:
            st.markdown(f"**Assigned to**: {case.assigned_to}")
        if case.analyst_notes:
            st.markdown("**Analyst Notes**")
            st.text(case.analyst_notes.strip())

    with c2:
        new_status = st.selectbox(
            "Update status",
            sorted(VALID_STATUSES),
            index=sorted(VALID_STATUSES).index(case.status),
            key=f"status_{case.case_id}",
        )
        note = st.text_input("Note", key=f"note_{case.case_id}")
        if st.button("Save", key=f"save_{case.case_id}"):
            _manager.update_status(case.case_id, new_status, note=note)
            st.success("Updated")
            st.rerun()

        analyst = st.text_input("Assign to", key=f"assign_{case.case_id}")
        if st.button("Assign", key=f"do_assign_{case.case_id}"):
            _manager.assign(case.case_id, analyst)
            st.success(f"Assigned to {analyst}")
            st.rerun()

    # Agent report
    with st.expander("Full Agent Report"):
        report = case.agent_report or {}
        st.json({k: v for k, v in report.items() if k != "_meta"})
        if "_meta" in report:
            st.caption("Tool calls:")
            for tc in report["_meta"].get("tool_calls", []):
                st.write(f"  • {tc['tool']} — {'✅' if tc['success'] else '❌'} "
                         f"({tc['execution_time_ms']:.0f} ms)")


if __name__ == "__main__":
    render()
