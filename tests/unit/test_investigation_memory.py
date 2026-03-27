"""Unit tests: InvestigationSession memory."""

from __future__ import annotations

import pytest
from src.memory.investigation_memory import (
    InvestigationSession,
    create_session,
    get_session,
    store_investigation,
    get_last_investigation,
    clear_investigation_memory,
)

SAMPLE_REPORT = {"fraud_probability": 0.92, "recommended_action": "flag for manual review"}
SAMPLE_FEATURES = {"Amount": 100.0, "V1": -1.2}


class TestInvestigationSession:
    def test_session_has_uuid(self):
        s = InvestigationSession()
        assert len(s.session_id) == 36  # standard UUID length

    def test_add_investigation(self):
        s = InvestigationSession()
        s.add_investigation(SAMPLE_FEATURES, [{"tool": "fraud_scoring", "success": True}], [], SAMPLE_REPORT)
        assert len(s.final_reports) == 1
        assert s.latest_report()["fraud_probability"] == 0.92

    def test_to_dict_shape(self):
        s = InvestigationSession()
        s.add_investigation(SAMPLE_FEATURES, [], [], SAMPLE_REPORT)
        d = s.to_dict()
        assert d["transaction_count"] == 1
        assert "session_id" in d

    def test_latest_report_none_on_empty(self):
        s = InvestigationSession()
        assert s.latest_report() is None


class TestSessionStore:
    def teardown_method(self):
        clear_investigation_memory()

    def test_create_and_get_session(self):
        session = create_session()
        retrieved = get_session(session.session_id)
        assert retrieved is session

    def test_get_nonexistent_session(self):
        assert get_session("00000000-0000-0000-0000-000000000000") is None

    def test_store_investigation_auto_creates_session(self):
        store_investigation("new_session_x", SAMPLE_REPORT, features=SAMPLE_FEATURES)
        last = get_last_investigation("new_session_x")
        assert last["fraud_probability"] == 0.92

    def test_multiple_investigations_in_session(self):
        sid = "multi_session"
        store_investigation(sid, {"fraud_probability": 0.1}, features=SAMPLE_FEATURES)
        store_investigation(sid, {"fraud_probability": 0.9}, features=SAMPLE_FEATURES)
        last = get_last_investigation(sid)
        assert last["fraud_probability"] == 0.9

    def test_clear_specific_session(self):
        store_investigation("sess_a", SAMPLE_REPORT, features=SAMPLE_FEATURES)
        store_investigation("sess_b", SAMPLE_REPORT, features=SAMPLE_FEATURES)
        clear_investigation_memory("sess_a")
        assert get_last_investigation("sess_a") is None
        assert get_last_investigation("sess_b") is not None

    def test_clear_all(self):
        store_investigation("sess_c", SAMPLE_REPORT, features=SAMPLE_FEATURES)
        clear_investigation_memory()
        assert get_last_investigation("sess_c") is None
