"""Unit tests: Tool base contract, ToolResult, and registry."""

from __future__ import annotations

import pytest
from src.tools.base import Tool, ToolResult
from src.tools.registry import TOOL_REGISTRY, register_tool, get_tool


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

class TestToolResult:
    def test_success_unwrap(self):
        r = ToolResult(success=True, data={"x": 1})
        assert r.unwrap() == {"x": 1}

    def test_failure_unwrap_raises(self):
        r = ToolResult(success=False, data={}, error="boom")
        with pytest.raises(RuntimeError, match="boom"):
            r.unwrap()


# ---------------------------------------------------------------------------
# Tool.run() — non-throwing contract
# ---------------------------------------------------------------------------

class TestToolRun:
    def _make_tool(self, fn, name="test_tool"):
        return Tool(
            name=name,
            description="test",
            input_schema={},
            output_schema={},
            execute=fn,
            timeout_seconds=5,
        )

    def test_success_returns_tool_result(self):
        tool = self._make_tool(lambda **kw: {"answer": 42})
        result = tool.run()
        assert result.success is True
        assert result.data["answer"] == 42
        assert result.error is None

    def test_exception_captured_as_failure(self):
        def bad(**kw):
            raise ValueError("intentional failure")

        tool = self._make_tool(bad)
        result = tool.run()
        assert result.success is False
        assert "intentional failure" in result.error
        assert result.data == {}

    def test_execution_time_recorded(self):
        tool = self._make_tool(lambda **kw: {})
        result = tool.run()
        assert result.execution_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_registered_tools_present(self):
        expected = {
            "fraud_scoring",
            "feature_explanation",
            "behavior_analysis",
            "drift_monitoring",
            "transaction_history",
        }
        assert expected.issubset(set(TOOL_REGISTRY.keys()))

    def test_get_tool_raises_on_missing(self):
        with pytest.raises(KeyError):
            get_tool("nonexistent_tool_xyz")

    def test_register_custom_tool(self):
        custom = Tool(
            name="_test_custom",
            description="ephemeral",
            input_schema={},
            output_schema={},
            execute=lambda **kw: {"ok": True},
        )
        register_tool(custom)
        assert "_test_custom" in TOOL_REGISTRY
        del TOOL_REGISTRY["_test_custom"]
