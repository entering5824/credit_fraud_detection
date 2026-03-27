"""Agent tools for fraud investigation.

Tools self-register via `register_tool()` on import.
Use `TOOL_REGISTRY` (the live dict) or the legacy alias `TOOLS_REGISTRY` interchangeably.
"""

from src.tools.base import Tool, ToolResult
from src.tools.registry import TOOL_REGISTRY, register_tool, get_tool, list_tools

# Import each tool module so they self-register
import src.tools.fraud_scoring_tool  # noqa: F401
import src.tools.feature_explanation_tool  # noqa: F401
import src.tools.behavior_analysis_tool  # noqa: F401
import src.tools.drift_monitoring_tool  # noqa: F401
import src.tools.transaction_history_tool  # noqa: F401
import src.tools.graph_analysis_tool  # noqa: F401

# Legacy alias kept for backward compatibility
TOOLS_REGISTRY = TOOL_REGISTRY

__all__ = [
    "Tool",
    "ToolResult",
    "TOOL_REGISTRY",
    "TOOLS_REGISTRY",
    "register_tool",
    "get_tool",
    "list_tools",
]
