"""Self-registration tool registry for the fraud investigation agent."""

from __future__ import annotations

from typing import Any

from src.tools.base import Tool

TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(tool: Tool) -> Tool:
    """Register a tool by name and return it (usable as a decorator or direct call)."""
    TOOL_REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> Tool:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' is not registered. Available: {list(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name]


def list_tools() -> list[dict[str, Any]]:
    """Return lightweight metadata for all registered tools (for LLM planning prompts)."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        }
        for t in TOOL_REGISTRY.values()
    ]
