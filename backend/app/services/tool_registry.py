"""Compatibility shim for the extracted tool catalog."""

from app.agent_core.tool_catalog import (
    ToolMetadata,
    get_tool_metadata,
    iter_builtin_tools,
)

__all__ = ["ToolMetadata", "get_tool_metadata", "iter_builtin_tools"]
