"""
MCP (Model Context Protocol) server for external AI agents.

This module provides an MCP server that allows external AI agents to access
the KnowledgeDB platform with proper authentication and access control.
"""

from app.mcp.server import mcp_router

__all__ = ["mcp_router"]
