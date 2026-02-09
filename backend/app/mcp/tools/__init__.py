"""
MCP tools for external AI agents.

Each tool provides a specific capability that external agents can invoke.
"""

from app.mcp.tools.search import SearchTool
from app.mcp.tools.documents import DocumentsTool
from app.mcp.tools.chat import ChatTool
from app.mcp.tools.generation import GenerationTool
from app.mcp.tools.web_scrape import WebScrapeTool

__all__ = [
    "SearchTool",
    "DocumentsTool",
    "ChatTool",
    "GenerationTool",
    "WebScrapeTool",
]
