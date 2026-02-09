"""
Shared MCP tool executor.

Used by:
- MCP HTTP endpoints
- Tool audit "run" endpoint for approved MCP calls
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.auth import MCPAuthContext
from app.mcp.tools.search import SearchTool
from app.mcp.tools.documents import DocumentsTool
from app.mcp.tools.chat import ChatTool
from app.mcp.tools.generation import GenerationTool
from app.mcp.tools.web_scrape import WebScrapeTool


_search_tool = SearchTool()
_documents_tool = DocumentsTool()
_chat_tool = ChatTool()
_generation_tool = GenerationTool()
_web_scrape_tool = WebScrapeTool()


async def execute_mcp_tool(
    *,
    tool_name: str,
    args: Dict[str, Any],
    auth: MCPAuthContext,
    db: AsyncSession,
) -> Dict[str, Any]:
    name = str(tool_name or "").strip()
    if not name:
        return {"error": "Missing tool name"}

    if name == "search":
        return await _search_tool.execute(
            auth=auth,
            db=db,
            query=args.get("query", ""),
            limit=args.get("limit", 10),
            source_ids=args.get("source_ids"),
            file_types=args.get("file_types"),
        )

    if name == "list_documents":
        return await _documents_tool.list_documents(
            auth=auth,
            db=db,
            limit=args.get("limit", 20),
            offset=args.get("offset", 0),
            source_id=args.get("source_id"),
            file_type=args.get("file_type"),
            search=args.get("search"),
        )

    if name == "get_document":
        return await _documents_tool.get_document(
            auth=auth,
            db=db,
            document_id=args.get("document_id", ""),
            include_content=args.get("include_content", True),
        )

    if name == "list_sources":
        return await _documents_tool.list_sources(
            auth=auth,
            db=db,
            limit=args.get("limit", 50),
        )

    if name == "chat":
        return await _chat_tool.execute(
            auth=auth,
            db=db,
            question=args.get("question", ""),
            source_ids=args.get("source_ids"),
            document_ids=args.get("document_ids"),
            max_context_chunks=args.get("max_context_chunks", 5),
            include_sources=args.get("include_sources", True),
        )

    if name == "web_scrape":
        return await _web_scrape_tool.execute(
            auth=auth,
            db=db,
            url=args.get("url", ""),
            follow_links=args.get("follow_links", False),
            max_pages=args.get("max_pages", 1),
            max_depth=args.get("max_depth", 0),
            same_domain_only=args.get("same_domain_only", True),
            include_links=args.get("include_links", True),
            allow_private_networks=args.get("allow_private_networks", False),
            max_content_chars=args.get("max_content_chars", 50000),
        )

    if name == "create_presentation":
        return await _generation_tool.create_presentation(
            auth=auth,
            db=db,
            topic=args.get("topic", ""),
            slide_count=args.get("slide_count", 10),
            style=args.get("style", "professional"),
            source_ids=args.get("source_ids"),
            include_diagrams=args.get("include_diagrams", True),
        )

    if name == "create_repo_report":
        return await _generation_tool.create_repo_report(
            auth=auth,
            db=db,
            repo_url=args.get("repo_url", ""),
            output_format=args.get("output_format", "docx"),
            title=args.get("title"),
            sections=args.get("sections"),
            style=args.get("style", "professional"),
        )

    if name == "get_job_status":
        return await _generation_tool.get_job_status(
            auth=auth,
            db=db,
            job_id=args.get("job_id", ""),
            job_type=args.get("job_type", ""),
        )

    if name == "list_jobs":
        return await _generation_tool.list_jobs(
            auth=auth,
            db=db,
            job_type=args.get("job_type", "all"),
            limit=args.get("limit", 20),
        )

    logger.warning(f"Unknown MCP tool: {name}")
    return {"error": f"Unknown tool '{name}'"}
