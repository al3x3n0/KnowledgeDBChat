"""
MCP Server implementation using FastAPI.

This module provides MCP-compatible endpoints that external AI agents
can use to interact with the KnowledgeDB platform.
"""

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.mcp.auth import (
    MCPAuthContext,
    extract_api_key,
    validate_api_key,
    log_api_usage,
)
from app.mcp.tools.search import SearchTool
from app.mcp.tools.documents import DocumentsTool
from app.mcp.tools.chat import ChatTool
from app.mcp.tools.generation import GenerationTool
from app.mcp.tools.web_scrape import WebScrapeTool


# Initialize tools
search_tool = SearchTool()
documents_tool = DocumentsTool()
chat_tool = ChatTool()
generation_tool = GenerationTool()
web_scrape_tool = WebScrapeTool()

# Create router
mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])


# =============================================================================
# Pydantic Models for MCP Protocol
# =============================================================================

class MCPToolDefinition(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPListToolsResponse(BaseModel):
    """Response for listing available tools."""
    tools: List[MCPToolDefinition]


class MCPToolCallRequest(BaseModel):
    """Request to call a tool."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    """Response from a tool call."""
    content: List[Dict[str, Any]]
    isError: bool = False


class MCPServerInfo(BaseModel):
    """MCP server information."""
    name: str = "KnowledgeDB MCP Server"
    version: str = "1.0.0"
    description: str = "MCP server for KnowledgeDB platform"
    protocolVersion: str = "2024-11-05"


# =============================================================================
# Authentication Dependency
# =============================================================================

async def get_mcp_auth(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> MCPAuthContext:
    """
    Dependency to validate API key and get auth context.
    """
    api_key = extract_api_key(request)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Use Authorization header, X-API-Key header, or api_key query param.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth = await validate_api_key(api_key, db)

    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth


# =============================================================================
# MCP Protocol Endpoints
# =============================================================================

@mcp_router.get("/info", response_model=MCPServerInfo)
async def get_server_info():
    """Get MCP server information."""
    return MCPServerInfo()


@mcp_router.get("/tools", response_model=MCPListToolsResponse)
async def list_tools(
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    List available MCP tools.

    Returns tool definitions with their input schemas, filtered by API key configuration.
    """
    # Check if MCP is enabled for this key
    if not auth.api_key.is_mcp_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MCP access is disabled for this API key"
        )

    all_tools = [
        ("search", "Search documents in the knowledge base using semantic search", search_tool.input_schema),
        ("list_documents", "List available documents", documents_tool.operations["list"]["input_schema"]),
        ("get_document", "Get a specific document by ID", documents_tool.operations["get"]["input_schema"]),
        ("list_sources", "List available document sources", documents_tool.operations["list_sources"]["input_schema"]),
        ("chat", "Ask questions and get AI-powered answers based on the knowledge base", chat_tool.input_schema),
        ("web_scrape", "Fetch web pages and extract readable text and links", web_scrape_tool.input_schema),
        ("create_presentation", "Create a PowerPoint presentation from a topic", generation_tool.operations["create_presentation"]["input_schema"]),
        ("create_repo_report", "Create a report from a GitHub/GitLab repository", generation_tool.operations["create_repo_report"]["input_schema"]),
        ("get_job_status", "Get status of a generation job", generation_tool.operations["get_job_status"]["input_schema"]),
        ("list_jobs", "List generation jobs", generation_tool.operations["list_jobs"]["input_schema"]),
    ]

    # Filter tools based on API key configuration
    tools = []
    for name, description, schema in all_tools:
        if auth.api_key.is_tool_allowed(name):
            # Apply platform policies too (deny-by-policy removes from list)
            try:
                from app.services.tool_policy_engine import evaluate_tool_policy

                decision = await evaluate_tool_policy(
                    db=db,
                    tool_name=f"mcp:{name}",
                    tool_args=None,
                    user=auth.user,
                    api_key_id=auth.api_key.id,
                )
                if not decision.allowed:
                    continue
            except Exception:
                continue

            tools.append(
                MCPToolDefinition(
                    name=name,
                    description=description,
                    inputSchema=schema,
                )
            )

    return MCPListToolsResponse(tools=tools)


@mcp_router.post("/tools/call", response_model=MCPToolCallResponse)
async def call_tool(
    request: Request,
    tool_call: MCPToolCallRequest,
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    Call an MCP tool.

    Execute a tool with the provided arguments and return the result.
    """
    start_time = time.time()
    tool_name = tool_call.name
    args = tool_call.arguments

    logger.info(f"MCP tool call: {tool_name}, user={auth.user.username}")

    # Check if MCP is enabled for this key
    if not auth.api_key.is_mcp_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="MCP access is disabled for this API key"
        )

    # Check if tool is allowed for this API key
    if not auth.api_key.is_tool_allowed(tool_name):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tool '{tool_name}' is not allowed for this API key"
        )

    # Platform tool policies (deny-by-policy, approval gate)
    try:
        from app.models.tool_audit import ToolExecutionAudit
        from app.services.tool_policy_engine import evaluate_tool_policy

        decision = await evaluate_tool_policy(
            db=db,
            tool_name=f"mcp:{tool_name}",
            tool_args=args,
            user=auth.user,
            api_key_id=auth.api_key.id,
        )
        if not decision.allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=decision.denied_reason or "Tool denied by policy")

        if decision.require_approval:
            audit = ToolExecutionAudit(
                user_id=auth.user_id,
                agent_definition_id=None,
                conversation_id=None,
                tool_name=f"mcp:{tool_name}",
                tool_input={"arguments": args, "api_key_id": str(auth.api_key.id)},
                policy_decision={
                    "allowed": bool(decision.allowed),
                    "require_approval": bool(decision.require_approval),
                    "denied_reason": decision.denied_reason,
                    "matched_policies": decision.matched_policies,
                },
                status="requires_approval",
                approval_required=True,
                approval_mode="owner_and_admin",
                approval_status="pending_owner",
            )
            db.add(audit)
            await db.commit()
            await db.refresh(audit)
            return MCPToolCallResponse(
                content=[
                    {
                        "type": "text",
                        "text": f"approval_required: tool '{tool_name}' requires approval; approval_id={audit.id}",
                        "approval_id": str(audit.id),
                    }
                ],
                isError=True,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"MCP policy evaluation failed: {e}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tool policy evaluation failed")

    try:
        result: Dict[str, Any] = {}

        # Route to appropriate tool
        if tool_name == "search":
            result = await search_tool.execute(
                auth=auth,
                db=db,
                query=args.get("query", ""),
                limit=args.get("limit", 10),
                source_ids=args.get("source_ids"),
                file_types=args.get("file_types"),
            )

        elif tool_name == "list_documents":
            result = await documents_tool.list_documents(
                auth=auth,
                db=db,
                limit=args.get("limit", 20),
                offset=args.get("offset", 0),
                source_id=args.get("source_id"),
                file_type=args.get("file_type"),
                search=args.get("search"),
            )

        elif tool_name == "get_document":
            result = await documents_tool.get_document(
                auth=auth,
                db=db,
                document_id=args.get("document_id", ""),
                include_content=args.get("include_content", True),
            )

        elif tool_name == "list_sources":
            result = await documents_tool.list_sources(
                auth=auth,
                db=db,
                limit=args.get("limit", 50),
            )

        elif tool_name == "chat":
            result = await chat_tool.execute(
                auth=auth,
                db=db,
                question=args.get("question", ""),
                source_ids=args.get("source_ids"),
                document_ids=args.get("document_ids"),
                max_context_chunks=args.get("max_context_chunks", 5),
                include_sources=args.get("include_sources", True),
            )

        elif tool_name == "web_scrape":
            result = await web_scrape_tool.execute(
                auth=auth,
                db=db,
                url=args.get("url", ""),
                follow_links=args.get("follow_links", False),
                max_pages=args.get("max_pages", 1),
                max_depth=args.get("max_depth", 0),
                same_domain_only=args.get("same_domain_only", True),
                include_links=args.get("include_links", True),
                allow_private_networks=args.get("allow_private_networks", False),
                max_content_chars=args.get("max_content_chars", 50_000),
            )

        elif tool_name == "create_presentation":
            result = await generation_tool.create_presentation(
                auth=auth,
                db=db,
                topic=args.get("topic", ""),
                slide_count=args.get("slide_count", 10),
                style=args.get("style", "professional"),
                source_ids=args.get("source_ids"),
                include_diagrams=args.get("include_diagrams", True),
            )

        elif tool_name == "create_repo_report":
            result = await generation_tool.create_repo_report(
                auth=auth,
                db=db,
                repo_url=args.get("repo_url", ""),
                output_format=args.get("output_format", "docx"),
                title=args.get("title"),
                sections=args.get("sections"),
                style=args.get("style", "professional"),
            )

        elif tool_name == "get_job_status":
            result = await generation_tool.get_job_status(
                auth=auth,
                db=db,
                job_id=args.get("job_id", ""),
                job_type=args.get("job_type", ""),
            )

        elif tool_name == "list_jobs":
            result = await generation_tool.list_jobs(
                auth=auth,
                db=db,
                job_type=args.get("job_type", "all"),
                limit=args.get("limit", 20),
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown tool: {tool_name}"
            )

        # Log usage
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_api_usage(
            db=db,
            api_key_id=auth.api_key.id,
            endpoint=f"/mcp/tools/call/{tool_name}",
            method="POST",
            status_code=200,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            response_time_ms=response_time_ms,
        )

        # Check for errors in result
        is_error = "error" in result and result.get("error")

        return MCPToolCallResponse(
            content=[{"type": "text", "text": result}],
            isError=is_error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP tool call error: {tool_name}, error={e}")

        # Log error
        response_time_ms = int((time.time() - start_time) * 1000)
        await log_api_usage(
            db=db,
            api_key_id=auth.api_key.id,
            endpoint=f"/mcp/tools/call/{tool_name}",
            method="POST",
            status_code=500,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            response_time_ms=response_time_ms,
        )

        return MCPToolCallResponse(
            content=[{"type": "text", "text": {"error": str(e)}}],
            isError=True,
        )


# =============================================================================
# Convenience Endpoints (Direct tool access)
# =============================================================================

@mcp_router.post("/search")
async def search_endpoint(
    request: Request,
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """Direct search endpoint."""
    body = await request.json()
    return await search_tool.execute(
        auth=auth,
        db=db,
        query=body.get("query", ""),
        limit=body.get("limit", 10),
        source_ids=body.get("source_ids"),
        file_types=body.get("file_types"),
    )


@mcp_router.post("/chat")
async def chat_endpoint(
    request: Request,
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """Direct chat endpoint."""
    body = await request.json()
    return await chat_tool.execute(
        auth=auth,
        db=db,
        question=body.get("question", ""),
        source_ids=body.get("source_ids"),
        document_ids=body.get("document_ids"),
        max_context_chunks=body.get("max_context_chunks", 5),
        include_sources=body.get("include_sources", True),
    )


@mcp_router.get("/documents")
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0,
    source_id: Optional[str] = None,
    file_type: Optional[str] = None,
    search: Optional[str] = None,
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """Direct list documents endpoint."""
    return await documents_tool.list_documents(
        auth=auth,
        db=db,
        limit=limit,
        offset=offset,
        source_id=source_id,
        file_type=file_type,
        search=search,
    )


@mcp_router.get("/documents/{document_id}")
async def get_document_endpoint(
    document_id: str,
    include_content: bool = True,
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """Direct get document endpoint."""
    return await documents_tool.get_document(
        auth=auth,
        db=db,
        document_id=document_id,
        include_content=include_content,
    )


@mcp_router.get("/sources")
async def list_sources_endpoint(
    limit: int = 50,
    auth: MCPAuthContext = Depends(get_mcp_auth),
    db: AsyncSession = Depends(get_db),
):
    """Direct list sources endpoint."""
    return await documents_tool.list_sources(
        auth=auth,
        db=db,
        limit=limit,
    )
