"""
Unified tool registry for agents, MCP, and workflows.

This is a lightweight in-process catalog. It provides:
- tool metadata (effects, network, cost, pii risk)
- schemas/descriptions (best-effort)

Database-backed policies are handled separately in tool_policy_engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    input_schema: Dict[str, Any]
    effects: str  # read | write
    network: str  # none | egress
    cost_tier: str  # low | medium | high
    pii_risk: str  # low | medium | high


def _default_metadata(*, name: str, description: str, input_schema: Dict[str, Any]) -> ToolMetadata:
    base = name.split("mcp:", 1)[1] if name.startswith("mcp:") else name

    write_tools = {
        "delete_document",
        "batch_delete_documents",
        "update_document_tags",
        "create_document_from_text",
        "ingest_url",
        "merge_entities",
        "delete_entity",
        "rebuild_document_knowledge_graph",
        "run_custom_tool",
        "run_workflow",
    }
    network_tools = {
        "web_scrape",
        "ingest_url",
        "search_arxiv",
        "ingest_arxiv_papers",
        "literature_review_arxiv",
        "create_repo_report",
    }

    effects = "write" if base in write_tools else "read"
    network = "egress" if base in network_tools else "none"

    pii_risk = "medium" if base in {"web_scrape", "ingest_url", "run_custom_tool"} else "low"
    cost_tier = "medium" if base in {"generate_report", "create_repo_report", "create_presentation"} else "low"

    return ToolMetadata(
        name=name,
        description=description,
        input_schema=input_schema,
        effects=effects,
        network=network,
        cost_tier=cost_tier,
        pii_risk=pii_risk,
    )


def iter_builtin_tools() -> Iterable[ToolMetadata]:
    # Agent + workflow builtins are defined in AGENT_TOOLS
    try:
        from app.services.agent_tools import AGENT_TOOLS
    except Exception:
        AGENT_TOOLS = []

    for t in AGENT_TOOLS or []:
        name = str(t.get("name") or "").strip()
        if not name:
            continue
        desc = str(t.get("description") or "").strip()
        schema = t.get("parameters") if isinstance(t.get("parameters"), dict) else {}
        yield _default_metadata(name=name, description=desc, input_schema=schema)

def get_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    name = str(tool_name or "").strip()
    if not name:
        return None
    is_mcp = name.startswith("mcp:")
    base_name = name.split("mcp:", 1)[1].strip() if is_mcp else name

    # Built-in tools (agent/workflow) by exact name.
    for meta in iter_builtin_tools():
        if meta.name == base_name:
            if is_mcp:
                # Return an MCP-namespaced view so policies can address MCP tools separately.
                return ToolMetadata(
                    name=f"mcp:{base_name}",
                    description=meta.description,
                    input_schema=meta.input_schema,
                    effects=meta.effects,
                    network=meta.network,
                    cost_tier=meta.cost_tier,
                    pii_risk=meta.pii_risk,
                )
            return meta

    # MCP-only tools (generation/list_documents etc.) are not part of AGENT_TOOLS.
    mcp_fallback: dict[str, ToolMetadata] = {
        "search": _default_metadata(
            name="mcp:search",
            description="Semantic search over the knowledge base",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
        "list_documents": _default_metadata(
            name="mcp:list_documents",
            description="List documents",
            input_schema={"type": "object", "properties": {}},
        ),
        "get_document": _default_metadata(
            name="mcp:get_document",
            description="Get document by id",
            input_schema={"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]},
        ),
        "list_sources": _default_metadata(
            name="mcp:list_sources",
            description="List document sources",
            input_schema={"type": "object", "properties": {}},
        ),
        "chat": _default_metadata(
            name="mcp:chat",
            description="Ask a question and get an answer grounded in the KB",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
        "create_presentation": _default_metadata(
            name="mcp:create_presentation",
            description="Create a presentation job",
            input_schema={"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
        ),
        "create_repo_report": _default_metadata(
            name="mcp:create_repo_report",
            description="Create a repo report job",
            input_schema={"type": "object", "properties": {"repo_url": {"type": "string"}}, "required": ["repo_url"]},
        ),
        "get_job_status": _default_metadata(
            name="mcp:get_job_status",
            description="Get status of a generation job",
            input_schema={"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        ),
        "list_jobs": _default_metadata(
            name="mcp:list_jobs",
            description="List generation jobs",
            input_schema={"type": "object", "properties": {}},
        ),
    }

    if is_mcp:
        return mcp_fallback.get(base_name)
    return None
