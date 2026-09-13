"""Unified tool catalog for agent core consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    input_schema: Dict[str, Any]
    effects: str
    network: str
    cost_tier: str
    pii_risk: str


def _default_metadata(
    *, name: str, description: str, input_schema: Dict[str, Any]
) -> ToolMetadata:
    base = name.split("mcp:", 1)[1] if name.startswith("mcp:") else name

    # Every builtin tool declares its own governance, and is believed.
    from app.agent_core.tool_specs import spec_for

    spec = spec_for(base)
    if spec is not None:
        return ToolMetadata(
            name=name,
            description=description or spec.description,
            input_schema=input_schema or spec.parameters,
            effects=spec.effects,
            network=spec.network,
            cost_tier=spec.cost_tier,
            pii_risk=spec.pii_risk,
        )

    # Below here is the MCP surface only, which has no specs. This is the older
    # form of classification -- a tool is whatever these lists remember to say
    # it is, so omitting one silently classifies it read-safe, cheap and
    # private. That is how 25 mutating tools were once classified read.
    #
    # These lists held 158 names until the builtin tools moved to specs. The
    # other 149 named tools that now carry their own classification, so editing
    # one here would have changed nothing while looking like a fix. Only the
    # names that still decide something are left.
    write_tools = {
        "create_presentation",
        "create_repo_report",
        "docker_execute",
    }
    network_tools = {
        "create_repo_report",
        "docker_execute",
    }
    high_pii = {
        "docker_execute",
    }
    medium_pii: set[str] = set()
    high_cost = {
        "docker_execute",
    }
    medium_cost = {
        "create_presentation",
        "create_repo_report",
    }
    return ToolMetadata(
        name=name,
        description=description,
        input_schema=input_schema,
        effects="write" if base in write_tools else "read",
        network="egress" if base in network_tools else "none",
        cost_tier="high"
        if base in high_cost
        else ("medium" if base in medium_cost else "low"),
        pii_risk="high"
        if base in high_pii
        else ("medium" if base in medium_pii else "low"),
    )


def iter_builtin_tools() -> Iterable[ToolMetadata]:
    """Every tool a run can execute.

    There is one registry to read now. There were several, and the
    data-analysis tools were in a different one from this, so 21 tools an agent
    job could call had no metadata at all: invisible to the tool-policy UI,
    unclassifiable by effects, and denied as "unknown" by any policy carrying
    constraints. The guard written for exactly that failure could not see them
    either, because it derived its universe from the same partial registry.
    """
    try:
        from app.services.agent_tools import AGENT_TOOLS
    except Exception:
        AGENT_TOOLS = []

    for tool in AGENT_TOOLS or []:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        yield _default_metadata(
            name=name,
            description=str(tool.get("description") or "").strip(),
            input_schema=tool.get("parameters")
            if isinstance(tool.get("parameters"), dict)
            else {},
        )


def get_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    name = str(tool_name or "").strip()
    if not name:
        return None
    is_mcp = name.startswith("mcp:")
    base_name = name.split("mcp:", 1)[1].strip() if is_mcp else name

    for meta in iter_builtin_tools():
        if meta.name == base_name:
            if not is_mcp:
                return meta
            return ToolMetadata(
                name=f"mcp:{base_name}",
                description=meta.description,
                input_schema=meta.input_schema,
                effects=meta.effects,
                network=meta.network,
                cost_tier=meta.cost_tier,
                pii_risk=meta.pii_risk,
            )

    return iter_mcp_tools().get(base_name) if is_mcp else None


def iter_mcp_tools() -> Dict[str, ToolMetadata]:
    """Metadata for MCP-exposed tools, keyed by unprefixed name.

    Module level so the classification guard can enumerate this surface too.
    It previously lived inside get_tool_metadata, which put it out of reach of
    the test that checks tools are classified deliberately — and that is exactly
    where two mutating tools were found classified read-safe.
    """
    return {
        "search": _default_metadata(
            name="mcp:search",
            description="Semantic search over the knowledge base",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        "list_documents": _default_metadata(
            name="mcp:list_documents",
            description="List documents",
            input_schema={"type": "object", "properties": {}},
        ),
        "get_document": _default_metadata(
            name="mcp:get_document",
            description="Get document by id",
            input_schema={
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        ),
        "list_sources": _default_metadata(
            name="mcp:list_sources",
            description="List document sources",
            input_schema={"type": "object", "properties": {}},
        ),
        "chat": _default_metadata(
            name="mcp:chat",
            description="Ask a question and get an answer grounded in the KB",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        "create_presentation": _default_metadata(
            name="mcp:create_presentation",
            description="Create a presentation job",
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        ),
        "create_repo_report": _default_metadata(
            name="mcp:create_repo_report",
            description="Create a repo report job",
            input_schema={
                "type": "object",
                "properties": {"repo_url": {"type": "string"}},
                "required": ["repo_url"],
            },
        ),
        "get_job_status": _default_metadata(
            name="mcp:get_job_status",
            description="Get status of a generation job",
            input_schema={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        "list_jobs": _default_metadata(
            name="mcp:list_jobs",
            description="List generation jobs",
            input_schema={"type": "object", "properties": {}},
        ),
        "docker_execute": _default_metadata(
            name="mcp:docker_execute",
            description="Execute a command inside a Docker container",
            input_schema={
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "command": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["image", "command"],
            },
        ),
    }
