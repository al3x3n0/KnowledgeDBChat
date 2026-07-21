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


def _default_metadata(*, name: str, description: str, input_schema: Dict[str, Any]) -> ToolMetadata:
    base = name.split("mcp:", 1)[1] if name.startswith("mcp:") else name
    write_tools = {
        "delete_document", "batch_delete_documents", "update_document_tags",
        "create_document_from_text", "ingest_url", "merge_entities", "delete_entity",
        "rebuild_document_knowledge_graph", "run_custom_tool", "run_workflow",
        "docker_execute", "delegate_subtask", "share_findings", "request_review",
        "execute_data_pipeline", "write_and_run_script", "write_file", "apply_patch",
        "run_command", "export_document", "create_memory", "execute_workflow",
        "send_message_to_agent", "send_notification", "send_email_alert", "create_chart",
        "render_diagram", "create_kg_entity", "create_kg_relationship", "schedule_job",
        "cancel_scheduled_job", "merge_documents", "create_handoff", "broadcast_to_siblings",
        "transcribe_document",
    }
    network_tools = {
        "web_scrape", "ingest_url", "search_arxiv", "ingest_arxiv_papers",
        "literature_review_arxiv", "create_repo_report", "docker_execute",
        "clone_and_index_repo", "search_web", "fetch_url_content", "summarize_url",
        "render_diagram",
    }
    high_pii = {"docker_execute", "write_and_run_script"}
    medium_pii = {
        "web_scrape", "ingest_url", "run_custom_tool", "execute_python",
        "execute_data_pipeline", "run_command", "search_code",
    }
    high_cost = {"docker_execute", "execute_data_pipeline", "write_and_run_script", "run_command"}
    medium_cost = {
        "generate_report", "create_repo_report", "create_presentation", "delegate_subtask",
        "request_review", "execute_python", "clone_and_index_repo", "export_document",
        "execute_workflow", "search_web", "summarize_url", "create_chart", "batch_search",
        "batch_summarize", "compress_history", "summarize_findings", "create_handoff",
        "transcribe_document", "analyze_image",
    }
    return ToolMetadata(
        name=name,
        description=description,
        input_schema=input_schema,
        effects="write" if base in write_tools else "read",
        network="egress" if base in network_tools else "none",
        cost_tier="high" if base in high_cost else ("medium" if base in medium_cost else "low"),
        pii_risk="high" if base in high_pii else ("medium" if base in medium_pii else "low"),
    )


def iter_builtin_tools() -> Iterable[ToolMetadata]:
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
            input_schema=tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {},
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

    mcp_fallback: Dict[str, ToolMetadata] = {
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
    return mcp_fallback.get(base_name) if is_mcp else None
