"""Resilient response mapping for autonomous-job memory APIs."""

import math
from typing import Any, Optional
from uuid import UUID

from app.schemas.agent_job import (
    AgentJobExtractedMemoryResponse,
    AgentJobMemoryExtractResponse,
    AgentJobMemoryGraphResponse,
    AgentJobMemoryListResponse,
    AgentJobMemoryResponse,
    AgentJobMemorySearchItemResponse,
    AgentJobMemorySearchResponse,
    AgentJobMemoryStatsResponse,
)


def to_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion for resilient response serialization."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort finite-float coercion for resilient serialization."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def to_string_list(value: Any) -> list[str]:
    """Normalize a list payload into clean, non-empty strings."""
    if not isinstance(value, list):
        return []
    return [token for item in value if (token := str(item or "").strip())]


def to_string(value: Any, default: str = "") -> str:
    """Normalize a scalar-like payload into a trimmed string."""
    text = str(value or "").strip()
    return text if text else default


def build_extract_job_memories_response(
    *,
    job_id: UUID,
    memories: list[Any],
    extraction_stats: Optional[dict] = None,
) -> AgentJobMemoryExtractResponse:
    """Serialize manual extraction with deduplication statistics."""
    stats = extraction_stats if isinstance(extraction_stats, dict) else {}
    return AgentJobMemoryExtractResponse(
        job_id=str(job_id),
        memories_created=len(memories),
        parsed_count=to_int(stats.get("parsed_count", 0), 0),
        candidate_count=to_int(stats.get("candidate_count", 0), 0),
        skipped_duplicates=to_int(stats.get("skipped_duplicates", 0), 0),
        is_relaunch_chain=bool(stats.get("is_relaunch_chain", False)),
        relaunch_root_job_id=str(stats.get("relaunch_root_job_id") or "").strip()
        or None,
        memories=[
            AgentJobExtractedMemoryResponse(
                id=str(memory.id),
                type=to_string(
                    getattr(memory, "memory_type", None),
                    "unknown",
                ),
                content=to_string(getattr(memory, "content", None)),
                importance_score=to_float(getattr(memory, "importance_score", 0.0)),
                tags=to_string_list(getattr(memory, "tags", None)),
            )
            for memory in memories
        ],
    )


def build_job_memory_response(
    *,
    job_id: UUID | str,
    memory: Any,
) -> AgentJobMemoryResponse:
    """Serialize one ConversationMemory-like object."""
    return AgentJobMemoryResponse(
        id=str(memory.id),
        job_id=str(job_id),
        type=to_string(getattr(memory, "memory_type", None), "unknown"),
        content=to_string(getattr(memory, "content", None)),
        importance_score=to_float(getattr(memory, "importance_score", 0.0)),
        tags=to_string_list(getattr(memory, "tags", None)),
        context=memory.context if isinstance(memory.context, dict) else None,
        access_count=to_int(getattr(memory, "access_count", 0)),
        created_at=memory.created_at.isoformat()
        if getattr(memory, "created_at", None)
        else None,
    )


def build_job_memories_list_response(
    *,
    job_id: UUID,
    memories: list[Any],
) -> AgentJobMemoryListResponse:
    """Serialize one job's memory collection."""
    return AgentJobMemoryListResponse(
        job_id=str(job_id),
        memories=[
            build_job_memory_response(job_id=job_id, memory=memory)
            for memory in memories
        ],
        total=len(memories),
    )


def build_memory_search_response(
    *,
    query: str,
    memories: list[Any],
) -> AgentJobMemorySearchResponse:
    """Serialize a task-memory search result."""
    return AgentJobMemorySearchResponse(
        query=query,
        memories=[
            AgentJobMemorySearchItemResponse(
                id=str(memory.id),
                type=to_string(
                    getattr(memory, "memory_type", None),
                    "unknown",
                ),
                content=to_string(getattr(memory, "content", None)),
                importance_score=to_float(getattr(memory, "importance_score", 0.0)),
                tags=to_string_list(getattr(memory, "tags", None)),
                job_id=str(memory.job_id) if getattr(memory, "job_id", None) else None,
                access_count=to_int(getattr(memory, "access_count", 0)),
                created_at=memory.created_at.isoformat()
                if getattr(memory, "created_at", None)
                else None,
            )
            for memory in memories
        ],
        total=len(memories),
    )


def build_memory_stats_response(
    *,
    stats: Optional[dict[str, Any]],
) -> AgentJobMemoryStatsResponse:
    """Normalize aggregate task-memory statistics."""
    payload = stats if isinstance(stats, dict) else {}

    by_type: dict[str, int] = {}
    if isinstance(payload.get("by_type"), dict):
        for key, value in payload["by_type"].items():
            token = str(key or "").strip()
            if token:
                by_type[token] = to_int(value)

    most_accessed: list[dict[str, Any]] = []
    if isinstance(payload.get("most_accessed"), list):
        for item in payload["most_accessed"]:
            if isinstance(item, dict):
                most_accessed.append(
                    {
                        "id": str(item.get("id") or ""),
                        "type": str(item.get("type") or ""),
                        "content": str(item.get("content") or ""),
                        "access_count": to_int(item.get("access_count")),
                    }
                )

    most_important: list[dict[str, Any]] = []
    if isinstance(payload.get("most_important"), list):
        for item in payload["most_important"]:
            if isinstance(item, dict):
                most_important.append(
                    {
                        "id": str(item.get("id") or ""),
                        "type": str(item.get("type") or ""),
                        "content": str(item.get("content") or ""),
                        "importance": to_float(item.get("importance")),
                    }
                )

    return AgentJobMemoryStatsResponse(
        total_memories=to_int(payload.get("total_memories", 0)),
        by_type=by_type,
        job_sourced=to_int(payload.get("job_sourced", 0)),
        chat_sourced=to_int(payload.get("chat_sourced", 0)),
        manual=to_int(payload.get("manual", 0)),
        most_accessed=most_accessed,
        most_important=most_important,
    )


def build_memory_graph_response(
    *,
    graph: Optional[dict[str, Any]],
    job_id: Optional[UUID | str] = None,
) -> AgentJobMemoryGraphResponse:
    """Normalize a task-memory graph payload."""
    payload = graph if isinstance(graph, dict) else {}

    nodes: list[dict[str, Any]] = []
    if isinstance(payload.get("nodes"), list):
        for node in payload["nodes"]:
            if not isinstance(node, dict):
                continue
            nodes.append(
                {
                    "id": str(node.get("id") or ""),
                    "type": str(node.get("type") or ""),
                    "content": str(node.get("content") or ""),
                    "importance_score": to_float(node.get("importance_score")),
                    "tags": to_string_list(node.get("tags")),
                    "job_id": str(node.get("job_id") or "").strip() or None,
                    "created_at": str(node.get("created_at") or "").strip() or None,
                    "project_scope": str(node.get("project_scope") or "").strip()
                    or None,
                    "execution_outcome": str(
                        node.get("execution_outcome") or ""
                    ).strip()
                    or None,
                    "strategy_signal": str(node.get("strategy_signal") or "").strip()
                    or None,
                    "access_count": to_int(node.get("access_count")),
                }
            )

    edges: list[dict[str, Any]] = []
    if isinstance(payload.get("edges"), list):
        for edge in payload["edges"]:
            if isinstance(edge, dict):
                edges.append(
                    {
                        "source": str(edge.get("source") or ""),
                        "target": str(edge.get("target") or ""),
                        "weight": to_float(edge.get("weight")),
                        "reasons": to_string_list(edge.get("reasons")),
                    }
                )

    stats_out: dict[str, Any] = {}
    if isinstance(payload.get("stats"), dict):
        for key, value in payload["stats"].items():
            token = str(key or "").strip()
            if token:
                stats_out[token] = value

    normalized_job_id = str(job_id).strip() if job_id is not None else ""
    if not normalized_job_id:
        normalized_job_id = str(payload.get("job_id") or "").strip()

    return AgentJobMemoryGraphResponse(
        nodes=nodes,
        edges=edges,
        stats=stats_out,
        job_id=normalized_job_id or None,
    )
