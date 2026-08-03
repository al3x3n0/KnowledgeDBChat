"""HTTP boundary for autonomous-job memories and task-memory graphs."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.memory import ConversationMemory
from app.models.user import User
from app.modules.autonomy.application import memory_presenters
from app.schemas.agent_job import (
    AgentJobMemoryDeleteResponse,
    AgentJobMemoryExtractResponse,
    AgentJobMemoryGraphResponse,
    AgentJobMemoryListResponse,
    AgentJobMemoryResponse,
    AgentJobMemorySearchResponse,
    AgentJobMemoryStatsResponse,
)
from app.services.agent_job_memory_service import agent_job_memory_service

router = APIRouter()


async def _get_owned_job(
    *,
    job_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> AgentJob:
    result = await db.execute(
        select(AgentJob).where(and_(AgentJob.id == job_id, AgentJob.user_id == user_id))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )
    return job


@router.get("/{job_id}/memories", response_model=AgentJobMemoryListResponse)
async def get_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryListResponse:
    """Return all active memories created from a user-owned job."""
    await _get_owned_job(job_id=job_id, user_id=current_user.id, db=db)
    memories = await agent_job_memory_service.get_job_memories(
        job_id=job_id,
        user_id=str(current_user.id),
        db=db,
    )
    return memory_presenters.build_job_memories_list_response(
        job_id=job_id,
        memories=memories,
    )


@router.post(
    "/{job_id}/memories/extract",
    response_model=AgentJobMemoryExtractResponse,
)
async def extract_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryExtractResponse:
    """Manually extract durable memories from a terminal user-owned job."""
    job = await _get_owned_job(
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    if job.status not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only extract memories from completed or failed jobs",
        )

    try:
        extraction_stats: dict = {}
        memories = await agent_job_memory_service.extract_memories_from_job(
            job=job,
            user_id=str(current_user.id),
            db=db,
            extraction_reason="manual_extract",
            force_extract=True,
            stats_out=extraction_stats,
        )
    except Exception as error:
        logger.error(f"Failed to extract memories from job {job_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory extraction failed: {str(error)}",
        ) from error

    logger.info(f"Manually extracted {len(memories)} memories from job {job_id}")
    return memory_presenters.build_extract_job_memories_response(
        job_id=job_id,
        memories=memories,
        extraction_stats=extraction_stats,
    )


@router.post("/{job_id}/memories", response_model=AgentJobMemoryResponse)
async def create_job_memory(
    job_id: UUID,
    memory_type: str = Query(
        ...,
        description="Memory type: finding, insight, pattern, or lesson",
    ),
    content: str = Query(..., description="Memory content"),
    importance: float = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Importance score",
    ),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryResponse:
    """Create a durable memory associated with a user-owned job."""
    if memory_type not in {
        "finding",
        "insight",
        "pattern",
        "lesson",
        "fact",
        "context",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid memory type. Use: finding, insight, pattern, "
                "lesson, fact, or context"
            ),
        )

    job = await _get_owned_job(
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    try:
        memory = await agent_job_memory_service.create_memory_from_job(
            job=job,
            memory_type=memory_type,
            content=content,
            user_id=str(current_user.id),
            db=db,
            importance=importance,
            tags=tag_list,
        )
    except Exception as error:
        logger.error(f"Failed to create memory for job {job_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory creation failed: {str(error)}",
        ) from error

    return memory_presenters.build_job_memory_response(
        job_id=job_id,
        memory=memory,
    )


@router.delete(
    "/{job_id}/memories",
    response_model=AgentJobMemoryDeleteResponse,
)
async def delete_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryDeleteResponse:
    """Soft-delete memories associated with a user-owned job."""
    await _get_owned_job(job_id=job_id, user_id=current_user.id, db=db)
    deleted_count = await agent_job_memory_service.delete_job_memories(
        job_id=job_id,
        user_id=str(current_user.id),
        db=db,
    )
    return AgentJobMemoryDeleteResponse(
        job_id=str(job_id),
        deleted_count=int(deleted_count or 0),
    )


@router.get("/memory/graph", response_model=AgentJobMemoryGraphResponse)
async def get_task_memory_graph(
    limit: int = Query(
        120,
        ge=20,
        le=300,
        description="Max memories to include as graph nodes",
    ),
    min_link_score: float = Query(
        1.0,
        ge=0.2,
        le=10.0,
        description="Minimum edge score",
    ),
    max_edges: int = Query(
        800,
        ge=50,
        le=3000,
        description="Maximum graph edges",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryGraphResponse:
    """Return the current user's reusable cross-job task-memory graph."""
    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=limit,
        min_link_score=min_link_score,
        max_edges=max_edges,
    )
    return memory_presenters.build_memory_graph_response(graph=graph)


@router.get(
    "/{job_id}/memories/graph",
    response_model=AgentJobMemoryGraphResponse,
)
async def get_job_memory_graph(
    job_id: UUID,
    neighbor_depth: int = Query(
        1,
        ge=1,
        le=2,
        description="Neighborhood depth around this job's memory nodes",
    ),
    limit: int = Query(180, ge=20, le=300, description="Max nodes to scan"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryGraphResponse:
    """Return the task-memory subgraph centered on one user-owned job."""
    await _get_owned_job(job_id=job_id, user_id=current_user.id, db=db)
    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=limit,
        min_link_score=1.0,
        max_edges=1200,
    )
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    job_node_ids = {
        str(node.get("id"))
        for node in nodes
        if str(node.get("job_id") or "") == str(job_id)
    }
    if not job_node_ids:
        return memory_presenters.build_memory_graph_response(
            graph={
                "nodes": [],
                "edges": [],
                "stats": {"memory_count": 0, "edge_count": 0},
            },
            job_id=job_id,
        )

    selected = set(job_node_ids)
    hops = max(1, min(int(neighbor_depth or 1), 2))
    for _ in range(hops):
        expanded = set(selected)
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if source in selected or target in selected:
                expanded.add(source)
                expanded.add(target)
        selected = expanded

    sub_nodes = [node for node in nodes if str(node.get("id")) in selected]
    sub_edges = [
        edge
        for edge in edges
        if str(edge.get("source")) in selected and str(edge.get("target")) in selected
    ]
    return memory_presenters.build_memory_graph_response(
        graph={
            "nodes": sub_nodes,
            "edges": sub_edges,
            "stats": {
                "memory_count": len(sub_nodes),
                "edge_count": len(sub_edges),
                "job_memory_count": len(job_node_ids),
                "neighbor_depth": hops,
            },
        },
        job_id=job_id,
    )


@router.get("/memory/stats", response_model=AgentJobMemoryStatsResponse)
async def get_memory_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryStatsResponse:
    """Return aggregate task-memory statistics for the current user."""
    stats = await agent_job_memory_service.get_memory_stats_for_user(
        user_id=str(current_user.id),
        db=db,
    )
    return memory_presenters.build_memory_stats_response(stats=stats)


@router.get("/memory/search", response_model=AgentJobMemorySearchResponse)
async def search_memories(
    query: str = Query(..., description="Search query"),
    memory_types: Optional[str] = Query(
        None,
        description="Comma-separated memory types to filter",
    ),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemorySearchResponse:
    """Search active task memories using the persisted text index."""
    type_list = (
        [memory_type.strip() for memory_type in memory_types.split(",")]
        if memory_types
        else None
    )
    query_statement = select(ConversationMemory).where(
        and_(
            ConversationMemory.user_id == current_user.id,
            ConversationMemory.is_active.is_(True),
        )
    )
    if type_list:
        query_statement = query_statement.where(
            ConversationMemory.memory_type.in_(type_list)
        )
    query_statement = query_statement.where(
        ConversationMemory.content.ilike(f"%{query}%")
    )
    query_statement = query_statement.order_by(
        desc(ConversationMemory.importance_score)
    ).limit(limit)

    result = await db.execute(query_statement)
    memories = list(result.scalars().all())
    return memory_presenters.build_memory_search_response(
        query=query,
        memories=memories,
    )
