"""Pure relaunch-lineage traversal for autonomous jobs."""

from collections import deque
from datetime import datetime
from typing import Any, Mapping, Optional
from uuid import UUID

from app.schemas.agent_job import (
    AgentJobRelaunchLineageNode,
    AgentJobRelaunchLineageResponse,
)


def extract_parent_job_id(config: Optional[dict]) -> Optional[UUID]:
    """Return a valid relaunch parent identifier from persisted job config."""
    if not isinstance(config, dict):
        return None
    raw = str(config.get("relaunch_from_job_id") or "").strip()
    if not raw:
        return None
    try:
        return UUID(raw)
    except (TypeError, ValueError, AttributeError):
        return None


def build_children_counts(
    config_rows: list[tuple[UUID, Optional[dict]]],
) -> dict[UUID, int]:
    """Aggregate direct relaunch-child counts from job config rows."""
    counts: dict[UUID, int] = {}
    for _job_id, config in config_rows:
        parent_id = extract_parent_job_id(config)
        if parent_id is not None:
            counts[parent_id] = counts.get(parent_id, 0) + 1
    return counts


def _launch_mode(config: Optional[dict]) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    launch_mode = str(config.get("launch_mode") or "").strip().lower()
    return launch_mode or None


def to_lineage_node(job: Any) -> AgentJobRelaunchLineageNode:
    """Map a job-like object to its public relaunch-lineage representation."""
    config = job.config if isinstance(job.config, dict) else {}
    return AgentJobRelaunchLineageNode(
        id=job.id,
        name=job.name,
        status=job.status,
        created_at=job.created_at,
        launch_mode=_launch_mode(config),
    )


def build_lineage(
    job: Any,
    jobs_by_id: Mapping[UUID, Any],
    *,
    max_ancestors: int = 100,
    max_descendants: int = 500,
) -> AgentJobRelaunchLineageResponse:
    """Build bounded ancestry and breadth-first descendants for a job."""
    max_ancestors = max(1, min(int(max_ancestors or 0), 300))
    max_descendants = max(1, min(int(max_descendants or 0), 2000))

    parent_by_child: dict[UUID, UUID] = {}
    children_by_parent: dict[UUID, list[Any]] = {}
    for item in jobs_by_id.values():
        config = item.config if isinstance(item.config, dict) else {}
        parent_id = extract_parent_job_id(config)
        if parent_id is None or parent_id not in jobs_by_id:
            continue
        parent_by_child[item.id] = parent_id
        children_by_parent.setdefault(parent_id, []).append(item)

    ancestors: list[AgentJobRelaunchLineageNode] = []
    seen_ancestors: set[UUID] = {job.id}
    current_id = parent_by_child.get(job.id)
    while (
        current_id
        and current_id in jobs_by_id
        and current_id not in seen_ancestors
        and len(ancestors) < max_ancestors
    ):
        current = jobs_by_id[current_id]
        ancestors.append(to_lineage_node(current))
        seen_ancestors.add(current_id)
        current_id = parent_by_child.get(current_id)
    ancestors_truncated = bool(
        current_id and current_id in jobs_by_id and current_id not in seen_ancestors
    )

    descendants: list[AgentJobRelaunchLineageNode] = []
    pending: deque[UUID] = deque([job.id])
    seen_descendants: set[UUID] = {job.id}
    while pending and len(descendants) < max_descendants:
        parent_id = pending.popleft()
        children = sorted(
            children_by_parent.get(parent_id, []),
            key=lambda child: child.created_at or datetime.min,
        )
        for child in children:
            if child.id in seen_descendants:
                continue
            seen_descendants.add(child.id)
            descendants.append(to_lineage_node(child))
            pending.append(child.id)
    descendants_truncated = bool(pending)

    root_job_id = ancestors[-1].id if ancestors else job.id
    parent_job_id = ancestors[0].id if ancestors else None
    latest_child_job_id = None
    if descendants:
        latest_child_job_id = max(
            descendants,
            key=lambda node: node.created_at or datetime.min,
        ).id

    return AgentJobRelaunchLineageResponse(
        job_id=job.id,
        root_job_id=root_job_id,
        parent_job_id=parent_job_id,
        latest_child_job_id=latest_child_job_id,
        ancestors_truncated=ancestors_truncated,
        descendants_truncated=descendants_truncated,
        ancestors=ancestors,
        descendants=descendants,
    )
