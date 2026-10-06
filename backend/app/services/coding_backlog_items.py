"""A coding backlog item's decomposition as the operator's endpoint keeps it,
and the job that starts orchestrating an item.

They lived in ``api/endpoints/coding_backlog.py``, which the blocker filer
(``agent_blocked_to_backlog``) then imported from -- a service reaching up into
an endpoint module for a private helper. They are here so both can call them.

``normalize_item_decomposition`` is the *endpoint's* normaliser: it keeps
unknown slice fields. The coding runner has its own, which rebuilds every slice
and reads the legacy ``slices_planned`` key; the two serve different writers
and stay separate (decided 2026-10-04).
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.coding_backlog import CodingBacklogItem
from app.services.coding_backlog_decomposition import (
    append_backlog_timeline,
    timeline_entry,
)
from app.services.job_dispatch import enqueue_agent_job


def default_decomposition() -> dict[str, Any]:
    return {
        "strategy": "portfolio_goal",
        "planned_slices": [],
        "active_slice_id": None,
        "completed_slices": [],
        "failed_slices": [],
        "promotion_decisions": [],
        "backlog_timeline": [],
        "lineage_summary": {
            "repair_job_count": 0,
            "apply_job_count": 0,
            "patch_pr_count": 0,
            "proposal_count": 0,
            "operator_action_count": 0,
        },
        "portfolio_progress": {
            "total_slices": 0,
            "pending_slices": 0,
            "completed_slices": 0,
            "failed_slices": 0,
            "auto_applied_slices": 0,
            "proposal_only_slices": 0,
        },
    }


def recompute_portfolio_progress(decomposition: dict[str, Any]) -> dict[str, Any]:
    planned = (
        decomposition.get("planned_slices")
        if isinstance(decomposition.get("planned_slices"), list)
        else []
    )
    completed = [
        str(v).strip()
        for v in (
            decomposition.get("completed_slices")
            if isinstance(decomposition.get("completed_slices"), list)
            else []
        )
        if str(v).strip()
    ]
    failed = [
        str(v).strip()
        for v in (
            decomposition.get("failed_slices")
            if isinstance(decomposition.get("failed_slices"), list)
            else []
        )
        if str(v).strip()
    ]
    return {
        "total_slices": len(planned),
        "pending_slices": sum(
            1
            for row in planned
            if str((row or {}).get("status") or "").strip().lower()
            in {"pending", "repairing", "retrying", "applying", "deferred"}
        ),
        "completed_slices": len(completed),
        "failed_slices": len(failed),
        "auto_applied_slices": sum(
            1
            for row in planned
            if str((row or {}).get("promotion_decision") or "").strip().lower()
            == "auto_applied"
        ),
        "proposal_only_slices": sum(
            1
            for row in planned
            if str((row or {}).get("promotion_decision") or "").strip().lower()
            in {"proposal_only", "patch_pr"}
        ),
    }


def normalize_item_decomposition(item: CodingBacklogItem) -> dict[str, Any]:
    raw = item.decomposition if isinstance(item.decomposition, dict) else {}
    dec = deepcopy(default_decomposition())
    if isinstance(raw, dict):
        dec.update(
            {
                k: deepcopy(v)
                for k, v in raw.items()
                if k in dec or k == "planned_slices"
            }
        )
    if not isinstance(dec.get("planned_slices"), list):
        dec["planned_slices"] = []
    if not isinstance(dec.get("completed_slices"), list):
        dec["completed_slices"] = []
    if not isinstance(dec.get("failed_slices"), list):
        dec["failed_slices"] = []
    if not isinstance(dec.get("promotion_decisions"), list):
        dec["promotion_decisions"] = []
    if not isinstance(dec.get("backlog_timeline"), list):
        dec["backlog_timeline"] = []
    if not isinstance(dec.get("lineage_summary"), dict):
        dec["lineage_summary"] = deepcopy(default_decomposition()["lineage_summary"])
    for row in dec.get("planned_slices") or []:
        if not isinstance(row, dict):
            continue
        if not isinstance(row.get("timeline"), list):
            row["timeline"] = []
        if not isinstance(row.get("job_lineage"), dict):
            row["job_lineage"] = {
                "repair_job_ids": [],
                "apply_job_ids": [],
                "patch_pr_ids": [],
                "proposal_ids": [],
                "retry_from_job_ids": [],
            }
        if not isinstance(row.get("artifact_history"), list):
            row["artifact_history"] = []
        if not isinstance(row.get("manual_promotion_history"), list):
            row["manual_promotion_history"] = []
    dec["portfolio_progress"] = recompute_portfolio_progress(dec)
    dec["lineage_summary"] = {
        "repair_job_count": sum(
            len((row.get("job_lineage") or {}).get("repair_job_ids") or [])
            for row in dec.get("planned_slices") or []
            if isinstance(row, dict)
        ),
        "apply_job_count": sum(
            len((row.get("job_lineage") or {}).get("apply_job_ids") or [])
            for row in dec.get("planned_slices") or []
            if isinstance(row, dict)
        ),
        "patch_pr_count": sum(
            len((row.get("job_lineage") or {}).get("patch_pr_ids") or [])
            for row in dec.get("planned_slices") or []
            if isinstance(row, dict)
        ),
        "proposal_count": sum(
            len((row.get("job_lineage") or {}).get("proposal_ids") or [])
            for row in dec.get("planned_slices") or []
            if isinstance(row, dict)
        ),
        "operator_action_count": sum(
            len(row.get("manual_promotion_history") or [])
            for row in dec.get("planned_slices") or []
            if isinstance(row, dict)
        ),
    }
    return dec


async def create_orchestrator_job(
    item: CodingBacklogItem,
    *,
    db: AsyncSession,
    start_immediately: bool = True,
) -> AgentJob:
    """Create the orchestrator job for ``item`` and mark the item running.

    With ``start_immediately`` the job is queued when ``db`` commits -- not
    before, since a worker cannot see a row that has only been flushed.
    """
    previous_status = str(item.status or "").strip() or "draft"
    job = AgentJob(
        name=f"Coding Backlog — {str(item.title or '').strip()[:120]}",
        description="Curated coding backlog orchestrator.",
        job_type="analysis",
        goal=str(item.portfolio_goal or "").strip()[:8000]
        or "Orchestrate coding backlog execution",
        config={
            "deterministic_runner": "coding_backlog_orchestrator",
            "coding_backlog_item_id": str(item.id),
        },
        user_id=item.user_id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=10,
    )
    db.add(job)
    await db.flush()
    item.orchestrator_job_id = job.id
    item.status = "running"
    item.started_at = item.started_at or datetime.utcnow()
    item.updated_at = datetime.utcnow()
    decomposition = normalize_item_decomposition(item)
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="system",
            action="orchestrator_started",
            previous_status=previous_status,
            new_status="running",
            related_job_id=str(job.id),
        ),
    )
    item.decomposition = decomposition
    if start_immediately:
        enqueue_agent_job(db, job.id, item.user_id)
    return job


__all__ = [
    "create_orchestrator_job",
    "default_decomposition",
    "normalize_item_decomposition",
    "recompute_portfolio_progress",
]
