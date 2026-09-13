from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.code_patch_proposal import CodePatchProposal
from app.models.coding_backlog import CodingBacklogItem
from app.models.document import DocumentSource
from app.models.patch_pr import PatchPR
from app.models.user import User
from app.schemas.coding_backlog import (
    CodingBacklogItemActionRequest,
    CodingBacklogItemCreate,
    CodingBacklogItemListResponse,
    CodingBacklogItemResponse,
    CodingBacklogItemUpdate,
)
from app.services.agent_job_templates import (
    REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
    get_builtin_agent_job_template,
)
from app.services.auth_service import get_current_user
from app.services.collaboration_service import (
    build_collaboration_summary,
    list_collaboration_user_ids,
    normalize_collaboration_visibility,
)
from app.tasks.agent_job_tasks import execute_agent_job_task

router = APIRouter()


def _normalize_str_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def _normalize_policy(policy: Any) -> dict[str, Any]:
    raw = policy if isinstance(policy, dict) else {}
    blocked = (
        raw.get("blocked_path_prefixes")
        if isinstance(raw.get("blocked_path_prefixes"), list)
        else []
    )
    return {
        "max_auto_retries": max(0, int(raw.get("max_auto_retries", 1) or 1)),
        "max_files_touched": max(0, int(raw.get("max_files_touched", 3) or 3)),
        "blocked_path_prefixes": [str(v).strip() for v in blocked if str(v).strip()],
        "require_experiments_ok": bool(raw.get("require_experiments_ok", True)),
        "confidence_threshold": max(
            0.0, min(float(raw.get("confidence_threshold", 0.55) or 0.55), 1.0)
        ),
    }


def _normalize_uuid_list(values: Any, limit: int = 200) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        try:
            value = str(UUID(str(raw))).strip()
        except Exception:
            continue
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _normalize_visibility(value: Any) -> str:
    return normalize_collaboration_visibility(value)


def _normalize_collaboration(
    payload: Any, *, fallback_owner_user_id: Optional[str] = None
) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    assigned_user_id = None
    assigned_by_user_id = None
    assigned_at = None
    for key in ("assigned_user_id", "assigned_by_user_id"):
        raw_value = str(raw.get(key) or "").strip()
        if not raw_value:
            continue
        try:
            normalized = str(UUID(raw_value))
        except Exception:
            normalized = None
        if key == "assigned_user_id":
            assigned_user_id = normalized
        else:
            assigned_by_user_id = normalized
    assigned_at_raw = raw.get("assigned_at")
    if isinstance(assigned_at_raw, datetime):
        assigned_at = assigned_at_raw.isoformat()
    elif str(assigned_at_raw or "").strip():
        assigned_at = str(assigned_at_raw).strip()
    owner_user_id = (
        str(raw.get("owner_user_id") or fallback_owner_user_id or "").strip() or None
    )
    visibility = _normalize_visibility(raw.get("visibility"))
    shared_with_user_ids = _normalize_uuid_list(raw.get("shared_with_user_ids"), 200)
    if assigned_user_id and assigned_user_id not in shared_with_user_ids:
        shared_with_user_ids.append(assigned_user_id)
    return {
        "owner_user_id": owner_user_id,
        "visibility": visibility,
        "shared_with_user_ids": shared_with_user_ids,
        "assigned_user_id": assigned_user_id,
        "assigned_at": assigned_at,
        "assigned_by_user_id": assigned_by_user_id,
        "note": str(raw.get("note") or "").strip() or None,
    }


def _is_backlog_visible_to_user(item: CodingBacklogItem, user_id: UUID) -> bool:
    if str(item.user_id) == str(user_id):
        return True
    if str(getattr(item, "assigned_user_id", "") or "").strip() == str(user_id):
        return True
    if _normalize_visibility(getattr(item, "visibility", "private")) != "shared":
        return False
    return str(user_id) in _normalize_uuid_list(
        getattr(item, "shared_with_user_ids", None), 200
    )


async def _get_visible_backlog_item_or_404(
    db: AsyncSession, item_id: UUID, user_id: UUID
) -> CodingBacklogItem:
    item = await db.get(CodingBacklogItem, item_id)
    if not item or not _is_backlog_visible_to_user(item, user_id):
        raise HTTPException(status_code=404, detail="Not found")
    return item


def _default_decomposition() -> dict[str, Any]:
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


def _recompute_portfolio_progress(decomposition: dict[str, Any]) -> dict[str, Any]:
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


def _normalize_decomposition(item: CodingBacklogItem) -> dict[str, Any]:
    raw = item.decomposition if isinstance(item.decomposition, dict) else {}
    dec = deepcopy(_default_decomposition())
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
        dec["lineage_summary"] = deepcopy(_default_decomposition()["lineage_summary"])
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
    dec["portfolio_progress"] = _recompute_portfolio_progress(dec)
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


def _timeline_entry(
    *,
    actor: str,
    action: str,
    previous_status: Optional[str] = None,
    new_status: Optional[str] = None,
    note: Optional[str] = None,
    related_job_id: Optional[str] = None,
    related_proposal_id: Optional[str] = None,
    related_patch_pr_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    entry = {
        "at": datetime.utcnow().isoformat(),
        "actor": actor,
        "action": action,
        "previous_status": previous_status,
        "new_status": new_status,
    }
    if note:
        entry["note"] = note
    if related_job_id:
        entry["job_id"] = related_job_id
    if related_proposal_id:
        entry["proposal_id"] = related_proposal_id
    if related_patch_pr_id:
        entry["patch_pr_id"] = related_patch_pr_id
    if metadata:
        entry["metadata"] = metadata
    return entry


def _append_backlog_timeline(
    decomposition: dict[str, Any], entry: dict[str, Any]
) -> None:
    rows = (
        decomposition.get("backlog_timeline")
        if isinstance(decomposition.get("backlog_timeline"), list)
        else []
    )
    rows.append(entry)
    decomposition["backlog_timeline"] = rows[-100:]


def _append_slice_timeline(slice_state: dict[str, Any], entry: dict[str, Any]) -> None:
    rows = (
        slice_state.get("timeline")
        if isinstance(slice_state.get("timeline"), list)
        else []
    )
    rows.append(entry)
    slice_state["timeline"] = rows[-60:]


def _append_lineage_id(
    slice_state: dict[str, Any], lineage_key: str, value: Optional[str]
) -> None:
    lineage = (
        slice_state.get("job_lineage")
        if isinstance(slice_state.get("job_lineage"), dict)
        else {}
    )
    lineage[lineage_key] = _append_unique(lineage.get(lineage_key), value)
    slice_state["job_lineage"] = lineage


def _append_artifact_history(
    slice_state: dict[str, Any],
    artifact_type: str,
    artifact_id: Optional[str],
    label: Optional[str] = None,
) -> None:
    if not artifact_id:
        return
    rows = (
        slice_state.get("artifact_history")
        if isinstance(slice_state.get("artifact_history"), list)
        else []
    )
    rows.append(
        {
            "at": datetime.utcnow().isoformat(),
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "label": label or artifact_type,
        }
    )
    slice_state["artifact_history"] = rows[-40:]


def _append_manual_promotion_history(
    slice_state: dict[str, Any],
    *,
    action: str,
    operator_note: Optional[str] = None,
    proposal_id: Optional[str] = None,
    patch_pr_id: Optional[str] = None,
    apply_job_id: Optional[str] = None,
) -> None:
    rows = (
        slice_state.get("manual_promotion_history")
        if isinstance(slice_state.get("manual_promotion_history"), list)
        else []
    )
    rows.append(
        {
            "at": datetime.utcnow().isoformat(),
            "action": action,
            "operator_note": operator_note,
            "proposal_id": proposal_id,
            "patch_pr_id": patch_pr_id,
            "apply_job_id": apply_job_id,
        }
    )
    slice_state["manual_promotion_history"] = rows[-40:]


def _find_slice(
    decomposition: dict[str, Any], slice_id: Optional[str]
) -> Optional[dict[str, Any]]:
    target = str(slice_id or "").strip()
    if not target:
        return None
    for row in decomposition.get("planned_slices") or []:
        if str((row or {}).get("slice_id") or "").strip() == target:
            return row
    return None


def _append_unique(values: Any, value: Optional[str]) -> list[str]:
    out = (
        [str(v).strip() for v in values if str(v).strip()]
        if isinstance(values, list)
        else []
    )
    target = str(value or "").strip()
    if target and target not in out:
        out.append(target)
    return out


def _upsert_promotion_decision(
    decomposition: dict[str, Any], entry: dict[str, Any]
) -> None:
    rows = (
        decomposition.get("promotion_decisions")
        if isinstance(decomposition.get("promotion_decisions"), list)
        else []
    )
    slice_id = str(entry.get("slice_id") or "").strip()
    kept = [
        row
        for row in rows
        if str((row or {}).get("slice_id") or "").strip() != slice_id
    ]
    kept.append(entry)
    decomposition["promotion_decisions"] = kept[-12:]


def _allowed_actions_for_slice(slice_state: dict[str, Any]) -> list[str]:
    status = str(slice_state.get("status") or "").strip().lower()
    if status in {"proposal_only", "blocked", "patch_pr"} or bool(
        slice_state.get("awaiting_operator_action")
    ):
        return [
            "apply_override",
            "create_patch_pr",
            "keep_proposal_only",
            "relaunch_slice",
            "skip_slice",
        ]
    if status == "failed":
        return ["relaunch_slice", "skip_slice"]
    return []


def _recommended_action_for_slice(slice_state: dict[str, Any]) -> Optional[str]:
    blocked_reason = str(slice_state.get("blocked_reason") or "").strip().lower()
    if blocked_reason in {
        "blocked_path_prefix",
        "max_files_touched_exceeded",
        "require_patch_pr",
    }:
        return "create_patch_pr"
    if blocked_reason == "confidence_below_threshold":
        return "apply_override"
    if str(slice_state.get("status") or "").strip().lower() == "failed":
        return "relaunch_slice"
    return "keep_proposal_only"


def _refresh_waiting_metadata(
    decomposition: dict[str, Any], slice_state: Optional[dict[str, Any]]
) -> None:
    for row in decomposition.get("planned_slices") or []:
        row["allowed_slice_actions"] = (
            _allowed_actions_for_slice(row) if row is slice_state else []
        )
    if slice_state is not None:
        slice_state["awaiting_operator_action"] = True
        slice_state["allowed_slice_actions"] = _allowed_actions_for_slice(slice_state)
        slice_state["recommended_next_action"] = _recommended_action_for_slice(
            slice_state
        )


def _clear_waiting_metadata(slice_state: dict[str, Any]) -> None:
    slice_state["awaiting_operator_action"] = False
    slice_state["allowed_slice_actions"] = []
    slice_state["recommended_next_action"] = None


def _set_latest_summary(
    item: CodingBacklogItem,
    decomposition: dict[str, Any],
    *,
    status_value: str,
    slice_state: Optional[dict[str, Any]] = None,
    note: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    summary = {
        "status": status_value,
        "portfolio_progress": _recompute_portfolio_progress(decomposition),
        "waiting_on_operator_action": bool(
            slice_state and slice_state.get("awaiting_operator_action")
        ),
        "allowed_slice_actions": (
            slice_state.get("allowed_slice_actions") if slice_state else []
        )
        or [],
        "recommended_next_action": slice_state.get("recommended_next_action")
        if slice_state
        else None,
        "active_slice_id": slice_state.get("slice_id")
        if slice_state
        else decomposition.get("active_slice_id"),
        "active_slice_title": slice_state.get("title") if slice_state else None,
        "note": note,
    }
    if extra:
        summary.update(extra)
    item.latest_summary = summary


_TERMINAL_CLOSURE_REASONS = {
    "fixed_through_backlog",
    "promoted_to_repair",
    "duplicate",
    "false_alarm",
    "outdated",
    "blocked_external",
}


def _normalize_closure_reason(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _TERMINAL_CLOSURE_REASONS else None


def _build_why_not_repair_summary(item: CodingBacklogItem) -> Optional[dict[str, Any]]:
    lineage = item.lineage if isinstance(item.lineage, dict) else {}
    if not str(lineage.get("originating_swarm_job_id") or "").strip():
        return None
    summary = item.latest_summary if isinstance(item.latest_summary, dict) else {}
    return {
        "review_reason": str(
            lineage.get("originating_swarm_review_reason") or ""
        ).strip()
        or None,
        "route_mode": str(lineage.get("originating_swarm_route_mode") or "").strip()
        or None,
        "candidate_role": str(
            lineage.get("originating_swarm_candidate_role") or ""
        ).strip()
        or None,
        "recommended_next_action": str(
            summary.get("recommended_next_action") or ""
        ).strip()
        or None,
        "waiting_on_operator_action": bool(summary.get("waiting_on_operator_action")),
        "backlog_note": str(summary.get("note") or "").strip() or None,
    }


def _derive_operator_queue_state(item: CodingBacklogItem) -> str:
    summary = item.latest_summary if isinstance(item.latest_summary, dict) else {}
    lineage = item.lineage if isinstance(item.lineage, dict) else {}
    closure_reason = _normalize_closure_reason(summary.get("closure_reason"))
    if closure_reason == "duplicate":
        return "superseded"
    if str(item.status or "").strip().lower() in {"cancelled"} and closure_reason:
        return (
            "superseded" if closure_reason in {"duplicate", "outdated"} else "blocked"
        )
    if str(item.status or "").strip().lower() in {"completed"}:
        return "blocked" if closure_reason == "blocked_external" else "in_progress"
    if bool(summary.get("waiting_on_operator_action")):
        return "awaiting_operator_decision"
    if str(item.status or "").strip().lower() in {"running"}:
        return "in_progress"
    if str(item.status or "").strip().lower() in {"failed"}:
        return "blocked"
    if str(item.status or "").strip().lower() in {"paused"}:
        return "blocked"
    assigned_user_id = str(getattr(item, "assigned_user_id", "") or "").strip()
    is_auto_routed = (
        str(lineage.get("originating_swarm_route_mode") or "").strip().lower() == "auto"
    )
    if is_auto_routed and str(item.status or "").strip().lower() == "draft":
        return "new_auto_routed" if not assigned_user_id else "ready_to_start"
    if str(item.status or "").strip().lower() == "draft":
        return "awaiting_assignment" if not assigned_user_id else "ready_to_start"
    return "ready_to_start"


def _to_response(
    item: CodingBacklogItem,
    *,
    current_user: Optional[User] = None,
    user_lookup: Optional[dict[str, User]] = None,
) -> CodingBacklogItemResponse:
    collaboration = _normalize_collaboration(
        getattr(item, "collaboration", None), fallback_owner_user_id=str(item.user_id)
    )
    latest_summary = (
        item.latest_summary if isinstance(item.latest_summary, dict) else {}
    )
    return CodingBacklogItemResponse.model_validate(
        {
            **item.__dict__,
            "child_job_ids": [
                str(v).strip() for v in _normalize_str_list(item.child_job_ids)
            ],
            "visibility": _normalize_visibility(getattr(item, "visibility", "private")),
            "shared_with_user_ids": _normalize_uuid_list(
                getattr(item, "shared_with_user_ids", None), 200
            ),
            "collaboration": collaboration,
            "collaboration_summary": build_collaboration_summary(
                owner_user_id=str(collaboration.get("owner_user_id") or item.user_id),
                visibility=str(
                    collaboration.get("visibility")
                    or getattr(item, "visibility", "private")
                ),
                shared_with_user_ids=list(
                    collaboration.get("shared_with_user_ids")
                    or getattr(item, "shared_with_user_ids", None)
                    or []
                ),
                assigned_user_id=str(
                    collaboration.get("assigned_user_id")
                    or getattr(item, "assigned_user_id", "")
                ).strip()
                or None,
                assigned_by_user_id=str(
                    collaboration.get("assigned_by_user_id")
                    or getattr(item, "assigned_by_user_id", "")
                ).strip()
                or None,
                assigned_at=str(
                    collaboration.get("assigned_at")
                    or getattr(item, "assigned_at", "")
                    or ""
                ).strip()
                or None,
                note=str(collaboration.get("note") or "").strip() or None,
                current_user_id=str(current_user.id)
                if current_user is not None
                else None,
                user_lookup=user_lookup,
            ),
            "operator_queue_state": _derive_operator_queue_state(item),
            "closure_reason": _normalize_closure_reason(
                latest_summary.get("closure_reason")
            ),
            "why_not_repair": _build_why_not_repair_summary(item),
        }
    )


async def _build_backlog_user_lookup(
    db: AsyncSession, *, current_user: User
) -> dict[str, User]:
    visible_user_ids = await list_collaboration_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)
    rows = list(
        (await db.execute(select(User).where(User.id.in_(visible_user_ids))))
        .scalars()
        .all()
    )
    return {str(row.id): row for row in rows}


def _build_orchestrator_chain_config(
    backlog_item_id: UUID, previous_child_kind: str
) -> dict[str, Any]:
    return {
        "trigger_condition": "on_any_end",
        "inherit_results": True,
        "inherit_config": False,
        "child_jobs": [
            {
                "name": "Coding Backlog — Continue",
                "job_type": "analysis",
                "goal": "Continue backlog orchestration after a child repair/apply run completes.",
                "config": {
                    "deterministic_runner": "coding_backlog_orchestrator",
                    "coding_backlog_item_id": str(backlog_item_id),
                    "coding_backlog_previous_child_kind": str(previous_child_kind or "")
                    .strip()
                    .lower()
                    or "repair",
                },
                "max_iterations": 1,
                "max_tool_calls": 0,
                "max_llm_calls": 0,
                "max_runtime_minutes": 10,
            }
        ],
    }


def _attach_terminal_continuation(
    chain_config: Optional[dict], backlog_item_id: UUID, previous_child_kind: str
) -> Optional[dict]:
    if not isinstance(chain_config, dict):
        return None
    updated = deepcopy(chain_config)
    cursor = updated
    while isinstance(cursor.get("child_jobs"), list) and cursor.get("child_jobs"):
        child = cursor["child_jobs"][-1]
        if not isinstance(child, dict):
            break
        if not isinstance(child.get("chain_config"), dict):
            child["chain_config"] = _build_orchestrator_chain_config(
                backlog_item_id, previous_child_kind
            )
            return updated
        cursor = child["chain_config"]
    return updated


async def _create_orchestrator_job(
    item: CodingBacklogItem,
    *,
    db: AsyncSession,
    start_immediately: bool = True,
) -> AgentJob:
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
    decomposition = _normalize_decomposition(item)
    _append_backlog_timeline(
        decomposition,
        _timeline_entry(
            actor="system",
            action="orchestrator_started",
            previous_status="draft" if not item.started_at else str(item.status or ""),
            new_status="running",
            related_job_id=str(job.id),
        ),
    )
    item.decomposition = decomposition
    if start_immediately:
        execute_agent_job_task.delay(str(job.id), str(item.user_id))
    return job


async def _spawn_slice_repair_job(
    item: CodingBacklogItem,
    slice_state: dict[str, Any],
    *,
    db: AsyncSession,
    operator_note: Optional[str] = None,
) -> AgentJob:
    source = await db.get(DocumentSource, item.source_id) if item.source_id else None
    template = get_builtin_agent_job_template(REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID)
    if not template:
        raise HTTPException(
            status_code=500, detail="Repo bug triage template unavailable"
        )

    symptom = (
        str(item.failure_symptom or "").strip()
        or str(slice_state.get("goal") or "").strip()
        or str(item.portfolio_goal or "").strip()[:4000]
    )
    scope = (
        str(slice_state.get("scope") or item.scope or "auto").strip().lower() or "auto"
    )
    file_paths = _normalize_str_list(slice_state.get("file_paths"))
    commands = _normalize_str_list(slice_state.get("commands"))
    search_query = (
        str(slice_state.get("search_query") or "").strip()
        or " ".join(
            part
            for part in [
                ("" if scope == "auto" else scope),
                symptom,
                " ".join(file_paths[:2]),
            ]
            if part
        ).strip()[:500]
    )
    merged_config = dict(template.default_config or {})
    merged_config.update(
        {
            "source_id": str(item.source_id),
            "launch_mode": "quick_start_repo_bug_triage",
            "failure_symptom": symptom,
            "scope": scope,
            "search_query": search_query,
            "quick_start": {
                "profile": "repo_bug_triage",
                "version": "v2",
                "source_name": str(getattr(source, "name", "") or ""),
                "source_type": str(getattr(source, "source_type", "") or "")
                .strip()
                .lower(),
                "scope": scope,
                "autonomy_mode": "patch_proposal",
                "execution_depth": "workspace_planned",
            },
            "coding_backlog_item_id": str(item.id),
            "coding_backlog_child_kind": "repair",
            "coding_backlog_goal_type": "portfolio_goal",
            "coding_backlog_slice_id": str(slice_state.get("slice_id") or ""),
            "coding_backlog_slice_title": str(slice_state.get("title") or ""),
        }
    )
    if file_paths:
        merged_config["file_paths"] = file_paths
    if commands:
        merged_config["commands"] = commands
    if str(item.error_output or "").strip():
        merged_config["error_output"] = str(item.error_output).strip()[:4000]
    if operator_note:
        merged_config["coding_backlog_operator_note"] = str(operator_note).strip()[
            :5000
        ]

    repair_job = AgentJob(
        name=f"{str(item.title or '').strip()[:120]} — {str(slice_state.get('title') or 'Repair')[:80]}",
        description="Repair slice launched from coding backlog action.",
        job_type=template.job_type,
        goal=str(slice_state.get("goal") or item.portfolio_goal or "").strip()[:8000]
        or template.default_goal,
        config=merged_config,
        user_id=item.user_id,
        status=AgentJobStatus.PENDING.value,
        parent_job_id=item.orchestrator_job_id,
        root_job_id=item.orchestrator_job_id,
        chain_depth=1,
        chain_config=_attach_terminal_continuation(
            template.default_chain_config, item.id, "repair"
        ),
        max_iterations=template.default_max_iterations,
        max_tool_calls=template.default_max_tool_calls,
        max_llm_calls=template.default_max_llm_calls,
        max_runtime_minutes=template.default_max_runtime_minutes,
    )
    db.add(repair_job)
    await db.flush()
    prev_status = str(slice_state.get("status") or "").strip() or None
    _append_slice_timeline(
        slice_state,
        _timeline_entry(
            actor="system",
            action="repair_job_started",
            previous_status=prev_status,
            new_status="repairing",
            note=operator_note,
            related_job_id=str(repair_job.id),
        ),
    )
    _append_lineage_id(slice_state, "repair_job_ids", str(repair_job.id))
    if operator_note:
        _append_manual_promotion_history(
            slice_state,
            action="relaunch_slice" if prev_status else "repair_job_started",
            operator_note=operator_note,
        )
    execute_agent_job_task.delay(str(repair_job.id), str(item.user_id))
    return repair_job


async def _spawn_slice_apply_job(
    item: CodingBacklogItem,
    slice_state: dict[str, Any],
    *,
    db: AsyncSession,
    proposal_id: str,
    operator_note: Optional[str] = None,
) -> AgentJob:
    apply_job = AgentJob(
        name=f"{str(item.title or '').strip()[:120]} — Apply {str(slice_state.get('title') or 'Patch')[:74]}",
        description="Operator-triggered backlog apply job.",
        job_type="analysis",
        goal="Apply the selected code patch proposal to the knowledge base.",
        config={
            "deterministic_runner": "code_patch_apply_to_kb",
            "proposal_id": proposal_id,
            "proposal_strategy": "explicit",
            "apply_patch_to_kb": True,
            "dry_run": False,
            "require_experiments_ok": True,
            "require_dry_run_first": False,
            "fail_on_block": True,
            "coding_backlog_item_id": str(item.id),
            "coding_backlog_child_kind": "apply",
            "coding_backlog_slice_id": str(slice_state.get("slice_id") or ""),
            "coding_backlog_operator_note": str(operator_note or "").strip()[:5000]
            or None,
        },
        user_id=item.user_id,
        status=AgentJobStatus.PENDING.value,
        parent_job_id=item.orchestrator_job_id,
        root_job_id=item.orchestrator_job_id,
        chain_depth=1,
        chain_config=_build_orchestrator_chain_config(item.id, "apply"),
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=15,
    )
    db.add(apply_job)
    await db.flush()
    _append_slice_timeline(
        slice_state,
        _timeline_entry(
            actor="user",
            action="apply_override_started",
            previous_status=str(slice_state.get("status") or "").strip() or None,
            new_status="applying",
            note=operator_note,
            related_job_id=str(apply_job.id),
            related_proposal_id=proposal_id,
        ),
    )
    _append_lineage_id(slice_state, "apply_job_ids", str(apply_job.id))
    _append_lineage_id(slice_state, "proposal_ids", proposal_id)
    _append_artifact_history(slice_state, "proposal", proposal_id, "Selected proposal")
    _append_manual_promotion_history(
        slice_state,
        action="apply_override",
        operator_note=operator_note,
        proposal_id=proposal_id,
        apply_job_id=str(apply_job.id),
    )
    execute_agent_job_task.delay(str(apply_job.id), str(item.user_id))
    return apply_job


@router.get("", response_model=CodingBacklogItemListResponse)
async def list_coding_backlog_items(
    status_filter: Optional[str] = Query(None, alias="status"),
    visibility_scope: str = Query("mine", description="mine|shared|all"),
    assigned_user_id: Optional[UUID] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    visibility_scope = str(visibility_scope or "mine").strip().lower() or "mine"
    stmt = select(CodingBacklogItem).where(
        or_(
            CodingBacklogItem.user_id == current_user.id,
            CodingBacklogItem.visibility == "shared",
            CodingBacklogItem.assigned_user_id == current_user.id,
        )
    )
    if status_filter:
        stmt = stmt.where(
            CodingBacklogItem.status == str(status_filter).strip().lower()
        )
    rows = [
        row
        for row in list(
            (await db.execute(stmt.order_by(desc(CodingBacklogItem.updated_at))))
            .scalars()
            .all()
        )
        if _is_backlog_visible_to_user(row, current_user.id)
    ]
    if visibility_scope == "mine":
        rows = [row for row in rows if str(row.user_id) == str(current_user.id)]
    elif visibility_scope == "shared":
        rows = [row for row in rows if str(row.user_id) != str(current_user.id)]
    if assigned_user_id:
        rows = [
            row
            for row in rows
            if str(getattr(row, "assigned_user_id", "") or "") == str(assigned_user_id)
        ]
    total = len(rows)
    rows = rows[offset : offset + limit]
    user_lookup = await _build_backlog_user_lookup(db, current_user=current_user)
    return CodingBacklogItemListResponse(
        items=[
            _to_response(item, current_user=current_user, user_lookup=user_lookup)
            for item in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "", response_model=CodingBacklogItemResponse, status_code=status.HTTP_201_CREATED
)
async def create_coding_backlog_item(
    payload: CodingBacklogItemCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source = await db.get(DocumentSource, payload.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Document source not found")
    collaboration_payload = _normalize_collaboration(
        (
            payload.collaboration
            if isinstance(payload.collaboration, dict)
            else {
                "owner_user_id": str(current_user.id),
                "visibility": payload.visibility,
                "shared_with_user_ids": payload.shared_with_user_ids,
                "assigned_user_id": payload.assigned_user_id,
                "assigned_by_user_id": payload.assigned_by_user_id,
                "assigned_at": payload.assigned_at,
            }
        ),
        fallback_owner_user_id=str(current_user.id),
    )

    item = CodingBacklogItem(
        user_id=current_user.id,
        source_id=payload.source_id,
        title=str(payload.title).strip(),
        portfolio_goal=str(payload.portfolio_goal).strip(),
        status="draft",
        priority=int(payload.priority),
        scope=str(payload.scope or "auto").strip().lower() or "auto",
        failure_symptom=str(payload.failure_symptom or "").strip() or None,
        error_output=str(payload.error_output or "").strip() or None,
        file_paths=_normalize_str_list(payload.file_paths),
        commands=_normalize_str_list(payload.commands),
        auto_apply_enabled=bool(payload.auto_apply_enabled),
        require_patch_pr=bool(payload.require_patch_pr),
        visibility=str(collaboration_payload.get("visibility") or "private"),
        shared_with_user_ids=list(
            collaboration_payload.get("shared_with_user_ids") or []
        )
        or None,
        assigned_user_id=UUID(str(collaboration_payload.get("assigned_user_id")))
        if str(collaboration_payload.get("assigned_user_id") or "").strip()
        else None,
        assigned_by_user_id=UUID(str(collaboration_payload.get("assigned_by_user_id")))
        if str(collaboration_payload.get("assigned_by_user_id") or "").strip()
        else None,
        assigned_at=datetime.fromisoformat(
            str(collaboration_payload.get("assigned_at"))
        )
        if str(collaboration_payload.get("assigned_at") or "").strip()
        else payload.assigned_at,
        collaboration=collaboration_payload,
        policy=_normalize_policy(payload.policy),
        lineage=deepcopy(payload.lineage)
        if isinstance(payload.lineage, dict)
        else None,
        decomposition=_default_decomposition(),
        child_job_ids=[],
        latest_summary={
            "status": "draft",
            "portfolio_progress": _default_decomposition()["portfolio_progress"],
        },
    )
    db.add(item)
    await db.flush()
    if payload.start_immediately:
        await _create_orchestrator_job(item, db=db, start_immediately=True)
    await db.commit()
    await db.refresh(item)
    user_lookup = await _build_backlog_user_lookup(db, current_user=current_user)
    return _to_response(item, current_user=current_user, user_lookup=user_lookup)


@router.get("/{item_id}", response_model=CodingBacklogItemResponse)
async def get_coding_backlog_item(
    item_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = await _get_visible_backlog_item_or_404(db, item_id, current_user.id)
    user_lookup = await _build_backlog_user_lookup(db, current_user=current_user)
    return _to_response(item, current_user=current_user, user_lookup=user_lookup)


@router.patch("/{item_id}", response_model=CodingBacklogItemResponse)
async def update_coding_backlog_item(
    item_id: UUID,
    payload: CodingBacklogItemUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = await _get_visible_backlog_item_or_404(db, item_id, current_user.id)
    if str(item.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=403, detail="Only the backlog owner can edit this item"
        )
    collaboration_source = (
        item.collaboration if isinstance(item.collaboration, dict) else {}
    )
    if payload.title is not None:
        item.title = str(payload.title).strip()
    if payload.portfolio_goal is not None:
        item.portfolio_goal = str(payload.portfolio_goal).strip()
    if payload.scope is not None:
        item.scope = str(payload.scope or "auto").strip().lower() or "auto"
    if payload.priority is not None:
        item.priority = int(payload.priority)
    if payload.failure_symptom is not None:
        item.failure_symptom = str(payload.failure_symptom).strip() or None
    if payload.error_output is not None:
        item.error_output = str(payload.error_output).strip() or None
    if payload.file_paths is not None:
        item.file_paths = _normalize_str_list(payload.file_paths)
    if payload.commands is not None:
        item.commands = _normalize_str_list(payload.commands)
    if payload.auto_apply_enabled is not None:
        item.auto_apply_enabled = bool(payload.auto_apply_enabled)
    if payload.require_patch_pr is not None:
        item.require_patch_pr = bool(payload.require_patch_pr)
    if payload.visibility is not None:
        item.visibility = _normalize_visibility(payload.visibility)
        collaboration_source["visibility"] = item.visibility
    if payload.shared_with_user_ids is not None:
        item.shared_with_user_ids = (
            _normalize_uuid_list(payload.shared_with_user_ids, 200) or None
        )
        collaboration_source["shared_with_user_ids"] = item.shared_with_user_ids or []
    if payload.assigned_user_id is not None:
        item.assigned_user_id = payload.assigned_user_id
        collaboration_source["assigned_user_id"] = (
            str(payload.assigned_user_id) if payload.assigned_user_id else None
        )
    if payload.assigned_by_user_id is not None:
        item.assigned_by_user_id = payload.assigned_by_user_id
        collaboration_source["assigned_by_user_id"] = (
            str(payload.assigned_by_user_id) if payload.assigned_by_user_id else None
        )
    if payload.assigned_at is not None:
        item.assigned_at = payload.assigned_at
        collaboration_source["assigned_at"] = payload.assigned_at
    if payload.collaboration is not None:
        collaboration_source = payload.collaboration
    if (
        payload.visibility is not None
        or payload.shared_with_user_ids is not None
        or payload.assigned_user_id is not None
        or payload.assigned_by_user_id is not None
        or payload.assigned_at is not None
        or payload.collaboration is not None
    ):
        item.collaboration = _normalize_collaboration(
            collaboration_source, fallback_owner_user_id=str(item.user_id)
        )
    if payload.policy is not None:
        item.policy = _normalize_policy(payload.policy)
    if payload.lineage is not None:
        item.lineage = (
            deepcopy(payload.lineage) if isinstance(payload.lineage, dict) else None
        )
    if payload.decomposition is not None:
        item.decomposition = deepcopy(payload.decomposition)
    item.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(item)
    user_lookup = await _build_backlog_user_lookup(db, current_user=current_user)
    return _to_response(item, current_user=current_user, user_lookup=user_lookup)


@router.post("/{item_id}/action", response_model=CodingBacklogItemResponse)
async def act_on_coding_backlog_item(
    item_id: UUID,
    payload: CodingBacklogItemActionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    item = await _get_visible_backlog_item_or_404(db, item_id, current_user.id)
    action = str(payload.action or "").strip().lower()
    if action not in {
        "start",
        "pause",
        "resume",
        "cancel",
        "close",
        "assign_backlog",
        "clear_backlog_assignment",
        "update_backlog_note",
        "apply_override",
        "create_patch_pr",
        "keep_proposal_only",
        "relaunch_slice",
        "skip_slice",
    }:
        raise HTTPException(status_code=400, detail="Unsupported action")
    slice_id = str(payload.slice_id or "").strip() or None
    operator_note = str(payload.operator_note or "").strip() or None
    closure_reason = _normalize_closure_reason(payload.closure_reason)
    decomposition = _normalize_decomposition(item)
    slice_state = _find_slice(decomposition, slice_id)
    collaboration = _normalize_collaboration(
        item.collaboration, fallback_owner_user_id=str(item.user_id)
    )

    owner_only_actions = {
        "cancel",
        "apply_override",
        "create_patch_pr",
        "keep_proposal_only",
        "relaunch_slice",
        "skip_slice",
        "close",
    }
    collaborator_actions = {
        "assign_backlog",
        "clear_backlog_assignment",
        "update_backlog_note",
        "start",
        "resume",
        "pause",
    }
    if action in owner_only_actions and str(item.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=403, detail="Only the backlog owner can perform this action"
        )
    if action in collaborator_actions and not _is_backlog_visible_to_user(
        item, current_user.id
    ):
        raise HTTPException(status_code=403, detail="Backlog item not visible")

    if (
        action
        in {
            "apply_override",
            "create_patch_pr",
            "keep_proposal_only",
            "relaunch_slice",
            "skip_slice",
        }
        and slice_state is None
    ):
        raise HTTPException(
            status_code=400, detail="slice_id is required for this action"
        )

    if (
        slice_state is not None
        and not slice_state.get("allowed_slice_actions")
        and not bool(slice_state.get("awaiting_operator_action"))
        and action not in {"relaunch_slice", "skip_slice"}
    ):
        raise HTTPException(
            status_code=409, detail="Slice is not awaiting operator action"
        )

    if action == "assign_backlog":
        assigned_user_id = str(payload.assigned_user_id or current_user.id).strip()
        try:
            assigned_user = await db.get(User, UUID(assigned_user_id))
        except Exception:
            assigned_user = None
        if assigned_user is None or not bool(
            getattr(assigned_user, "is_active", False)
        ):
            raise HTTPException(status_code=422, detail="Assigned user not found")
        item.assigned_user_id = assigned_user.id
        item.assigned_by_user_id = current_user.id
        item.assigned_at = datetime.utcnow()
        collaboration = _normalize_collaboration(
            {
                **collaboration,
                "visibility": "shared"
                if bool(collaboration.get("shared_with_user_ids"))
                or str(item.user_id) != assigned_user_id
                else collaboration.get("visibility"),
                "assigned_user_id": assigned_user_id,
                "assigned_by_user_id": str(current_user.id),
                "assigned_at": item.assigned_at.isoformat(),
                "shared_with_user_ids": [
                    *list(collaboration.get("shared_with_user_ids") or []),
                    assigned_user_id,
                ],
                "note": operator_note or collaboration.get("note"),
            },
            fallback_owner_user_id=str(item.user_id),
        )
        item.visibility = str(
            collaboration.get("visibility") or item.visibility or "private"
        )
        item.shared_with_user_ids = (
            list(collaboration.get("shared_with_user_ids") or []) or None
        )
        item.collaboration = collaboration
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="assign_backlog",
                note=operator_note,
                metadata={"assigned_user_id": assigned_user_id},
            ),
        )
        item.decomposition = decomposition
    elif action == "clear_backlog_assignment":
        item.assigned_user_id = None
        item.assigned_by_user_id = None
        item.assigned_at = None
        collaboration = _normalize_collaboration(
            {
                **collaboration,
                "assigned_user_id": None,
                "assigned_by_user_id": None,
                "assigned_at": None,
                "note": operator_note or collaboration.get("note"),
            },
            fallback_owner_user_id=str(item.user_id),
        )
        item.collaboration = collaboration
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user", action="clear_backlog_assignment", note=operator_note
            ),
        )
        item.decomposition = decomposition
    elif action == "update_backlog_note":
        collaboration = _normalize_collaboration(
            {
                **collaboration,
                "note": operator_note,
            },
            fallback_owner_user_id=str(item.user_id),
        )
        item.collaboration = collaboration
        summary = item.latest_summary if isinstance(item.latest_summary, dict) else {}
        summary["operator_note"] = operator_note
        item.latest_summary = summary
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user", action="update_backlog_note", note=operator_note
            ),
        )
        item.decomposition = decomposition
    elif action in {"start", "resume"}:
        await _create_orchestrator_job(item, db=db, start_immediately=True)
    elif action == "pause":
        previous_status = str(item.status or "").strip() or None
        item.status = "paused"
        item.updated_at = datetime.utcnow()
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="pause",
                previous_status=previous_status,
                new_status="paused",
                note=operator_note,
            ),
        )
        item.decomposition = decomposition
    elif action == "cancel":
        if not closure_reason:
            raise HTTPException(
                status_code=400, detail="closure_reason is required for cancel"
            )
        previous_status = str(item.status or "").strip() or None
        item.status = "cancelled"
        item.completed_at = datetime.utcnow()
        item.updated_at = datetime.utcnow()
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="cancel",
                previous_status=previous_status,
                new_status="cancelled",
                note=operator_note,
                metadata={"closure_reason": closure_reason},
            ),
        )
        item.decomposition = decomposition
        _set_latest_summary(
            item,
            decomposition,
            status_value="cancelled",
            note=operator_note,
            extra={"closure_reason": closure_reason},
        )
    elif action == "close":
        if not closure_reason:
            raise HTTPException(
                status_code=400, detail="closure_reason is required for close"
            )
        previous_status = str(item.status or "").strip() or None
        item.status = (
            "completed"
            if closure_reason in {"fixed_through_backlog", "promoted_to_repair"}
            else "cancelled"
        )
        item.completed_at = datetime.utcnow()
        item.updated_at = datetime.utcnow()
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="close",
                previous_status=previous_status,
                new_status=item.status,
                note=operator_note,
                metadata={"closure_reason": closure_reason},
            ),
        )
        item.decomposition = decomposition
        _set_latest_summary(
            item,
            decomposition,
            status_value="closed",
            note=operator_note,
            extra={"closure_reason": closure_reason},
        )
    elif action == "apply_override" and slice_state is not None:
        proposal_id = str(
            slice_state.get("selected_proposal_id") or item.latest_proposal_id or ""
        ).strip()
        proposal_uuid = None
        try:
            proposal_uuid = UUID(proposal_id)
        except Exception:
            raise HTTPException(
                status_code=400, detail="Slice does not have a valid proposal to apply"
            )
        proposal = await db.get(CodePatchProposal, proposal_uuid)
        if not proposal or proposal.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Proposal not found")
        apply_job = await _spawn_slice_apply_job(
            item,
            slice_state,
            db=db,
            proposal_id=proposal_id,
            operator_note=operator_note,
        )
        item.child_job_ids = _append_unique(item.child_job_ids, str(apply_job.id))
        item.current_job_id = apply_job.id
        item.latest_apply_job_id = apply_job.id
        item.status = "running"
        previous_status = str(slice_state.get("status") or "").strip() or None
        slice_state["status"] = "applying"
        slice_state["apply_job_id"] = str(apply_job.id)
        slice_state["operator_decision"] = "apply_override"
        slice_state["operator_note"] = operator_note
        slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
        _clear_waiting_metadata(slice_state)
        _append_slice_timeline(
            slice_state,
            _timeline_entry(
                actor="user",
                action="apply_override_confirmed",
                previous_status=previous_status,
                new_status="applying",
                note=operator_note,
                related_job_id=str(apply_job.id),
                related_proposal_id=proposal_id,
            ),
        )
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="apply_override",
                previous_status="awaiting_operator",
                new_status="running",
                note=operator_note,
                related_job_id=str(apply_job.id),
                related_proposal_id=proposal_id,
            ),
        )
        decomposition["active_slice_id"] = slice_state.get("slice_id")
        decomposition["portfolio_progress"] = _recompute_portfolio_progress(
            decomposition
        )
        item.decomposition = decomposition
        _set_latest_summary(
            item,
            decomposition,
            status_value="apply_started",
            slice_state=slice_state,
            extra={
                "current_child_job_id": str(apply_job.id),
                "promotion_decision": "auto_applied",
                "selected_proposal_id": proposal_id,
                "waiting_on_operator_action": False,
            },
        )
    elif action == "create_patch_pr" and slice_state is not None:
        proposal_id = str(
            slice_state.get("selected_proposal_id") or item.latest_proposal_id or ""
        ).strip()
        proposal_uuid = None
        try:
            proposal_uuid = UUID(proposal_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Slice does not have a valid proposal to promote",
            )
        proposal = await db.get(CodePatchProposal, proposal_uuid)
        if not proposal or proposal.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Proposal not found")
        pr = PatchPR(
            user_id=current_user.id,
            source_id=proposal.source_id,
            title=f"{item.title}: {str(slice_state.get('title') or 'Patch')}"[:500],
            description=operator_note,
            status="draft",
            selected_proposal_id=proposal.id,
            proposal_ids=[str(proposal.id)],
            approvals=[],
            checks={
                "coding_backlog": {
                    "item_id": str(item.id),
                    "slice_id": str(slice_state.get("slice_id") or ""),
                    "operator_note": operator_note,
                }
            },
        )
        db.add(pr)
        await db.flush()
        previous_status = str(slice_state.get("status") or "").strip() or None
        slice_state["status"] = "patch_pr"
        slice_state["promotion_decision"] = "patch_pr"
        slice_state["patch_pr_id"] = str(pr.id)
        slice_state["operator_decision"] = "create_patch_pr"
        slice_state["operator_note"] = operator_note
        slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
        slice_state["completed_at"] = (
            slice_state.get("completed_at") or datetime.utcnow().isoformat()
        )
        _clear_waiting_metadata(slice_state)
        _append_slice_timeline(
            slice_state,
            _timeline_entry(
                actor="user",
                action="create_patch_pr",
                previous_status=previous_status,
                new_status="patch_pr",
                note=operator_note,
                related_proposal_id=proposal_id,
                related_patch_pr_id=str(pr.id),
            ),
        )
        _append_lineage_id(slice_state, "patch_pr_ids", str(pr.id))
        _append_lineage_id(slice_state, "proposal_ids", proposal_id)
        _append_artifact_history(slice_state, "patch_pr", str(pr.id), "Patch PR")
        _append_artifact_history(
            slice_state, "proposal", proposal_id, "Selected proposal"
        )
        _append_manual_promotion_history(
            slice_state,
            action="create_patch_pr",
            operator_note=operator_note,
            proposal_id=proposal_id,
            patch_pr_id=str(pr.id),
        )
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="create_patch_pr",
                previous_status="awaiting_operator",
                new_status="completed",
                note=operator_note,
                related_proposal_id=proposal_id,
                related_patch_pr_id=str(pr.id),
            ),
        )
        decomposition["completed_slices"] = _append_unique(
            decomposition.get("completed_slices"), slice_state.get("slice_id")
        )
        _upsert_promotion_decision(
            decomposition,
            {
                "slice_id": str(slice_state.get("slice_id") or ""),
                "title": str(slice_state.get("title") or ""),
                "decision": "patch_pr",
                "proposal_id": proposal_id,
                "blocked_reason": str(slice_state.get("blocked_reason") or "").strip()
                or None,
                "proposal_confidence": float(
                    slice_state.get("proposal_confidence", 0.0) or 0.0
                ),
                "patch_pr_id": str(pr.id),
            },
        )
        decomposition["active_slice_id"] = None
        decomposition["portfolio_progress"] = _recompute_portfolio_progress(
            decomposition
        )
        item.decomposition = decomposition
        item.status = "completed"
        item.completed_at = datetime.utcnow()
        _set_latest_summary(
            item,
            decomposition,
            status_value="completed",
            slice_state=slice_state,
            note="Operator created a patch PR for this slice.",
            extra={
                "promotion_decision": "patch_pr",
                "selected_proposal_id": proposal_id,
                "patch_pr_id": str(pr.id),
                "waiting_on_operator_action": False,
            },
        )
    elif action == "keep_proposal_only" and slice_state is not None:
        proposal_id = (
            str(
                slice_state.get("selected_proposal_id") or item.latest_proposal_id or ""
            ).strip()
            or None
        )
        previous_status = str(slice_state.get("status") or "").strip() or None
        slice_state["status"] = "proposal_only"
        slice_state["promotion_decision"] = "proposal_only"
        slice_state["operator_decision"] = "keep_proposal_only"
        slice_state["operator_note"] = operator_note
        slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
        slice_state["completed_at"] = (
            slice_state.get("completed_at") or datetime.utcnow().isoformat()
        )
        _clear_waiting_metadata(slice_state)
        _append_slice_timeline(
            slice_state,
            _timeline_entry(
                actor="user",
                action="keep_proposal_only",
                previous_status=previous_status,
                new_status="proposal_only",
                note=operator_note,
                related_proposal_id=proposal_id,
            ),
        )
        _append_lineage_id(slice_state, "proposal_ids", proposal_id)
        _append_artifact_history(
            slice_state, "proposal", proposal_id, "Selected proposal"
        )
        _append_manual_promotion_history(
            slice_state,
            action="keep_proposal_only",
            operator_note=operator_note,
            proposal_id=proposal_id,
        )
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="keep_proposal_only",
                previous_status="awaiting_operator",
                new_status="completed",
                note=operator_note,
                related_proposal_id=proposal_id,
            ),
        )
        decomposition["completed_slices"] = _append_unique(
            decomposition.get("completed_slices"), slice_state.get("slice_id")
        )
        _upsert_promotion_decision(
            decomposition,
            {
                "slice_id": str(slice_state.get("slice_id") or ""),
                "title": str(slice_state.get("title") or ""),
                "decision": "proposal_only",
                "proposal_id": proposal_id,
                "blocked_reason": str(slice_state.get("blocked_reason") or "").strip()
                or None,
                "proposal_confidence": float(
                    slice_state.get("proposal_confidence", 0.0) or 0.0
                ),
            },
        )
        decomposition["active_slice_id"] = None
        decomposition["portfolio_progress"] = _recompute_portfolio_progress(
            decomposition
        )
        item.decomposition = decomposition
        item.status = "completed"
        item.completed_at = datetime.utcnow()
        _set_latest_summary(
            item,
            decomposition,
            status_value="completed",
            slice_state=slice_state,
            note="Operator kept this slice as a reviewable proposal only.",
            extra={
                "promotion_decision": "proposal_only",
                "selected_proposal_id": proposal_id,
                "waiting_on_operator_action": False,
            },
        )
    elif action == "relaunch_slice" and slice_state is not None:
        previous_job_id = str(slice_state.get("child_job_id") or "").strip() or None
        repair_job = await _spawn_slice_repair_job(
            item, slice_state, db=db, operator_note=operator_note
        )
        item.child_job_ids = _append_unique(item.child_job_ids, str(repair_job.id))
        item.current_job_id = repair_job.id
        item.status = "running"
        slice_state["status"] = "retrying"
        slice_state["retry_count"] = (
            max(0, int(slice_state.get("retry_count", 0) or 0)) + 1
        )
        slice_state["child_job_id"] = str(repair_job.id)
        slice_state["operator_decision"] = "relaunch_slice"
        slice_state["operator_note"] = operator_note
        slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
        _clear_waiting_metadata(slice_state)
        _append_lineage_id(slice_state, "retry_from_job_ids", previous_job_id)
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="relaunch_slice",
                previous_status="awaiting_operator",
                new_status="running",
                note=operator_note,
                related_job_id=str(repair_job.id),
                metadata={"retried_from_job_id": previous_job_id},
            ),
        )
        decomposition["active_slice_id"] = slice_state.get("slice_id")
        decomposition["portfolio_progress"] = _recompute_portfolio_progress(
            decomposition
        )
        item.decomposition = decomposition
        _set_latest_summary(
            item,
            decomposition,
            status_value="repair_started",
            slice_state=slice_state,
            extra={
                "current_child_job_id": str(repair_job.id),
                "retry_from_job_id": str(slice_state.get("child_job_id") or ""),
                "waiting_on_operator_action": False,
            },
        )
    elif action == "skip_slice" and slice_state is not None:
        previous_status = str(slice_state.get("status") or "").strip() or None
        slice_state["status"] = "deferred"
        slice_state["operator_decision"] = "skip_slice"
        slice_state["operator_note"] = operator_note
        slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
        _clear_waiting_metadata(slice_state)
        _append_slice_timeline(
            slice_state,
            _timeline_entry(
                actor="user",
                action="skip_slice",
                previous_status=previous_status,
                new_status="deferred",
                note=operator_note,
            ),
        )
        _append_manual_promotion_history(
            slice_state, action="skip_slice", operator_note=operator_note
        )
        _append_backlog_timeline(
            decomposition,
            _timeline_entry(
                actor="user",
                action="skip_slice",
                previous_status="awaiting_operator",
                new_status="running",
                note=operator_note,
            ),
        )
        decomposition["active_slice_id"] = None
        decomposition["portfolio_progress"] = _recompute_portfolio_progress(
            decomposition
        )
        next_slice = None
        for row in decomposition.get("planned_slices") or []:
            if str((row or {}).get("status") or "").strip().lower() == "pending":
                next_slice = row
                break
        if next_slice is not None:
            repair_job = await _spawn_slice_repair_job(
                item, next_slice, db=db, operator_note=operator_note
            )
            item.child_job_ids = _append_unique(item.child_job_ids, str(repair_job.id))
            item.current_job_id = repair_job.id
            item.status = "running"
            next_slice["status"] = "repairing"
            next_slice["child_job_id"] = str(repair_job.id)
            decomposition["active_slice_id"] = next_slice.get("slice_id")
            decomposition["portfolio_progress"] = _recompute_portfolio_progress(
                decomposition
            )
            item.decomposition = decomposition
            _set_latest_summary(
                item,
                decomposition,
                status_value="repair_started",
                slice_state=next_slice,
                note="Previous slice deferred by operator; continuing with next pending slice.",
                extra={
                    "current_child_job_id": str(repair_job.id),
                    "waiting_on_operator_action": False,
                },
            )
        else:
            item.decomposition = decomposition
            item.status = "completed"
            item.completed_at = datetime.utcnow()
            _set_latest_summary(
                item,
                decomposition,
                status_value="completed",
                slice_state=slice_state,
                note="Slice deferred by operator; no further pending slices remain.",
                extra={"waiting_on_operator_action": False},
            )

    await db.commit()
    await db.refresh(item)
    user_lookup = await _build_backlog_user_lookup(db, current_user=current_user)
    return _to_response(item, current_user=current_user, user_lookup=user_lookup)
