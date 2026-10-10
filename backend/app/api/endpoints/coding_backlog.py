from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.coding_backlog import CodingBacklogItem
from app.models.document import DocumentSource
from app.models.user import User
from app.modules.coding_backlog.application import backlog_actions, backlog_store
from app.modules.coding_backlog.application.backlog_slices import (
    normalize_closure_reason as _normalize_closure_reason,
)
from app.modules.coding_backlog.application.backlog_slices import (
    normalize_collaboration as _normalize_collaboration,
)
from app.modules.coding_backlog.application.backlog_slices import (
    normalize_str_list as _normalize_str_list,
)
from app.modules.coding_backlog.application.errors import ActionRefused
from app.schemas.coding_backlog import (
    CodingBacklogItemActionRequest,
    CodingBacklogItemCreate,
    CodingBacklogItemListResponse,
    CodingBacklogItemResponse,
    CodingBacklogItemUpdate,
)
from app.services.auth_service import get_current_user
from app.services.coding_backlog_items import (
    create_orchestrator_job as _create_orchestrator_job,
)
from app.services.coding_backlog_items import (
    default_decomposition as _default_decomposition,
)
from app.services.collaboration_service import build_collaboration_summary
from app.services.collaboration_service import (
    build_collaboration_user_lookup as _build_backlog_user_lookup,
)
from app.services.collaboration_service import normalize_collaboration_visibility
from app.services.config_values import uuid_list

router = APIRouter()

#: How each kind of refusal is told to a client. The application layer names
#: the kind; this is the only place that knows the numbers.
_REFUSAL_STATUS = {
    "invalid": 400,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "unknown_user": 422,
    "unavailable": 500,
}


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
    return uuid_list(values, limit)


def _normalize_visibility(value: Any) -> str:
    return normalize_collaboration_visibility(value)


_is_backlog_visible_to_user = backlog_store.is_visible_to


async def _get_visible_backlog_item_or_404(
    db: AsyncSession, item_id: UUID, user_id: UUID
) -> CodingBacklogItem:
    try:
        return await backlog_store.get_visible_item(db, item_id, user_id)
    except backlog_store.BacklogItemNotFound:
        raise HTTPException(status_code=404, detail="Not found")


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
                    or ""
                ).strip()
                or None,
                assigned_by_user_id=str(
                    collaboration.get("assigned_by_user_id")
                    or getattr(item, "assigned_by_user_id", "")
                    or ""
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
    rows, total = await backlog_store.list_visible_items(
        db,
        current_user.id,
        scope=visibility_scope,
        status=status_filter,
        assigned_user_id=assigned_user_id,
        limit=limit,
        offset=offset,
    )
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
    try:
        await backlog_actions.perform(
            db,
            item,
            current_user.id,
            action=payload.action,
            slice_id=payload.slice_id,
            operator_note=payload.operator_note,
            closure_reason=payload.closure_reason,
            assigned_user_id=payload.assigned_user_id,
        )
    except ActionRefused as refusal:
        raise HTTPException(
            status_code=_REFUSAL_STATUS[refusal.kind], detail=refusal.detail
        )

    await db.commit()
    await db.refresh(item)
    user_lookup = await _build_backlog_user_lookup(db, current_user=current_user)
    return _to_response(item, current_user=current_user, user_lookup=user_lookup)
