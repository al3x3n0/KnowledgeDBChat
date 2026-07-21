"""
Collaboration visibility helpers shared across operator surfaces.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import and_, cast, or_, select, String
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.coding_backlog import CodingBacklogItem
from app.models.user import User


def _normalize_uuid_str_list(value: Any, limit: int = 200) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return items
    for raw in value:
        normalized = str(raw or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append(normalized)
        if len(items) >= limit:
            break
    return items


def normalize_collaboration_visibility(value: Any) -> str:
    return "shared" if str(value or "private").strip().lower() == "shared" else "private"


def _build_swarm_collaboration_payload(job: AgentJob) -> dict[str, Any]:
    results = job.results if isinstance(job.results, dict) else {}
    raw = results.get("swarm_collaboration") if isinstance(results.get("swarm_collaboration"), dict) else {}
    shared_with_user_ids = _normalize_uuid_str_list(raw.get("shared_with_user_ids"), 200)
    return {
        "owner_user_id": str(raw.get("owner_user_id") or job.user_id),
        "visibility": "shared" if bool(raw.get("shared_review")) or shared_with_user_ids else "private",
        "shared_with_user_ids": shared_with_user_ids,
        "assigned_user_id": str(raw.get("assigned_user_id") or "").strip() or None,
        "assigned_by_user_id": str(raw.get("assigned_by_user_id") or "").strip() or None,
        "assigned_at": str(raw.get("assigned_at") or "").strip() or None,
        "review_note": str(raw.get("review_note") or "").strip() or None,
    }


def _is_swarm_job_visible_to_user(job: AgentJob, user_id: UUID, *, is_admin: bool = False) -> bool:
    if is_admin or str(job.user_id) == str(user_id):
        return True
    collaboration = _build_swarm_collaboration_payload(job)
    if str(collaboration.get("assigned_user_id") or "").strip() == str(user_id):
        return True
    return str(user_id) in _normalize_uuid_str_list(collaboration.get("shared_with_user_ids"), 200)


def build_collaboration_summary(
    *,
    owner_user_id: Optional[str],
    visibility: Optional[str],
    shared_with_user_ids: Optional[list[str]],
    assigned_user_id: Optional[str] = None,
    assigned_by_user_id: Optional[str] = None,
    assigned_at: Optional[str] = None,
    note: Optional[str] = None,
    current_user_id: Optional[str] = None,
    user_lookup: Optional[dict[str, User]] = None,
) -> dict[str, Any]:
    normalized_owner_user_id = str(owner_user_id or "").strip() or None
    normalized_assigned_user_id = str(assigned_user_id or "").strip() or None
    normalized_assigned_by_user_id = str(assigned_by_user_id or "").strip() or None
    normalized_shared_with_user_ids = _normalize_uuid_str_list(shared_with_user_ids, 200)
    normalized_visibility = normalize_collaboration_visibility(visibility)
    owner = user_lookup.get(normalized_owner_user_id) if user_lookup and normalized_owner_user_id else None
    assignee = user_lookup.get(normalized_assigned_user_id) if user_lookup and normalized_assigned_user_id else None
    current_user_key = str(current_user_id or "").strip()
    return {
        "owner_user_id": normalized_owner_user_id,
        "owner_label": str(owner.full_name or owner.username or owner.email or owner.id).strip() if owner is not None else None,
        "assigned_user_id": normalized_assigned_user_id,
        "assignee_label": str(assignee.full_name or assignee.username or assignee.email or assignee.id).strip() if assignee is not None else None,
        "assigned_by_user_id": normalized_assigned_by_user_id,
        "assigned_at": assigned_at,
        "shared_with_user_ids": normalized_shared_with_user_ids,
        "visibility_scope": normalized_visibility,
        "is_owned_by_current_user": bool(current_user_key) and normalized_owner_user_id == current_user_key,
        "is_assigned_to_current_user": bool(current_user_key) and normalized_assigned_user_id == current_user_key,
        "is_shared_with_current_user": bool(current_user_key) and current_user_key in normalized_shared_with_user_ids,
        "note": str(note or "").strip() or None,
    }


async def list_collaboration_user_ids(
    db: AsyncSession,
    *,
    current_user: User,
) -> set[UUID]:
    if current_user.is_admin():
        rows = list((await db.execute(select(User.id).where(User.is_active.is_(True)))).scalars().all())
        return {row for row in rows if row is not None}

    visible_user_ids: set[UUID] = {current_user.id}
    user_id_str = str(current_user.id)

    backlog_rows = list(
        (
            await db.execute(
                select(CodingBacklogItem).where(
                    or_(
                        CodingBacklogItem.user_id == current_user.id,
                        CodingBacklogItem.assigned_user_id == current_user.id,
                        and_(
                            CodingBacklogItem.visibility == "shared",
                            cast(CodingBacklogItem.shared_with_user_ids, String).ilike(f"%{user_id_str}%"),
                        ),
                    )
                )
            )
        ).scalars().all()
    )
    for item in backlog_rows:
        for candidate in (item.user_id, item.assigned_user_id, item.assigned_by_user_id):
            if candidate is not None:
                visible_user_ids.add(candidate)
        for raw_id in _normalize_uuid_str_list(getattr(item, "shared_with_user_ids", None), 200):
            try:
                visible_user_ids.add(UUID(str(raw_id)))
            except Exception:
                continue

    swarm_rows = list(
        (
            await db.execute(
                select(AgentJob).where(
                    or_(
                        AgentJob.user_id == current_user.id,
                        cast(AgentJob.results, String).ilike("%swarm_collaboration%"),
                        cast(AgentJob.results, String).ilike(f"%{user_id_str}%"),
                    )
                )
            )
        ).scalars().all()
    )
    for job in swarm_rows:
        if not _is_swarm_job_visible_to_user(job, current_user.id, is_admin=current_user.is_admin()):
            continue
        collaboration = _build_swarm_collaboration_payload(job)
        for raw_id in [
            collaboration.get("owner_user_id"),
            collaboration.get("assigned_user_id"),
            collaboration.get("assigned_by_user_id"),
            *list(collaboration.get("shared_with_user_ids") or []),
        ]:
            if not str(raw_id or "").strip():
                continue
            try:
                visible_user_ids.add(UUID(str(raw_id)))
            except Exception:
                continue

    trace_rows = list(
        (
            await db.execute(
                select(AutonomyDecisionEvent).where(
                    or_(
                        AutonomyDecisionEvent.user_id == current_user.id,
                        AutonomyDecisionEvent.assigned_to_user_id == current_user.id,
                        AutonomyDecisionEvent.assigned_by_user_id == current_user.id,
                    )
                )
            )
        ).scalars().all()
    )
    for row in trace_rows:
        for candidate in (row.user_id, row.assigned_to_user_id, row.assigned_by_user_id):
            if candidate is not None:
                visible_user_ids.add(candidate)

    active_ids = set(
        row
        for row in (
            await db.execute(select(User.id).where(User.is_active.is_(True), User.id.in_(visible_user_ids)))
        ).scalars().all()
        if row is not None
    )
    active_ids.add(current_user.id)
    return active_ids


async def list_collaboration_users(
    db: AsyncSession,
    *,
    current_user: User,
    search: Optional[str] = None,
) -> list[User]:
    visible_user_ids = await list_collaboration_user_ids(db, current_user=current_user)
    query = select(User).where(User.is_active.is_(True), User.id.in_(visible_user_ids))
    if search:
        search_term = f"%{str(search).strip()}%"
        query = query.where(
            or_(
                User.username.ilike(search_term),
                User.email.ilike(search_term),
                User.full_name.ilike(search_term),
            )
        )
    query = query.order_by(User.username.asc())
    return list((await db.execute(query)).scalars().all())
