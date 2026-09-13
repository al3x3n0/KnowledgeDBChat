"""LLM call snapshot endpoints (read-only for owners/admins).

Snapshots are recorded by LLMService when LLM_CALL_SNAPSHOT_ENABLED is on;
these endpoints let operators replay/debug the exact prompts and responses
of an agent job's LLM calls.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.llm_call_snapshot import LLMCallSnapshot
from app.models.user import User
from app.schemas.llm_call_snapshot import (
    LLMCallSnapshotResponse,
    LLMCallSnapshotSummary,
)
from app.services.auth_service import get_current_user

router = APIRouter()


def _is_admin(user: User) -> bool:
    try:
        return bool(user.is_admin())
    except Exception:
        return str(getattr(user, "role", "") or "").lower() == "admin"


@router.get("/", response_model=list[LLMCallSnapshotSummary])
async def list_llm_snapshots(
    job_id: Optional[UUID] = Query(default=None),
    phase: Optional[str] = Query(default=None, max_length=50),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(LLMCallSnapshot)
    if not _is_admin(current_user):
        query = query.where(LLMCallSnapshot.user_id == current_user.id)
    if job_id is not None:
        query = query.where(LLMCallSnapshot.job_id == job_id)
    if phase:
        query = query.where(LLMCallSnapshot.phase == phase)
    query = query.order_by(LLMCallSnapshot.created_at.asc()).offset(offset).limit(limit)
    result = await db.execute(query)
    rows = result.scalars().all()
    return [LLMCallSnapshotSummary.model_validate(row) for row in rows]


@router.get("/{snapshot_id}", response_model=LLMCallSnapshotResponse)
async def get_llm_snapshot(
    snapshot_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    snapshot = await db.get(LLMCallSnapshot, snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Not found")
    if snapshot.user_id != current_user.id and not _is_admin(current_user):
        raise HTTPException(status_code=404, detail="Not found")
    return LLMCallSnapshotResponse.model_validate(snapshot)
