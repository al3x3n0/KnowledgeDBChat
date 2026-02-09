"""Retrieval trace endpoints (read-only for owners/admins)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.retrieval_trace import RetrievalTrace
from app.models.user import User
from app.schemas.retrieval_trace import RetrievalTraceResponse
from app.services.auth_service import get_current_user

router = APIRouter()


def _is_admin(user: User) -> bool:
    try:
        return bool(user.is_admin())
    except Exception:
        return str(getattr(user, "role", "") or "").lower() == "admin"


@router.get("/{trace_id}", response_model=RetrievalTraceResponse)
async def get_retrieval_trace(
    trace_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    trace = await db.get(RetrievalTrace, trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Not found")
    if trace.user_id != current_user.id and not _is_admin(current_user):
        raise HTTPException(status_code=404, detail="Not found")
    return RetrievalTraceResponse.model_validate(trace)

