"""
Admin endpoints for tool policies.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.tool_policy import ToolPolicy
from app.models.user import User
from app.schemas.tool_policy import AdminToolPolicyCreate, ToolPolicyResponse
from app.services.auth_service import require_admin


router = APIRouter()


@router.get("/tool-policies", response_model=List[ToolPolicyResponse])
async def list_tool_policies(
    subject_type: Optional[str] = Query(None),
    tool_name: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ToolPolicy).order_by(desc(ToolPolicy.created_at)).limit(limit)
    if subject_type:
        stmt = stmt.where(ToolPolicy.subject_type == subject_type)
    if tool_name:
        stmt = stmt.where(ToolPolicy.tool_name == tool_name)
    res = await db.execute(stmt)
    return [ToolPolicyResponse.model_validate(p) for p in res.scalars().all()]


@router.post("/tool-policies", response_model=ToolPolicyResponse, status_code=201)
async def create_tool_policy(
    payload: AdminToolPolicyCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    subject_type = str(payload.subject_type).strip().lower()
    subject_id: Optional[UUID] = None
    if payload.subject_id:
        try:
            subject_id = UUID(str(payload.subject_id))
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid subject_id")

    pol = ToolPolicy(
        subject_type=subject_type,
        subject_id=subject_id,
        subject_key=(payload.subject_key.strip() if payload.subject_key else None),
        tool_name=payload.tool_name.strip(),
        effect=payload.effect,
        require_approval=bool(payload.require_approval),
        constraints=payload.constraints,
    )
    db.add(pol)
    await db.commit()
    await db.refresh(pol)
    return ToolPolicyResponse.model_validate(pol)


@router.delete("/tool-policies/{policy_id}", status_code=204)
async def delete_tool_policy(
    policy_id: UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    stmt = delete(ToolPolicy).where(ToolPolicy.id == policy_id)
    res = await db.execute(stmt)
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")
    await db.commit()

