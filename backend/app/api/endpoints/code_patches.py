"""
Code patch proposal endpoints.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from loguru import logger
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.code_patch_proposal import CodePatchProposal
from app.models.user import User
from app.schemas.code_patch import (
    CodePatchProposalListItem,
    CodePatchProposalListResponse,
    CodePatchProposalResponse,
    CodePatchProposalUpdateRequest,
)
from app.services.auth_service import get_current_user

router = APIRouter()


@router.get("", response_model=CodePatchProposalListResponse)
async def list_code_patches(
    job_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(CodePatchProposal).where(CodePatchProposal.user_id == current_user.id)
    if job_id:
        try:
            stmt = stmt.where(CodePatchProposal.job_id == UUID(job_id))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid job_id")
    if status:
        stmt = stmt.where(CodePatchProposal.status == status)

    count_q = select(func.count()).select_from(stmt.subquery())
    total = int((await db.execute(count_q)).scalar() or 0)

    stmt = stmt.order_by(desc(CodePatchProposal.created_at)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    items = list(res.scalars().all())
    return CodePatchProposalListResponse(
        items=[CodePatchProposalListItem.model_validate(p) for p in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{proposal_id}", response_model=CodePatchProposalResponse)
async def get_code_patch(
    proposal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proposal = await db.get(CodePatchProposal, proposal_id)
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return CodePatchProposalResponse.model_validate(proposal)


@router.get("/{proposal_id}/download", response_class=PlainTextResponse)
async def download_code_patch(
    proposal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proposal = await db.get(CodePatchProposal, proposal_id)
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return PlainTextResponse(
        content=proposal.diff_unified or "", media_type="text/plain"
    )


@router.patch("/{proposal_id}", response_model=CodePatchProposalResponse)
async def update_code_patch(
    proposal_id: UUID,
    payload: CodePatchProposalUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proposal = await db.get(
        CodePatchProposal, proposal_id, with_for_update=True, populate_existing=True
    )
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    if payload.status is not None:
        status = str(payload.status).strip().lower()
        if status not in {"proposed", "applied", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        if status != proposal.status and "applied" in (status, proposal.status):
            raise HTTPException(
                status_code=409, detail="The apply workflow manages applied proposals"
            )
        proposal.status = status

    try:
        await db.commit()
        await db.refresh(proposal)
    except Exception as exc:
        logger.error(f"Failed to update code patch proposal: {exc}")
        raise HTTPException(status_code=500, detail="Failed to update")

    return CodePatchProposalResponse.model_validate(proposal)


@router.post("/{proposal_id}/apply")
async def apply_code_patch_to_kb(
    proposal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Apply a proposal through its approved Patch PR."""
    from app.api.endpoints.patch_prs import merge_patch_pr
    from app.models.patch_pr import PatchPR
    from app.schemas.patch_pr import PatchPRMergeRequest

    proposal = await db.get(CodePatchProposal, proposal_id)
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    review = await db.scalar(
        select(PatchPR)
        .where(
            PatchPR.user_id == current_user.id,
            PatchPR.selected_proposal_id == proposal.id,
            PatchPR.status.in_(("approved", "merged")),
        )
        .order_by((PatchPR.status == "merged").desc(), PatchPR.updated_at.desc())
        .limit(1)
        .with_for_update()
    )
    if review is None:
        raise HTTPException(
            status_code=409, detail="Approve this proposal in a Patch PR first"
        )

    result = await merge_patch_pr(
        review.id,
        PatchPRMergeRequest(dry_run=False),
        current_user,
        db,
    )
    return {
        "proposal_id": str(proposal_id),
        "status": "applied" if result.ok else "proposed",
        "applied": result.applied_files,
        "errors": result.errors,
    }
