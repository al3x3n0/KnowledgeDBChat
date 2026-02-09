"""
Code patch proposal endpoints.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, desc, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.models.code_patch_proposal import CodePatchProposal
from app.models.user import User
from app.schemas.code_patch import (
    CodePatchProposalResponse,
    CodePatchProposalListResponse,
    CodePatchProposalListItem,
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
    return PlainTextResponse(content=proposal.diff_unified or "", media_type="text/plain")


@router.patch("/{proposal_id}", response_model=CodePatchProposalResponse)
async def update_code_patch(
    proposal_id: UUID,
    payload: CodePatchProposalUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proposal = await db.get(CodePatchProposal, proposal_id)
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    if payload.status is not None:
        s = str(payload.status).strip().lower()
        if s not in {"proposed", "applied", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        proposal.status = s

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
    """
    Apply this patch proposal to KnowledgeDB code documents (best-effort).

    This updates Document.content for matching files under the proposal's `source_id`.
    """
    from hashlib import sha256

    from app.models.document import Document
    from app.services.code_patch_apply_service import code_patch_apply_service, UnifiedDiffApplyError
    from app.services.document_service import DocumentService

    proposal = await db.get(CodePatchProposal, proposal_id)
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    if not proposal.source_id:
        raise HTTPException(status_code=422, detail="Proposal missing source_id (cannot apply)")

    try:
        file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
    except UnifiedDiffApplyError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid diff: {exc}")

    if not file_diffs:
        raise HTTPException(status_code=422, detail="No file diffs found")

    service = DocumentService()
    applied: list[dict] = []
    errors: list[dict] = []

    for fd in file_diffs:
        path = (fd.path or "").strip()
        if not path:
            continue

        # Try to find the code document for this path.
        res = await db.execute(
            select(Document)
            .where(
                and_(
                    Document.source_id == proposal.source_id,
                    or_(
                        Document.file_path == path,
                        Document.source_identifier == path,
                        Document.title == path,
                    ),
                )
            )
            .limit(1)
        )
        doc = res.scalar_one_or_none()
        if not doc:
            errors.append({"path": path, "error": "Document not found"})
            continue

        try:
            new_text, debug = code_patch_apply_service.apply_to_text(doc.content or "", fd)
        except UnifiedDiffApplyError as exc:
            errors.append({"path": path, "error": str(exc)})
            continue

        doc.content = new_text
        doc.content_hash = sha256(new_text.encode("utf-8")).hexdigest()
        doc.is_processed = False
        doc.processing_error = None
        applied.append({"path": path, "document_id": str(doc.id), "debug": debug})

        # Best-effort reprocess to update chunks/vector index.
        try:
            await service.reprocess_document(doc.id, db, user_id=current_user.id)
        except Exception:
            pass

    proposal.proposal_metadata = proposal.proposal_metadata if isinstance(proposal.proposal_metadata, dict) else {}
    proposal.proposal_metadata["apply_results"] = {"applied": applied, "errors": errors}
    if errors:
        proposal.status = "proposed"
    else:
        proposal.status = "applied"

    await db.commit()

    return {
        "proposal_id": str(proposal.id),
        "status": proposal.status,
        "applied": applied,
        "errors": errors,
    }
