"""Artifact draft endpoints (human-in-the-loop review for non-code artifacts)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.artifact_draft import ArtifactDraft
from app.models.presentation import PresentationJob
from app.models.repo_report import RepoReportJob
from app.models.user import User
from app.schemas.artifact_draft import (
    ArtifactDraftApproveRequest,
    ArtifactDraftListItem,
    ArtifactDraftListResponse,
    ArtifactDraftResponse,
    ArtifactDraftSubmitRequest,
)
from app.services.auth_service import get_current_user
from app.services.storage_service import StorageService

router = APIRouter()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _is_admin(user: User) -> bool:
    try:
        return bool(user.is_admin())
    except Exception:
        return str(getattr(user, "role", "") or "").lower() == "admin"


@router.get("", response_model=ArtifactDraftListResponse)
async def list_artifact_drafts(
    artifact_type: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ArtifactDraft)
    if not _is_admin(current_user):
        stmt = stmt.where(ArtifactDraft.user_id == current_user.id)
    if artifact_type:
        stmt = stmt.where(ArtifactDraft.artifact_type == str(artifact_type).strip())
    if status_filter:
        stmt = stmt.where(ArtifactDraft.status == str(status_filter).strip().lower())

    count_q = select(func.count()).select_from(stmt.subquery())
    total = int((await db.execute(count_q)).scalar() or 0)

    stmt = stmt.order_by(desc(ArtifactDraft.created_at)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    items = list(res.scalars().all())

    return ArtifactDraftListResponse(
        items=[ArtifactDraftListItem.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{draft_id}", response_model=ArtifactDraftResponse)
async def get_artifact_draft(
    draft_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await db.get(ArtifactDraft, draft_id)
    if not draft or (draft.user_id != current_user.id and not _is_admin(current_user)):
        raise HTTPException(status_code=404, detail="Not found")
    return ArtifactDraftResponse.model_validate(draft)


@router.post("/from-presentation/{job_id}", response_model=ArtifactDraftResponse, status_code=status.HTTP_201_CREATED)
async def create_draft_from_presentation(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(PresentationJob, job_id)
    if not job or (job.user_id != current_user.id and not _is_admin(current_user)):
        raise HTTPException(status_code=404, detail="Presentation job not found")
    if job.status != "completed":
        raise HTTPException(status_code=422, detail="Presentation is not completed")
    if not job.generated_outline:
        raise HTTPException(status_code=422, detail="Presentation has no generated outline")

    draft = ArtifactDraft(
        user_id=job.user_id,
        artifact_type="presentation",
        source_id=job.id,
        title=f"Presentation: {job.title}"[:500],
        description=None,
        status="draft",
        approvals=[],
        draft_payload={
            "presentation_job_id": str(job.id),
            "title": job.title,
            "topic": job.topic,
            "style": job.style,
            "slide_count": job.slide_count,
            "include_diagrams": bool(job.include_diagrams),
            "source_document_ids": job.source_document_ids or [],
            "retrieval_trace_id": str(job.retrieval_trace_id) if getattr(job, "retrieval_trace_id", None) else None,
            "generated_outline": job.generated_outline,
            "file_path": job.file_path,
            "sources_used": {
                "source_document_ids": job.source_document_ids or [],
                "retrieval_trace_id": str(job.retrieval_trace_id) if getattr(job, "retrieval_trace_id", None) else None,
            },
        },
    )
    db.add(draft)
    await db.commit()
    await db.refresh(draft)
    return ArtifactDraftResponse.model_validate(draft)


@router.post("/from-repo-report/{job_id}", response_model=ArtifactDraftResponse, status_code=status.HTTP_201_CREATED)
async def create_draft_from_repo_report(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(RepoReportJob, job_id)
    if not job or (job.user_id != current_user.id and not _is_admin(current_user)):
        raise HTTPException(status_code=404, detail="Repo report job not found")
    if job.status != "completed":
        raise HTTPException(status_code=422, detail="Repo report is not completed")
    if not job.file_path:
        raise HTTPException(status_code=422, detail="Repo report has no output file")

    draft = ArtifactDraft(
        user_id=job.user_id,
        artifact_type="repo_report",
        source_id=job.id,
        title=f"Repo report: {job.title}"[:500],
        description=None,
        status="draft",
        approvals=[],
        draft_payload={
            "repo_report_job_id": str(job.id),
            "title": job.title,
            "repo_name": job.repo_name,
            "repo_url": job.repo_url,
            "repo_type": job.repo_type,
            "output_format": job.output_format,
            "sections": job.sections,
            "style": job.style,
            "slide_count": job.slide_count,
            "include_diagrams": bool(job.include_diagrams),
            "analysis_data": job.analysis_data,
            "file_path": job.file_path,
            "sources_used": {
                "repo_url": job.repo_url,
                "repo_name": job.repo_name,
                "repo_type": job.repo_type,
            },
        },
    )
    db.add(draft)
    await db.commit()
    await db.refresh(draft)
    return ArtifactDraftResponse.model_validate(draft)


@router.post("/{draft_id}/submit", response_model=ArtifactDraftResponse)
async def submit_draft_for_review(
    draft_id: UUID,
    payload: ArtifactDraftSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await db.get(ArtifactDraft, draft_id)
    if not draft or draft.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    if draft.status != "draft":
        raise HTTPException(status_code=422, detail="Draft is not in draft status")

    approvals = draft.approvals if isinstance(draft.approvals, list) else []
    if payload.note:
        approvals.append({"user_id": str(current_user.id), "role": "submitter", "at": _now_iso(), "note": payload.note})
    draft.approvals = approvals
    draft.status = "in_review"

    await db.commit()
    await db.refresh(draft)
    return ArtifactDraftResponse.model_validate(draft)


@router.post("/{draft_id}/approve", response_model=ArtifactDraftResponse)
async def approve_draft(
    draft_id: UUID,
    payload: ArtifactDraftApproveRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await db.get(ArtifactDraft, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Not found")

    role: Optional[str] = None
    if draft.user_id == current_user.id:
        role = "owner"
    elif _is_admin(current_user):
        role = "admin"

    if not role:
        raise HTTPException(status_code=403, detail="Not allowed")

    approvals = draft.approvals if isinstance(draft.approvals, list) else []
    approvals.append({"user_id": str(current_user.id), "role": role, "at": _now_iso(), "note": (payload.note or None)})
    draft.approvals = approvals

    owner_ok = any(a.get("role") == "owner" for a in approvals)
    admin_ok = any(a.get("role") == "admin" for a in approvals) or (role == "owner" and _is_admin(current_user))
    if owner_ok and admin_ok and draft.status in {"draft", "in_review"}:
        draft.status = "approved"

    await db.commit()
    await db.refresh(draft)
    return ArtifactDraftResponse.model_validate(draft)


@router.post("/{draft_id}/publish", response_model=ArtifactDraftResponse)
async def publish_draft(
    draft_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await db.get(ArtifactDraft, draft_id)
    if not draft or (draft.user_id != current_user.id and not _is_admin(current_user)):
        raise HTTPException(status_code=404, detail="Not found")

    if draft.status != "approved":
        raise HTTPException(status_code=422, detail="Draft must be approved before publishing")

    draft.status = "published"
    draft.published_payload = draft.draft_payload
    draft.published_at = datetime.utcnow()

    await db.commit()
    await db.refresh(draft)
    return ArtifactDraftResponse.model_validate(draft)


@router.get("/{draft_id}/download")
async def download_published_artifact(
    draft_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await db.get(ArtifactDraft, draft_id)
    if not draft or (draft.user_id != current_user.id and not _is_admin(current_user)):
        raise HTTPException(status_code=404, detail="Not found")

    if draft.status not in {"approved", "published"}:
        raise HTTPException(status_code=422, detail="Draft must be approved before download")

    payload = draft.published_payload if draft.status == "published" else draft.draft_payload
    file_path = (payload or {}).get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Artifact has no file")

    storage = StorageService()
    try:
        content = await storage.get_file_content(str(file_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact file not found")

    filename = str(file_path).split("/")[-1]
    media_type = "application/octet-stream"
    if draft.artifact_type == "presentation":
        media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        if not filename.endswith(".pptx"):
            filename = f"{draft.title}.pptx"
    elif draft.artifact_type == "repo_report":
        # Guess by extension
        if filename.endswith(".pptx"):
            media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif filename.endswith(".docx"):
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filename.endswith(".pdf"):
            media_type = "application/pdf"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Content-Length": str(len(content))},
    )
