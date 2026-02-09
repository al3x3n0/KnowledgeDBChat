"""Patch PR endpoints (PR-style review/merge flow for CodePatchProposal)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.code_patch_proposal import CodePatchProposal
from app.models.patch_pr import PatchPR
from app.models.user import User
from app.schemas.patch_pr import (
    PatchPRApproveRequest,
    PatchPRCreateRequest,
    PatchPRFromChainRequest,
    PatchPRListItem,
    PatchPRListResponse,
    PatchPRMergeRequest,
    PatchPRMergeResponse,
    PatchPRResponse,
    PatchPRUpdateRequest,
)
from app.services.auth_service import get_current_user

router = APIRouter()


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _as_uuid(raw: Optional[str], *, field: str) -> Optional[UUID]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return UUID(s)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field}")


@router.get("", response_model=PatchPRListResponse)
async def list_patch_prs(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(PatchPR).where(PatchPR.user_id == current_user.id)
    if status:
        stmt = stmt.where(PatchPR.status == str(status).strip().lower())

    count_q = select(func.count()).select_from(stmt.subquery())
    total = int((await db.execute(count_q)).scalar() or 0)

    stmt = stmt.order_by(desc(PatchPR.created_at)).offset(offset).limit(limit)
    res = await db.execute(stmt)
    items = list(res.scalars().all())
    return PatchPRListResponse(
        items=[PatchPRListItem.model_validate(p) for p in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{pr_id}", response_model=PatchPRResponse)
async def get_patch_pr(
    pr_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pr = await db.get(PatchPR, pr_id)
    if not pr or pr.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return PatchPRResponse.model_validate(pr)


@router.post("", response_model=PatchPRResponse)
async def create_patch_pr(
    payload: PatchPRCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source_uuid = _as_uuid(payload.source_id, field="source_id")
    initial_proposal_uuid = _as_uuid(payload.initial_proposal_id, field="initial_proposal_id")

    proposal_ids: list[str] = []
    selected_proposal_id: Optional[UUID] = None
    if initial_proposal_uuid:
        proposal = await db.get(CodePatchProposal, initial_proposal_uuid)
        if not proposal or proposal.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="initial_proposal_id not found")
        selected_proposal_id = proposal.id
        proposal_ids = [str(proposal.id)]
        source_uuid = source_uuid or proposal.source_id

    pr = PatchPR(
        user_id=current_user.id,
        source_id=source_uuid,
        title=payload.title.strip()[:500],
        description=(payload.description or None),
        status="draft",
        selected_proposal_id=selected_proposal_id,
        proposal_ids=proposal_ids,
        checks={},
        approvals=[],
    )
    db.add(pr)
    await db.commit()
    await db.refresh(pr)
    return PatchPRResponse.model_validate(pr)


@router.post("/from-chain", response_model=PatchPRResponse)
async def create_patch_pr_from_chain(
    payload: PatchPRFromChainRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    root_uuid = _as_uuid(payload.root_job_id, field="root_job_id")
    if not root_uuid:
        raise HTTPException(status_code=400, detail="Invalid root_job_id")

    stmt = (
        select(AgentJob)
        .where(
            and_(
                AgentJob.user_id == current_user.id,
                or_(
                    AgentJob.id == root_uuid,
                    AgentJob.root_job_id == root_uuid,
                    AgentJob.parent_job_id == root_uuid,
                ),
            )
        )
        .order_by(AgentJob.created_at.asc())
    )
    res = await db.execute(stmt)
    jobs = list(res.scalars().all())
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found for root_job_id")

    proposal_ids: list[str] = []
    proposal_ids_set: set[str] = set()
    experiment_runs: list[dict] = []

    for j in jobs:
        results = j.results if isinstance(j.results, dict) else {}

        cp = results.get("code_patch") if isinstance(results.get("code_patch"), dict) else None
        if isinstance(cp, dict) and str(cp.get("proposal_id") or "").strip():
            pid = str(cp["proposal_id"]).strip()
            if pid not in proposal_ids_set:
                proposal_ids.append(pid)
                proposal_ids_set.add(pid)

        hist = results.get("code_patches") if isinstance(results.get("code_patches"), list) else []
        for item in hist:
            if isinstance(item, dict) and str(item.get("proposal_id") or "").strip():
                pid = str(item["proposal_id"]).strip()
                if pid not in proposal_ids_set:
                    proposal_ids.append(pid)
                    proposal_ids_set.add(pid)

        er = results.get("experiment_run") if isinstance(results.get("experiment_run"), dict) else None
        if isinstance(er, dict):
            experiment_runs.append(
                {
                    "job_id": str(j.id),
                    "created_at": (j.created_at.isoformat() if getattr(j, "created_at", None) else None),
                    "proposal_id": str(er.get("proposal_id") or "").strip() or None,
                    "ok": er.get("ok"),
                    "enabled": er.get("enabled"),
                    "ran": er.get("ran"),
                    "backend": er.get("backend"),
                    "commands": er.get("commands"),
                }
            )

    proposals: list[CodePatchProposal] = []
    for pid in proposal_ids:
        try:
            pu = UUID(pid)
        except Exception:
            continue
        p = await db.get(CodePatchProposal, pu)
        if p and p.user_id == current_user.id:
            proposals.append(p)

    if not proposals:
        raise HTTPException(status_code=422, detail="No CodePatchProposal found in this chain")

    proposals.sort(key=lambda p: p.created_at or datetime.min)

    selected: CodePatchProposal = proposals[-1]
    strategy = str(payload.proposal_strategy or "best_passing").strip().lower()
    if strategy not in {"best_passing", "latest"}:
        strategy = "best_passing"

    if strategy == "best_passing":
        ok_proposal_ids: set[str] = set()
        for er in experiment_runs:
            pid = str(er.get("proposal_id") or "").strip()
            if pid and er.get("ok") is True:
                ok_proposal_ids.add(pid)
        ok_candidates = [p for p in proposals if str(p.id) in ok_proposal_ids]
        if ok_candidates:
            selected = ok_candidates[-1]

    title = (payload.title or "").strip() or f"PatchPR: {selected.title}"
    description = payload.description or None

    pr = PatchPR(
        user_id=current_user.id,
        source_id=selected.source_id,
        title=title[:500],
        description=description,
        status=("open" if payload.open_after_create else "draft"),
        selected_proposal_id=selected.id,
        proposal_ids=[str(p.id) for p in proposals],
        approvals=[],
        checks={
            "chain": {"root_job_id": str(root_uuid), "job_ids": [str(j.id) for j in jobs[:200]]},
            "experiment_runs": experiment_runs[:50],
            "proposal_strategy": strategy,
            "selected_proposal_id": str(selected.id),
            "created_at": _now_iso(),
        },
    )
    db.add(pr)
    await db.commit()
    await db.refresh(pr)
    return PatchPRResponse.model_validate(pr)


@router.patch("/{pr_id}", response_model=PatchPRResponse)
async def update_patch_pr(
    pr_id: UUID,
    payload: PatchPRUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pr = await db.get(PatchPR, pr_id)
    if not pr or pr.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    if payload.title is not None:
        pr.title = str(payload.title).strip()[:500]
    if payload.description is not None:
        pr.description = payload.description
    if payload.status is not None:
        s = str(payload.status).strip().lower()
        if s not in {"draft", "open", "approved", "merged", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        pr.status = s
    if payload.selected_proposal_id is not None:
        sel_uuid = _as_uuid(payload.selected_proposal_id, field="selected_proposal_id")
        if sel_uuid:
            proposal = await db.get(CodePatchProposal, sel_uuid)
            if not proposal or proposal.user_id != current_user.id:
                raise HTTPException(status_code=404, detail="selected_proposal_id not found")
            pr.selected_proposal_id = proposal.id
            hist = pr.proposal_ids if isinstance(pr.proposal_ids, list) else []
            pid = str(proposal.id)
            if pid not in set(str(x) for x in hist):
                hist.append(pid)
            pr.proposal_ids = hist
            pr.source_id = pr.source_id or proposal.source_id

    await db.commit()
    await db.refresh(pr)
    return PatchPRResponse.model_validate(pr)


@router.post("/{pr_id}/approve", response_model=PatchPRResponse)
async def approve_patch_pr(
    pr_id: UUID,
    payload: PatchPRApproveRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pr = await db.get(PatchPR, pr_id)
    if not pr or pr.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    approvals = pr.approvals if isinstance(pr.approvals, list) else []
    approvals.append({"user_id": str(current_user.id), "at": _now_iso(), "note": (payload.note or None)})
    pr.approvals = approvals
    if pr.status in {"draft", "open"}:
        pr.status = "approved"

    await db.commit()
    await db.refresh(pr)
    return PatchPRResponse.model_validate(pr)


@router.post("/{pr_id}/merge", response_model=PatchPRMergeResponse)
async def merge_patch_pr(
    pr_id: UUID,
    payload: PatchPRMergeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from hashlib import sha256

    from app.models.document import Document
    from app.services.code_patch_apply_service import UnifiedDiffApplyError, code_patch_apply_service
    from app.services.document_service import DocumentService

    pr = await db.get(PatchPR, pr_id)
    if not pr or pr.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Not found")

    if bool(payload.require_approved):
        approvals = pr.approvals if isinstance(pr.approvals, list) else []
        if not approvals:
            raise HTTPException(status_code=422, detail="PR must be approved before merge")

    if not pr.selected_proposal_id:
        raise HTTPException(status_code=422, detail="PR has no selected_proposal_id")

    proposal = await db.get(CodePatchProposal, pr.selected_proposal_id)
    if not proposal or proposal.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Selected proposal not found")
    if not proposal.source_id:
        raise HTTPException(status_code=422, detail="Selected proposal missing source_id")

    try:
        file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
    except UnifiedDiffApplyError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid diff: {exc}")
    if not file_diffs:
        raise HTTPException(status_code=422, detail="No file diffs found")

    applied_files: list[dict] = []
    errors: list[dict] = []

    service = DocumentService()

    for fd in file_diffs:
        path = (fd.path or "").strip()
        if not path:
            continue

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

        applied_files.append({"path": path, "document_id": str(doc.id), "debug": debug})

        if not bool(payload.dry_run):
            doc.content = new_text
            doc.content_hash = sha256(new_text.encode("utf-8")).hexdigest()
            doc.is_processed = False
            doc.processing_error = None
            try:
                await service.reprocess_document(doc.id, db, user_id=current_user.id)
            except Exception:
                pass

    ok = len(errors) == 0
    merge_record: Dict[str, Any] = {
        "at": _now_iso(),
        "dry_run": bool(payload.dry_run),
        "proposal_id": str(proposal.id),
        "ok": ok,
        "applied_files": applied_files,
        "errors": errors,
    }

    pr.checks = pr.checks if isinstance(pr.checks, dict) else {}
    pr.checks["merge"] = merge_record

    if bool(payload.dry_run):
        await db.commit()
        return PatchPRMergeResponse(
            pr_id=str(pr.id),
            dry_run=True,
            ok=ok,
            selected_proposal_id=str(proposal.id),
            applied_files=applied_files,
            errors=errors,
        )

    if ok:
        pr.status = "merged"
        pr.merged_at = datetime.utcnow()
        pr.source_id = pr.source_id or proposal.source_id
        proposal.status = "applied"
    else:
        pr.status = pr.status if pr.status in {"draft", "open", "approved"} else "open"

    try:
        await db.commit()
    except Exception as exc:
        logger.error(f"Failed to merge PatchPR: {exc}")
        raise HTTPException(status_code=500, detail="Failed to merge")

    return PatchPRMergeResponse(
        pr_id=str(pr.id),
        dry_run=False,
        ok=ok,
        selected_proposal_id=str(proposal.id),
        applied_files=applied_files,
        errors=errors,
    )
