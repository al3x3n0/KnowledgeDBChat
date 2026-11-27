"""Patch PR endpoints (PR-style review/merge flow for CodePatchProposal)."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
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


async def _owned_pr(
    db: AsyncSession, pr_id: UUID, user_id: UUID, *, lock: bool = False
) -> PatchPR:
    statement = select(PatchPR).where(PatchPR.id == pr_id, PatchPR.user_id == user_id)
    if lock:
        statement = statement.with_for_update().execution_options(
            populate_existing=True
        )
    pr = await db.scalar(statement)
    if pr is None:
        raise HTTPException(status_code=404, detail="Not found")
    return pr


async def _selected_proposal(
    db: AsyncSession, pr: PatchPR, *, lock: bool = False
) -> Optional[CodePatchProposal]:
    if not pr.selected_proposal_id:
        return None
    statement = select(CodePatchProposal).where(
        CodePatchProposal.id == pr.selected_proposal_id,
        CodePatchProposal.user_id == pr.user_id,
    )
    if lock:
        statement = statement.with_for_update().execution_options(
            populate_existing=True
        )
    return await db.scalar(statement)


def _diff_sha256(proposal: CodePatchProposal) -> str:
    return sha256((proposal.diff_unified or "").encode("utf-8")).hexdigest()


def _matching_approval(pr: PatchPR, proposal: CodePatchProposal) -> bool:
    digest = _diff_sha256(proposal)
    return any(
        isinstance(approval, dict)
        and approval.get("user_id") == str(pr.user_id)
        and approval.get("proposal_id") == str(proposal.id)
        and approval.get("diff_sha256") == digest
        for approval in (pr.approvals or [])
    )


async def _detail(db: AsyncSession, pr: PatchPR) -> PatchPRResponse:
    proposal = await _selected_proposal(db, pr)
    response = PatchPRResponse.model_validate(pr)
    if proposal and proposal.diff_unified:
        response.diff_unified = proposal.diff_unified
        response.diff_sha256 = _diff_sha256(proposal)
        response.approval_valid = _matching_approval(pr, proposal)
    return response


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
    pr = await _owned_pr(db, pr_id, current_user.id)
    return await _detail(db, pr)


@router.post("", response_model=PatchPRResponse)
async def create_patch_pr(
    payload: PatchPRCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source_uuid = _as_uuid(payload.source_id, field="source_id")
    initial_proposal_uuid = _as_uuid(
        payload.initial_proposal_id, field="initial_proposal_id"
    )

    proposal_ids: list[str] = []
    selected_proposal_id: Optional[UUID] = None
    if initial_proposal_uuid:
        proposal = await db.get(CodePatchProposal, initial_proposal_uuid)
        if not proposal or proposal.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="initial_proposal_id not found")
        if source_uuid and source_uuid != proposal.source_id:
            raise HTTPException(
                status_code=422, detail="Proposal belongs to a different source"
            )
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
    return await _detail(db, pr)


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

        cp = (
            results.get("code_patch")
            if isinstance(results.get("code_patch"), dict)
            else None
        )
        if isinstance(cp, dict) and str(cp.get("proposal_id") or "").strip():
            pid = str(cp["proposal_id"]).strip()
            if pid not in proposal_ids_set:
                proposal_ids.append(pid)
                proposal_ids_set.add(pid)

        hist = (
            results.get("code_patches")
            if isinstance(results.get("code_patches"), list)
            else []
        )
        for item in hist:
            if isinstance(item, dict) and str(item.get("proposal_id") or "").strip():
                pid = str(item["proposal_id"]).strip()
                if pid not in proposal_ids_set:
                    proposal_ids.append(pid)
                    proposal_ids_set.add(pid)

        er = (
            results.get("experiment_run")
            if isinstance(results.get("experiment_run"), dict)
            else None
        )
        if isinstance(er, dict):
            experiment_runs.append(
                {
                    "job_id": str(j.id),
                    "created_at": (
                        j.created_at.isoformat()
                        if getattr(j, "created_at", None)
                        else None
                    ),
                    "proposal_id": str(er.get("proposal_id") or "").strip() or None,
                    "ok": er.get("ok"),
                    "enabled": er.get("enabled"),
                    "ran": er.get("ran"),
                    "backend": er.get("backend"),
                    "commands": er.get("commands"),
                }
            )

    proposal_uuids: list[UUID] = []
    for pid in proposal_ids:
        try:
            proposal_uuids.append(UUID(pid))
        except ValueError:
            continue

    proposals = list(
        (
            await db.scalars(
                select(CodePatchProposal).where(
                    CodePatchProposal.user_id == current_user.id,
                    or_(
                        CodePatchProposal.id.in_(proposal_uuids),
                        CodePatchProposal.job_id.in_([job.id for job in jobs]),
                    ),
                )
            )
        ).all()
    )

    if not proposals:
        raise HTTPException(
            status_code=422, detail="No CodePatchProposal found in this chain"
        )

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
            "chain": {
                "root_job_id": str(root_uuid),
                "job_ids": [str(j.id) for j in jobs[:200]],
            },
            "experiment_runs": experiment_runs[:50],
            "proposal_strategy": strategy,
            "selected_proposal_id": str(selected.id),
            "created_at": _now_iso(),
        },
    )
    db.add(pr)
    await db.commit()
    await db.refresh(pr)
    return await _detail(db, pr)


@router.patch("/{pr_id}", response_model=PatchPRResponse)
async def update_patch_pr(
    pr_id: UUID,
    payload: PatchPRUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pr = await _owned_pr(db, pr_id, current_user.id, lock=True)
    if pr.status == "merged" and (
        payload.status is not None or payload.selected_proposal_id is not None
    ):
        raise HTTPException(
            status_code=409, detail="An applied patch cannot change state or proposal"
        )

    if payload.title is not None:
        pr.title = str(payload.title).strip()[:500]
    if payload.description is not None:
        pr.description = payload.description
    if payload.status is not None:
        s = str(payload.status).strip().lower()
        if s not in {"draft", "open", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        pr.status = s
        pr.approvals = []
    if payload.selected_proposal_id is not None:
        sel_uuid = _as_uuid(payload.selected_proposal_id, field="selected_proposal_id")
        if sel_uuid:
            proposal = await db.get(CodePatchProposal, sel_uuid)
            if not proposal or proposal.user_id != current_user.id:
                raise HTTPException(
                    status_code=404, detail="selected_proposal_id not found"
                )
            if pr.source_id and pr.source_id != proposal.source_id:
                raise HTTPException(
                    status_code=422, detail="Proposal belongs to a different source"
                )
            if pr.selected_proposal_id != proposal.id:
                pr.selected_proposal_id = proposal.id
                pr.approvals = []
                pr.status = "open"
            hist = list(pr.proposal_ids or [])
            pid = str(proposal.id)
            if pid not in set(str(x) for x in hist):
                hist.append(pid)
            pr.proposal_ids = hist
            pr.source_id = pr.source_id or proposal.source_id

    await db.commit()
    await db.refresh(pr)
    return await _detail(db, pr)


@router.post("/{pr_id}/approve", response_model=PatchPRResponse)
async def approve_patch_pr(
    pr_id: UUID,
    payload: PatchPRApproveRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pr = await _owned_pr(db, pr_id, current_user.id, lock=True)
    if pr.status not in {"draft", "open", "approved"}:
        raise HTTPException(status_code=409, detail="Patch is not open for approval")
    proposal = await _selected_proposal(db, pr, lock=True)
    if not proposal or not proposal.diff_unified:
        raise HTTPException(status_code=422, detail="No patch diff to approve")
    if proposal.status == "applied":
        raise HTTPException(status_code=409, detail="The proposal was already applied")
    if proposal.status == "rejected":
        raise HTTPException(
            status_code=409, detail="The selected proposal was rejected"
        )
    digest = _diff_sha256(proposal)
    if payload.diff_sha256 is not None and payload.diff_sha256 != digest:
        raise HTTPException(
            status_code=409, detail="The selected diff changed. Review it again."
        )

    if not _matching_approval(pr, proposal):
        pr.approvals = [
            {
                "user_id": str(current_user.id),
                "at": _now_iso(),
                "note": payload.note or None,
                "proposal_id": str(proposal.id),
                "diff_sha256": digest,
            }
        ]
    if pr.status != "approved":
        pr.status = "approved"

    await db.commit()
    await db.refresh(pr)
    return await _detail(db, pr)


@router.post("/{pr_id}/merge", response_model=PatchPRMergeResponse)
async def merge_patch_pr(
    pr_id: UUID,
    payload: PatchPRMergeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.services.code_patch_apply_service import (
        UnifiedDiffApplyError,
        code_patch_apply_service,
    )
    from app.services.coding_source_access import can_access_coding_source

    pr = await _owned_pr(db, pr_id, current_user.id, lock=True)
    proposal = await _selected_proposal(db, pr, lock=True)
    if proposal is None or not proposal.source_id:
        raise HTTPException(
            status_code=422, detail="No Library source on the selected proposal"
        )
    if pr.source_id and pr.source_id != proposal.source_id:
        raise HTTPException(
            status_code=422, detail="Proposal belongs to a different source"
        )
    if payload.diff_sha256 is not None and payload.diff_sha256 != _diff_sha256(
        proposal
    ):
        raise HTTPException(
            status_code=409, detail="The selected diff changed. Review it again."
        )
    if not await can_access_coding_source(
        db, current_user, proposal.source_id, write=True
    ):
        raise HTTPException(status_code=403, detail="Not authorized for this source")
    if not payload.dry_run and pr.status == "merged":
        return PatchPRMergeResponse(
            pr_id=str(pr.id),
            dry_run=False,
            ok=True,
            selected_proposal_id=str(proposal.id),
            applied_files=(pr.checks or {}).get("merge", {}).get("applied_files", []),
        )
    if pr.status not in {"draft", "open", "approved"}:
        raise HTTPException(status_code=409, detail="Patch is not open for application")
    if proposal.status == "applied":
        raise HTTPException(status_code=409, detail="The proposal was already applied")
    if proposal.status == "rejected":
        raise HTTPException(
            status_code=409, detail="The selected proposal was rejected"
        )
    if not payload.dry_run and not _matching_approval(pr, proposal):
        raise HTTPException(
            status_code=422, detail="Approve the selected diff before applying it"
        )

    try:
        file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
    except UnifiedDiffApplyError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid diff: {exc}") from exc
    if not file_diffs:
        raise HTTPException(status_code=422, detail="No file diffs found")
    metadata = proposal.proposal_metadata or {}
    if metadata.get("diff_truncated_from"):
        raise HTTPException(
            status_code=422, detail="The proposal contains a truncated diff"
        )

    applied_files, errors = await code_patch_apply_service.apply_to_library(
        db,
        source_id=proposal.source_id,
        user_id=current_user.id,
        file_diffs=file_diffs,
        dry_run=payload.dry_run,
        base_hashes=metadata.get("base_hashes"),
    )
    ok = not errors
    pr.checks = {
        **(pr.checks or {}),
        ("dry_run" if payload.dry_run else "merge"): {
            "at": _now_iso(),
            "dry_run": payload.dry_run,
            "proposal_id": str(proposal.id),
            "diff_sha256": _diff_sha256(proposal),
            "ok": ok,
            "applied_files": applied_files,
            "errors": errors,
        },
    }
    if ok and not payload.dry_run:
        pr.status = "merged"
        pr.merged_at = datetime.utcnow()
        pr.source_id = proposal.source_id
        proposal.status = "applied"
    response = PatchPRMergeResponse(
        pr_id=str(pr.id),
        dry_run=payload.dry_run,
        ok=ok,
        selected_proposal_id=str(proposal.id),
        applied_files=applied_files,
        errors=errors,
    )
    await db.commit()
    return response
