"""
Research Inbox endpoints.

This API stores and exposes discovered items from continuous/recurring monitoring jobs
so users can triage and feed signals back into monitoring behavior.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
import sqlalchemy as sa
from sqlalchemy import select, desc, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.research_inbox import ResearchInboxItem
from app.models.user import User
from app.schemas.research_inbox import (
    ResearchInboxBulkFollowUpRelaunchRequest,
    ResearchInboxBulkFollowUpRelaunchResponse,
    ResearchInboxBulkFollowUpRelaunchResult,
    ResearchInboxFollowUpRelaunchRequest,
    ResearchInboxItemResponse,
    ResearchInboxListResponse,
    ResearchInboxItemUpdateRequest,
    ResearchInboxStatsResponse,
)
from app.api.endpoints.agent_jobs import (
    _apply_follow_up_policy_on_accept,
    _relaunch_follow_up_inbox_item,
)
from app.services.auth_service import get_current_user
from app.services.research_inbox_follow_up_service import _resolve_follow_up_opportunity_origin
from app.services.research_monitor_profile_service import research_monitor_profile_service


router = APIRouter()


async def _serialize_research_inbox_item(
    item: ResearchInboxItem,
    db: AsyncSession,
    *,
    follow_up_job: AgentJob | None = None,
) -> ResearchInboxItemResponse:
    job = follow_up_job
    if job is None and getattr(item, "follow_up_job_id", None):
        job = await db.get(AgentJob, item.follow_up_job_id)

    origin_source_kind = None
    origin_source_id = None
    origin_opportunity_id = None
    if job is not None:
        origin_source_kind, origin_source_id, origin_opportunity_id = _resolve_follow_up_opportunity_origin(job)

    response = ResearchInboxItemResponse.model_validate(item)
    return response.model_copy(
        update={
            "follow_up_last_job_id": getattr(item, "follow_up_job_id", None),
            "origin_source_kind": origin_source_kind,
            "origin_source_id": str(origin_source_id or "").strip() or None,
            "origin_opportunity_id": str(origin_opportunity_id or "").strip() or None,
        }
    )

def _extract_repo_urls(text: str) -> list[dict]:
    """
    Extract GitHub/GitLab repo URLs from a blob of text.

    Returns list of {provider, repo, url}.
    """
    import re

    s = (text or "")
    out: list[dict] = []
    seen: set[str] = set()

    # GitHub patterns
    for m in re.finditer(r"(https?://github\\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+))", s):
        url = m.group(1)
        owner = m.group(2)
        repo = m.group(3)
        repo_id = f"{owner}/{repo}"
        key = f"github:{repo_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"provider": "github", "repo": repo_id, "url": url})

    # GitLab patterns (best-effort project path)
    for m in re.finditer(r"(https?://gitlab\\.com/([A-Za-z0-9_\\-./]+))", s):
        url = m.group(1)
        path = m.group(2).strip("/")
        # Drop obvious non-project paths
        if path.count("/") < 1:
            continue
        repo_id = path.split("#")[0].split("?")[0]
        key = f"gitlab:{repo_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"provider": "gitlab", "repo": repo_id, "url": url})

    return out[:20]


@router.get("", response_model=ResearchInboxListResponse)
async def list_inbox_items(
    status: Optional[str] = Query(None, description="new | accepted | rejected"),
    item_type: Optional[str] = Query(None, description="Filter by type (e.g. document, arxiv)"),
    customer: Optional[str] = Query(None, description="Filter by customer tag"),
    job_id: Optional[str] = Query(None, description="Filter by source monitor job id"),
    q: Optional[str] = Query(None, min_length=2, max_length=200, description="Search title/summary"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(ResearchInboxItem).where(ResearchInboxItem.user_id == current_user.id)

    if status:
        query = query.where(ResearchInboxItem.status == status)
    if item_type:
        query = query.where(ResearchInboxItem.item_type == item_type)
    if customer:
        query = query.where(ResearchInboxItem.customer == customer)
    if job_id:
        try:
            query = query.where(ResearchInboxItem.job_id == UUID(str(job_id).strip()))
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid job_id filter")
    if q:
        like = f"%{q}%"
        query = query.where(
            or_(
                ResearchInboxItem.title.ilike(like),
                ResearchInboxItem.summary.ilike(like),
            )
        )

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = int(total_result.scalar() or 0)

    query = query.order_by(desc(ResearchInboxItem.discovered_at)).offset(offset).limit(limit)
    result = await db.execute(query)
    items = list(result.scalars().all())
    jobs_by_id: dict[object, AgentJob] = {}
    follow_up_job_ids = [item.follow_up_job_id for item in items if getattr(item, "follow_up_job_id", None)]
    if follow_up_job_ids:
        jobs_result = await db.execute(select(AgentJob).where(AgentJob.id.in_(follow_up_job_ids)))
        jobs_by_id = {job.id: job for job in jobs_result.scalars().all()}

    return ResearchInboxListResponse(
        items=[
            await _serialize_research_inbox_item(
                it,
                db,
                follow_up_job=jobs_by_id.get(it.follow_up_job_id),
            )
            for it in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=ResearchInboxStatsResponse)
async def inbox_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await db.execute(
            select(ResearchInboxItem.status, func.count())
            .where(ResearchInboxItem.user_id == current_user.id)
            .group_by(ResearchInboxItem.status)
        )
        rows = result.all()
        by_status = {str(r[0]): int(r[1] or 0) for r in rows}
        total = sum(by_status.values())
        return ResearchInboxStatsResponse(
            total=total,
            new=by_status.get("new", 0),
            accepted=by_status.get("accepted", 0),
            rejected=by_status.get("rejected", 0),
        )
    except Exception as exc:
        logger.error(f"Failed to compute inbox stats: {exc}")
        raise HTTPException(status_code=500, detail="Failed to compute inbox stats")


@router.patch("/{item_id}", response_model=ResearchInboxItemResponse)
async def update_inbox_item(
    item_id: str,
    payload: ResearchInboxItemUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        item_uuid = UUID(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item id")

    item = await db.get(ResearchInboxItem, item_uuid)
    if not item or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Inbox item not found")

    previous_status = str(item.status or "").strip().lower()
    if payload.status is not None:
        s = str(payload.status).strip().lower()
        if s not in {"new", "accepted", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        item.status = s
        if s != "accepted":
            item.follow_up_decision = None
            item.follow_up_policy_mode = None
            item.follow_up_launch_status = None
            item.follow_up_block_reason = None
            item.follow_up_budget_decision = None
            item.follow_up_budget_reason = None
            item.follow_up_budget_throttle_state = None
            item.follow_up_customer_budget_decision = None
            item.follow_up_customer_budget_reason = None
            item.follow_up_customer_budget_throttle_state = None
            item.follow_up_recommendation_key = None
            item.follow_up_operator_decision = None
            item.follow_up_operator_note = None
            item.follow_up_operator_acted_at = None
            item.follow_up_operator_user_id = None
            item.follow_up_job_id = None
            item.follow_up_chain_definition_id = None
            item.follow_up_launched_at = None
            item.follow_up_outcome_status = None
            item.follow_up_outcome_recorded_at = None
            item.follow_up_outcome_summary = None

    if payload.feedback is not None:
        item.feedback = (payload.feedback or "").strip() or None

    if payload.metadata_patch is not None:
        patch = payload.metadata_patch if isinstance(payload.metadata_patch, dict) else {}
        meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        # Allowlisted metadata updates (avoid clobbering system-populated fields like repos).
        if "paper_algo_run_demo_check" in patch:
            meta["paper_algo_run_demo_check"] = bool(patch.get("paper_algo_run_demo_check"))
        if "paper_algo_entrypoint" in patch:
            ep_val = patch.get("paper_algo_entrypoint")
            if ep_val is None:
                meta["paper_algo_entrypoint"] = None
            else:
                ep_raw = str(ep_val).strip()
                if not ep_raw:
                    meta["paper_algo_entrypoint"] = None
                    item.item_metadata = meta
                    await db.commit()
                    await db.refresh(item)
                    return ResearchInboxItemResponse.model_validate(item)
                ep = ep_raw.replace("\\", "/").strip()
                while ep.startswith("./"):
                    ep = ep[2:]
                if ep.startswith("/") or ep.startswith("~") or ":" in ep:
                    raise HTTPException(status_code=422, detail="Invalid paper_algo_entrypoint (absolute paths not allowed)")
                if any(part == ".." for part in ep.split("/")):
                    raise HTTPException(status_code=422, detail="Invalid paper_algo_entrypoint ('..' not allowed)")
                if any(ch.isspace() for ch in ep):
                    raise HTTPException(status_code=422, detail="Invalid paper_algo_entrypoint (whitespace not allowed)")
                if not ep.endswith(".py"):
                    raise HTTPException(status_code=422, detail="Invalid paper_algo_entrypoint (must end with .py)")
                meta["paper_algo_entrypoint"] = ep[:200]
        item.item_metadata = meta

    if item.status == "accepted" and previous_status != "accepted":
        await _apply_follow_up_policy_on_accept(
            item=item,
            current_user=current_user,
            db=db,
        )

    await db.commit()
    await db.refresh(item)

    # Recompute monitor profile for this customer (best-effort).
    try:
        await research_monitor_profile_service.recompute_profile(
            db=db, user_id=current_user.id, customer=item.customer
        )
    except Exception:
        pass

    return await _serialize_research_inbox_item(item, db)


class ResearchInboxBulkUpdateRequest(ResearchInboxItemUpdateRequest):
    item_ids: list[UUID]


@router.patch("/bulk")
async def bulk_update_inbox_items(
    payload: ResearchInboxBulkUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Bulk update status/feedback for multiple inbox items owned by the current user.
    """
    if not payload.item_ids:
        raise HTTPException(status_code=422, detail="item_ids is required")

    new_status = None
    if payload.status is not None:
        s = str(payload.status).strip().lower()
        if s not in {"new", "accepted", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        new_status = s

    new_feedback = None
    if payload.feedback is not None:
        new_feedback = (payload.feedback or "").strip() or None

    if new_status is None and payload.feedback is None:
        return {"updated": 0}

    # Capture impacted customers so we can recompute profiles after update.
    try:
        cust_res = await db.execute(
            select(ResearchInboxItem.customer)
            .where(
                ResearchInboxItem.user_id == current_user.id,
                ResearchInboxItem.id.in_(payload.item_ids),
            )
        )
        customers = {r[0] for r in cust_res.all()}
    except Exception:
        customers = set()

    try:
        if new_status == "accepted":
            result = await db.execute(
                select(ResearchInboxItem).where(
                    ResearchInboxItem.user_id == current_user.id,
                    ResearchInboxItem.id.in_(payload.item_ids),
                )
            )
            items = list(result.scalars().all())
            updated = 0
            for item in items:
                previous_status = str(item.status or "").strip().lower()
                item.status = "accepted"
                if payload.feedback is not None:
                    item.feedback = new_feedback
                if previous_status != "accepted":
                    await _apply_follow_up_policy_on_accept(
                        item=item,
                        current_user=current_user,
                        db=db,
                    )
                updated += 1
            await db.commit()
        else:
            values: dict = {}
            if new_status is not None:
                values["status"] = new_status
                if new_status != "accepted":
                    values.update(
                        {
                            "follow_up_decision": None,
                            "follow_up_policy_mode": None,
                            "follow_up_launch_status": None,
                            "follow_up_block_reason": None,
                            "follow_up_budget_decision": None,
                            "follow_up_budget_reason": None,
                            "follow_up_budget_throttle_state": None,
                            "follow_up_customer_budget_decision": None,
                            "follow_up_customer_budget_reason": None,
                            "follow_up_customer_budget_throttle_state": None,
                            "follow_up_recommendation_key": None,
                            "follow_up_operator_decision": None,
                            "follow_up_operator_note": None,
                            "follow_up_operator_acted_at": None,
                            "follow_up_operator_user_id": None,
                            "follow_up_job_id": None,
                            "follow_up_chain_definition_id": None,
                            "follow_up_launched_at": None,
                            "follow_up_outcome_status": None,
                            "follow_up_outcome_recorded_at": None,
                            "follow_up_outcome_summary": None,
                        }
                    )
            if payload.feedback is not None:
                values["feedback"] = new_feedback

            result = await db.execute(
                sa.update(ResearchInboxItem)
                .where(
                    ResearchInboxItem.user_id == current_user.id,
                    ResearchInboxItem.id.in_(payload.item_ids),
                )
                .values(**values)
            )
            updated = int(result.rowcount or 0)
            await db.commit()

        # Recompute profiles for impacted customers (best-effort).
        for cust in customers:
            try:
                await research_monitor_profile_service.recompute_profile(
                    db=db, user_id=current_user.id, customer=cust
                )
            except Exception:
                pass

        return {"updated": updated}
    except Exception as exc:
        logger.error(f"Failed to bulk update inbox items: {exc}")
        raise HTTPException(status_code=500, detail="Failed to bulk update inbox items")


@router.post("/{item_id}/relaunch-follow-up", response_model=ResearchInboxItemResponse)
async def relaunch_inbox_follow_up(
    item_id: str,
    payload: ResearchInboxFollowUpRelaunchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        item_uuid = UUID(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item id")

    item = await db.get(ResearchInboxItem, item_uuid)
    if not item or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Inbox item not found")
    if str(item.status or "").strip().lower() != "accepted":
        raise HTTPException(status_code=400, detail="Only accepted inbox items can relaunch a follow-up")

    await _relaunch_follow_up_inbox_item(
        item=item,
        operator_note=payload.operator_note,
        db=db,
        current_user=current_user,
    )

    await db.commit()
    await db.refresh(item)
    return await _serialize_research_inbox_item(item, db)


@router.post("/follow-up-bulk-relaunch", response_model=ResearchInboxBulkFollowUpRelaunchResponse)
async def bulk_relaunch_inbox_follow_up(
    payload: ResearchInboxBulkFollowUpRelaunchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    requested_ids: list[UUID] = []
    seen_ids: set[UUID] = set()
    for item_id in payload.item_ids:
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        requested_ids.append(item_id)

    results: list[ResearchInboxBulkFollowUpRelaunchResult] = []
    applied = 0
    for item_id in requested_ids:
        item = await db.get(ResearchInboxItem, item_id)
        if not item or item.user_id != current_user.id:
            results.append(
                ResearchInboxBulkFollowUpRelaunchResult(
                    item_id=item_id,
                    ok=False,
                    error="Inbox item not found",
                )
            )
            continue
        try:
            response = await _relaunch_follow_up_inbox_item(
                item=item,
                operator_note=payload.operator_note,
                db=db,
                current_user=current_user,
            )
            applied += 1
            results.append(
                ResearchInboxBulkFollowUpRelaunchResult(
                    item_id=item_id,
                    ok=True,
                    follow_up_job_id=response.follow_up_job_id,
                )
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Failed to relaunch follow-up"
            results.append(
                ResearchInboxBulkFollowUpRelaunchResult(
                    item_id=item_id,
                    ok=False,
                    error=detail,
                )
            )

    await db.commit()
    for item_id, result in zip(requested_ids, results):
        if result.ok:
            item = await db.get(ResearchInboxItem, item_id)
            if item is not None:
                await db.refresh(item)

    return ResearchInboxBulkFollowUpRelaunchResponse(
        requested_count=len(requested_ids),
        applied=applied,
        failed=len(requested_ids) - applied,
        results=results,
    )


@router.post("/{item_id}/extract-repos")
async def extract_repos_for_inbox_item(
    item_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Extract GitHub/GitLab repository URLs from an arXiv inbox item (title/summary/url).

    Stores results into `item.metadata.repos` for downstream actions (e.g. repo ingestion, code agent).
    """
    try:
        item_uuid = UUID(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item id")

    item = await db.get(ResearchInboxItem, item_uuid)
    if not item or item.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Inbox item not found")

    if item.item_type != "arxiv":
        raise HTTPException(status_code=422, detail="Only supported for arxiv inbox items")

    meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
    combined = " ".join([str(item.title or ""), str(item.summary or ""), str(item.url or ""), str(meta.get("entry_url") or ""), str(meta.get("pdf_url") or "")])
    repos = _extract_repo_urls(combined)

    # If none found, try fetching the arXiv abs page (best-effort).
    if not repos:
        try:
            import httpx

            entry_url = str(meta.get("entry_url") or item.url or "").strip()
            if entry_url:
                async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "KnowledgeDBChat-RepoScout"}) as client:
                    resp = await client.get(entry_url)
                    if resp.status_code == 200:
                        repos = _extract_repo_urls(resp.text)
        except Exception:
            repos = repos or []

    meta["repos"] = repos
    item.item_metadata = meta
    await db.commit()
    return {"item_id": str(item.id), "repos": repos, "count": len(repos)}
