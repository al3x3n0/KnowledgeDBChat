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
from sqlalchemy import select, desc, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.research_inbox import ResearchInboxItem
from app.models.user import User
from app.schemas.research_inbox import (
    ResearchInboxItemResponse,
    ResearchInboxListResponse,
    ResearchInboxItemUpdateRequest,
    ResearchInboxStatsResponse,
)
from app.services.auth_service import get_current_user
from app.services.research_monitor_profile_service import research_monitor_profile_service


router = APIRouter()

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

    return ResearchInboxListResponse(
        items=[ResearchInboxItemResponse.model_validate(it) for it in items],
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

    if payload.status is not None:
        s = str(payload.status).strip().lower()
        if s not in {"new", "accepted", "rejected"}:
            raise HTTPException(status_code=422, detail="Invalid status")
        item.status = s

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

    await db.commit()
    await db.refresh(item)

    # Recompute monitor profile for this customer (best-effort).
    try:
        await research_monitor_profile_service.recompute_profile(
            db=db, user_id=current_user.id, customer=item.customer
        )
    except Exception:
        pass

    return ResearchInboxItemResponse.model_validate(item)


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

    values: dict = {}
    if new_status is not None:
        values["status"] = new_status
    if payload.feedback is not None:
        values["feedback"] = new_feedback

    if not values:
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
        result = await db.execute(
            sa.update(ResearchInboxItem)
            .where(
                ResearchInboxItem.user_id == current_user.id,
                ResearchInboxItem.id.in_(payload.item_ids),
            )
            .values(**values)
        )
        await db.commit()

        # Recompute profiles for impacted customers (best-effort).
        for cust in customers:
            try:
                await research_monitor_profile_service.recompute_profile(
                    db=db, user_id=current_user.id, customer=cust
                )
            except Exception:
                pass

        return {"updated": int(result.rowcount or 0)}
    except Exception as exc:
        logger.error(f"Failed to bulk update inbox items: {exc}")
        raise HTTPException(status_code=500, detail="Failed to bulk update inbox items")


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
