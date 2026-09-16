"""Research campaigns over HTTP.

The campaign subsystem shipped without any of this: `create_campaign` and
`advance` existed as service functions, a beat task drove active campaigns,
and nothing outside the process could start or see one. These routes are the
smallest surface that makes campaigns usable by a person -- create, list, read
-- and they are what the chat's campaign widget posts to.

Scoped by user like every other resource here; a campaign belongs to whoever
started it.
"""

from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.research_campaign import ResearchCampaign, ResearchCampaignItem
from app.models.user import User
from app.schemas.research_campaign import (
    ResearchCampaignCreate,
    ResearchCampaignItemResponse,
    ResearchCampaignListResponse,
    ResearchCampaignResponse,
)
from app.services import research_campaign_service

router = APIRouter()


async def _respond(
    campaign: ResearchCampaign, db: AsyncSession, *, with_items: bool = False
) -> ResearchCampaignResponse:
    """A campaign plus the service's own view of its progress.

    `with_items` is off for the list: loading every question of every campaign
    to render a page that shows none of them is a query per campaign for
    nothing.
    """
    base = ResearchCampaignResponse.model_validate(campaign)
    try:
        summary = await research_campaign_service.summarize(db, campaign)
    except Exception as exc:  # pragma: no cover - defensive
        # A campaign that cannot be summarised is still a campaign; losing the
        # whole response over a progress figure would be the wrong trade.
        logger.warning(f"Could not summarize campaign {campaign.id}: {exc}")
        summary = None
    update: Dict[str, Any] = {"summary": summary}
    if with_items:
        rows = (
            await db.execute(
                select(ResearchCampaignItem)
                .where(ResearchCampaignItem.campaign_id == campaign.id)
                .order_by(ResearchCampaignItem.created_at)
            )
        ).scalars()
        update["items"] = [
            ResearchCampaignItemResponse.model_validate(row) for row in rows
        ]
    return base.model_copy(update=update)


@router.post(
    "", response_model=ResearchCampaignResponse, status_code=status.HTTP_201_CREATED
)
async def create_research_campaign(
    payload: ResearchCampaignCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Start a campaign. It begins active; the beat task advances it."""
    try:
        campaign = await research_campaign_service.create_campaign(
            db,
            user_id=current_user.id,
            name=payload.name,
            goal=payload.goal,
            items=[item.model_dump() for item in payload.items],
            max_jobs=payload.max_jobs,
            job_template=payload.job_template or {},
        )
        await db.commit()
        await db.refresh(campaign)
    except ValueError as exc:
        # The service refuses a campaign with no goal, and says why.
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to create research campaign: {exc}")
        raise HTTPException(status_code=500, detail="Failed to create campaign")
    return await _respond(campaign, db)


@router.get("", response_model=ResearchCampaignListResponse)
async def list_research_campaigns(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    base = select(ResearchCampaign).where(ResearchCampaign.user_id == current_user.id)
    if str(status_filter or "").strip():
        base = base.where(ResearchCampaign.status == str(status_filter).strip())

    # count(*) rather than fetching every row to len() it.
    total = int(
        (await db.execute(select(func.count()).select_from(base.subquery()))).scalar()
        or 0
    )
    rows = list(
        (
            await db.execute(
                base.order_by(desc(ResearchCampaign.created_at))
                .offset(offset)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    items: List[ResearchCampaignResponse] = [await _respond(row, db) for row in rows]
    return ResearchCampaignListResponse(items=items, total=total)


@router.get("/{campaign_id}", response_model=ResearchCampaignResponse)
async def get_research_campaign(
    campaign_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    campaign = (
        await db.execute(
            select(ResearchCampaign).where(
                ResearchCampaign.id == campaign_id,
                ResearchCampaign.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return await _respond(campaign, db, with_items=True)
