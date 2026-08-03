"""HTTP boundary for AI Hub recommendation feedback."""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.core.feature_flags import get_str as get_feature_str
from app.models.agent_job import AgentJob
from app.models.ai_hub_recommendation_feedback import AIHubRecommendationFeedback
from app.models.user import User
from app.schemas.ai_hub_recommendation_feedback import (
    AIHubRecommendationFeedbackCreate,
    AIHubRecommendationFeedbackListResponse,
    AIHubRecommendationFeedbackResponse,
)
from app.schemas.customer_profile import CustomerProfile

router = APIRouter()


async def _require_owned_job(
    *,
    job_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> None:
    result = await db.execute(
        select(AgentJob).where(and_(AgentJob.id == job_id, AgentJob.user_id == user_id))
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Agent job not found")


@router.get(
    "/{job_id}/ai-hub/recommendation-feedback",
    response_model=AIHubRecommendationFeedbackListResponse,
)
async def list_ai_hub_recommendation_feedback(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List recommendation feedback for one user-owned AI Scientist job."""
    await _require_owned_job(
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )
    result = await db.execute(
        select(AIHubRecommendationFeedback)
        .where(
            and_(
                AIHubRecommendationFeedback.agent_job_id == job_id,
                AIHubRecommendationFeedback.user_id == current_user.id,
            )
        )
        .order_by(AIHubRecommendationFeedback.created_at.desc())
    )
    items = result.scalars().all()
    return AIHubRecommendationFeedbackListResponse(
        items=[
            AIHubRecommendationFeedbackResponse.model_validate(item) for item in items
        ],
        total=len(items),
    )


@router.post(
    "/{job_id}/ai-hub/recommendation-feedback",
    response_model=AIHubRecommendationFeedbackResponse,
)
async def create_ai_hub_recommendation_feedback(
    job_id: UUID,
    payload: AIHubRecommendationFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create or update feedback for one AI Scientist recommendation."""
    await _require_owned_job(
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )

    customer_profile_id = None
    customer_profile_name = None
    customer_keywords = None
    raw_profile = await get_feature_str("ai_hub_customer_profile")
    if raw_profile:
        try:
            profile = CustomerProfile.model_validate(json.loads(raw_profile))
            customer_profile_id = profile.id
            customer_profile_name = profile.name
            customer_keywords = profile.keywords
        except Exception:
            pass

    item_id = payload.item_id.strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="item_id required")

    existing_result = await db.execute(
        select(AIHubRecommendationFeedback)
        .where(
            and_(
                AIHubRecommendationFeedback.agent_job_id == job_id,
                AIHubRecommendationFeedback.user_id == current_user.id,
                AIHubRecommendationFeedback.workflow == payload.workflow,
                AIHubRecommendationFeedback.item_type == payload.item_type,
                AIHubRecommendationFeedback.item_id == item_id,
            )
        )
        .order_by(AIHubRecommendationFeedback.created_at.desc())
        .limit(1)
    )
    row = existing_result.scalar_one_or_none()
    if row is None:
        row = AIHubRecommendationFeedback(
            user_id=current_user.id,
            agent_job_id=job_id,
            customer_profile_id=customer_profile_id,
            customer_profile_name=customer_profile_name,
            customer_keywords=customer_keywords,
            workflow=payload.workflow,
            item_type=payload.item_type,
            item_id=item_id,
            decision=payload.decision,
            reason=payload.reason,
        )
        db.add(row)
    else:
        row.decision = payload.decision
        row.reason = payload.reason
        row.customer_profile_id = customer_profile_id
        row.customer_profile_name = customer_profile_name
        row.customer_keywords = customer_keywords

    await db.commit()
    await db.refresh(row)
    return AIHubRecommendationFeedbackResponse.model_validate(row)
