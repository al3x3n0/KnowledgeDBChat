"""One answer to "what is waiting on me". See modules/review."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.modules.review.application.review_inbox import build_review_inbox
from app.schemas.review import ReviewInboxResponse, ReviewSectionResponse
from app.services.auth_service import get_current_user

router = APIRouter()


@router.get("", response_model=ReviewInboxResponse)
async def get_review_inbox(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Every decision this person owes, grouped by where it is made."""
    sections = await build_review_inbox(db, current_user)
    return ReviewInboxResponse(
        total=sum(section.count for section in sections),
        sections=[ReviewSectionResponse.model_validate(s) for s in sections],
    )
