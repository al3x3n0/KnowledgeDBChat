"""
API endpoints for LangGraph workflows.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.langgraph_issue_pr import (
    LangGraphIssuePrRequest,
    LangGraphIssuePrResponse,
)
from app.services.langgraph_issue_pr_service import LangGraphIssuePrService

router = APIRouter()


@router.post(
    "/issue-pr/draft",
    response_model=LangGraphIssuePrResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_issue_pr_draft(
    request: LangGraphIssuePrRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    service = LangGraphIssuePrService()
    try:
        return await service.run(request, user_id=current_user.id, db=db)
    except RuntimeError as exc:
        logger.warning(
            f"LangGraph workflow unavailable for user={current_user.id}: {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    except Exception as exc:
        logger.exception(
            f"LangGraph issue-pr orchestration failed for user={current_user.id}: {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LangGraph issue-pr orchestration failed.",
        )
