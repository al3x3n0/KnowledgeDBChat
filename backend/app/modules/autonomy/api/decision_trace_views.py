"""Saved-view CRUD boundary for autonomous decision traces."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.autonomy_decision_trace_view import AutonomyDecisionTraceView
from app.models.user import User
from app.schemas.agent_job import (
    AgentDecisionTraceViewCreate,
    AgentDecisionTraceViewListResponse,
    AgentDecisionTraceViewResponse,
    AgentDecisionTraceViewUpdate,
)

TraceViewFilterNormalizer = Callable[[dict[str, Any] | None], dict[str, Any]]


@dataclass(frozen=True)
class DecisionTraceViewApi:
    router: APIRouter
    list_decision_trace_views: Callable[..., Any]
    create_decision_trace_view: Callable[..., Any]
    update_decision_trace_view: Callable[..., Any]
    delete_decision_trace_view: Callable[..., Any]


def build_decision_trace_view_api(
    *,
    router: APIRouter,
    normalize_filters: TraceViewFilterNormalizer,
) -> DecisionTraceViewApi:
    """Register ownership-scoped saved-view routes."""

    async def load_owned_view(
        *,
        view_id: UUID,
        db: AsyncSession,
        current_user: User,
    ) -> AutonomyDecisionTraceView:
        row = (
            (
                await db.execute(
                    select(AutonomyDecisionTraceView).where(
                        AutonomyDecisionTraceView.id == view_id,
                        AutonomyDecisionTraceView.user_id == current_user.id,
                    )
                )
            )
            .scalars()
            .first()
        )
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Decision trace view not found",
            )
        return row

    @router.get(
        "/decision-trace/views",
        response_model=AgentDecisionTraceViewListResponse,
    )
    async def list_decision_trace_views(
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        rows = list(
            (
                await db.execute(
                    select(AutonomyDecisionTraceView)
                    .where(AutonomyDecisionTraceView.user_id == current_user.id)
                    .order_by(
                        AutonomyDecisionTraceView.is_default.desc(),
                        AutonomyDecisionTraceView.updated_at.desc(),
                    )
                )
            )
            .scalars()
            .all()
        )
        return AgentDecisionTraceViewListResponse(
            items=[AgentDecisionTraceViewResponse.model_validate(row) for row in rows],
            total=len(rows),
        )

    @router.post(
        "/decision-trace/views",
        response_model=AgentDecisionTraceViewResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_decision_trace_view(
        request: AgentDecisionTraceViewCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        name = str(request.name or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Decision trace view name is required",
            )
        if request.is_default:
            await db.execute(
                AutonomyDecisionTraceView.__table__.update()
                .where(AutonomyDecisionTraceView.user_id == current_user.id)
                .values(is_default=False)
            )
        row = AutonomyDecisionTraceView(
            user_id=current_user.id,
            name=name,
            filters=normalize_filters(request.filters),
            is_default=bool(request.is_default),
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return AgentDecisionTraceViewResponse.model_validate(row)

    @router.patch(
        "/decision-trace/views/{view_id}",
        response_model=AgentDecisionTraceViewResponse,
    )
    async def update_decision_trace_view(
        view_id: UUID,
        request: AgentDecisionTraceViewUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        row = await load_owned_view(
            view_id=view_id,
            db=db,
            current_user=current_user,
        )
        if request.name is not None:
            next_name = str(request.name or "").strip()
            if not next_name:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Decision trace view name is required",
                )
            row.name = next_name
        if request.filters is not None:
            row.filters = normalize_filters(request.filters)
        if request.is_default is not None:
            if bool(request.is_default):
                await db.execute(
                    AutonomyDecisionTraceView.__table__.update()
                    .where(
                        AutonomyDecisionTraceView.user_id == current_user.id,
                        AutonomyDecisionTraceView.id != row.id,
                    )
                    .values(is_default=False)
                )
            row.is_default = bool(request.is_default)
        row.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(row)
        return AgentDecisionTraceViewResponse.model_validate(row)

    @router.delete(
        "/decision-trace/views/{view_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_decision_trace_view(
        view_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        row = await load_owned_view(
            view_id=view_id,
            db=db,
            current_user=current_user,
        )
        await db.delete(row)
        await db.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return DecisionTraceViewApi(
        router=router,
        list_decision_trace_views=list_decision_trace_views,
        create_decision_trace_view=create_decision_trace_view,
        update_decision_trace_view=update_decision_trace_view,
        delete_decision_trace_view=delete_decision_trace_view,
    )
