"""
Dashboard API endpoints for frontend widgets and visualizations.

Provides optimized data for:
- Overview statistics cards
- Activity feeds
- Charts (pie, bar, line)
- Health monitoring
- Quick actions
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.dashboard_service import dashboard_service
from app.utils.websocket_auth import authenticate_websocket

router = APIRouter()


@router.get("/overview")
async def get_dashboard_overview(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get overview statistics for dashboard cards.

    Returns document counts, source stats, workflow stats, and agent stats.
    """
    return await dashboard_service.get_overview_stats(db, user_id=current_user.id)


@router.get("/activity")
async def get_activity_feed(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent activity feed.

    Includes document uploads, workflow executions, and other events.
    """
    activities = await dashboard_service.get_activity_feed(
        db, user_id=current_user.id, limit=limit
    )
    return {"activities": activities, "count": len(activities)}


@router.get("/charts/documents-by-type")
async def get_documents_by_type_chart(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get document distribution by file type for pie/doughnut chart."""
    return await dashboard_service.get_documents_by_type_chart(db)


@router.get("/charts/documents-by-source")
async def get_documents_by_source_chart(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get document distribution by source for bar chart."""
    return await dashboard_service.get_documents_by_source_chart(db)


@router.get("/charts/documents-timeline")
async def get_documents_timeline_chart(
    days: int = Query(30, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get document creation timeline for line chart."""
    return await dashboard_service.get_documents_timeline_chart(db, days=days)


@router.get("/sources/health")
async def get_sources_health(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get health status for all document sources.

    Returns status (healthy, warning, stale, inactive) for each source.
    """
    sources = await dashboard_service.get_source_health(db)
    return {"sources": sources, "count": len(sources)}


@router.get("/trending/tags")
async def get_trending_tags(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get trending tags from recent documents."""
    tags = await dashboard_service.get_trending_tags(db, days=days, limit=limit)
    return {"tags": tags, "period_days": days}


@router.get("/quick-actions")
async def get_quick_actions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get suggested quick actions based on current system state.

    Actions are prioritized and include links, modals, or workflow triggers.
    """
    actions = await dashboard_service.get_quick_actions(db, user_id=current_user.id)
    return {"actions": actions}


@router.get("/agents/summary")
async def get_agent_usage_summary(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get agent usage summary for the dashboard."""
    return await dashboard_service.get_agent_usage_summary(
        db, user_id=current_user.id, days=days
    )


@router.get("/workflows/summary")
async def get_workflow_summary(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get workflow execution summary."""
    return await dashboard_service.get_workflow_summary(
        db, user_id=current_user.id, days=days
    )


@router.get("/all")
async def get_full_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all dashboard data in a single request.

    Useful for initial page load to minimize API calls.
    """
    overview = await dashboard_service.get_overview_stats(db, user_id=current_user.id)
    activity = await dashboard_service.get_activity_feed(
        db, user_id=current_user.id, limit=10
    )
    docs_by_type = await dashboard_service.get_documents_by_type_chart(db)
    docs_by_source = await dashboard_service.get_documents_by_source_chart(db)
    timeline = await dashboard_service.get_documents_timeline_chart(db, days=30)
    sources_health = await dashboard_service.get_source_health(db)
    trending = await dashboard_service.get_trending_tags(db, days=7, limit=10)
    quick_actions = await dashboard_service.get_quick_actions(
        db, user_id=current_user.id
    )
    agent_summary = await dashboard_service.get_agent_usage_summary(
        db, user_id=current_user.id, days=7
    )
    workflow_summary = await dashboard_service.get_workflow_summary(
        db, user_id=current_user.id, days=7
    )

    return {
        "overview": overview,
        "activity": activity,
        "charts": {
            "documents_by_type": docs_by_type,
            "documents_by_source": docs_by_source,
            "documents_timeline": timeline,
        },
        "sources_health": sources_health,
        "trending_tags": trending,
        "quick_actions": quick_actions,
        "agent_summary": agent_summary,
        "workflow_summary": workflow_summary,
    }


# WebSocket for real-time dashboard updates
AUTH_WAIT_SECONDS = 10.0
UPDATE_INTERVAL_SECONDS = 30


async def _socket_user(websocket: WebSocket) -> Optional[User]:
    """Who connected: a token in the query string or header, or in a first
    `{"type": "auth", "token": ...}` message. Never an id the client states."""
    user = await authenticate_websocket(websocket)
    if user is not None:
        return user
    try:
        message = await asyncio.wait_for(
            websocket.receive_json(), timeout=AUTH_WAIT_SECONDS
        )
    except asyncio.TimeoutError:
        return None
    if not isinstance(message, dict) or message.get("type") != "auth":
        return None
    token = message.get("token")
    if not isinstance(token, str) or not token:
        return None
    return await authenticate_websocket(websocket, token=token)


async def _send_widget(websocket: WebSocket, user: User, widget: str, **extra) -> None:
    """Send one widget's data, read on a session that lives for this read.

    The handler used to take the request's session as a dependency and keep
    it -- one of a limited number -- for as long as the socket stayed open.
    """
    async with AsyncSessionLocal() as db:
        if widget == "overview":
            data = await dashboard_service.get_overview_stats(db, user_id=user.id)
            kind = "overview"
        elif widget == "activity":
            data = await dashboard_service.get_activity_feed(
                db, user_id=user.id, limit=10
            )
            kind = "activity"
        else:
            data = await dashboard_service.get_source_health(db)
            kind = "sources_health"
    await websocket.send_json({"type": kind, "data": data, **extra})


@router.websocket("/ws")
async def dashboard_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.

    Sends periodic updates with latest stats.
    """
    await websocket.accept()

    try:
        # It took `user_id` from the client's first message and checked no
        # token: anyone who could reach the server was sent the overview,
        # document titles and source health.
        user = await _socket_user(websocket)
        if user is None:
            await websocket.close(code=4001, reason="Authentication required")
            return
        logger.info("Dashboard WebSocket connected")

        await _send_widget(websocket, user, "overview")

        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(), timeout=UPDATE_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                # No message received, send periodic update
                await _send_widget(websocket, user, "overview", is_periodic=True)
                continue

            if isinstance(message, dict) and message.get("type") == "refresh":
                widget = message.get("widget", "all")
                for name in ("overview", "activity", "sources"):
                    if widget in (name, "all"):
                        await _send_widget(websocket, user, name)

    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket disconnected")
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        try:
            await websocket.send_json(
                {"type": "error", "message": "Dashboard update failed"}
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
