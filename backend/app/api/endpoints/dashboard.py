"""
Dashboard API endpoints for frontend widgets and visualizations.

Provides optimized data for:
- Overview statistics cards
- Activity feeds
- Charts (pie, bar, line)
- Health monitoring
- Quick actions
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import json
import asyncio

from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.dashboard_service import dashboard_service

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
    activity = await dashboard_service.get_activity_feed(db, user_id=current_user.id, limit=10)
    docs_by_type = await dashboard_service.get_documents_by_type_chart(db)
    docs_by_source = await dashboard_service.get_documents_by_source_chart(db)
    timeline = await dashboard_service.get_documents_timeline_chart(db, days=30)
    sources_health = await dashboard_service.get_source_health(db)
    trending = await dashboard_service.get_trending_tags(db, days=7, limit=10)
    quick_actions = await dashboard_service.get_quick_actions(db, user_id=current_user.id)
    agent_summary = await dashboard_service.get_agent_usage_summary(db, user_id=current_user.id, days=7)
    workflow_summary = await dashboard_service.get_workflow_summary(db, user_id=current_user.id, days=7)

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
@router.websocket("/ws")
async def dashboard_websocket(
    websocket: WebSocket,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket endpoint for real-time dashboard updates.

    Sends periodic updates with latest stats.
    """
    await websocket.accept()
    logger.info("Dashboard WebSocket connected")

    try:
        # Try to authenticate from query params or first message
        user_id = None

        # Wait for auth message
        try:
            auth_message = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=10.0
            )
            if auth_message.get("type") == "auth":
                # In production, validate the token
                user_id = auth_message.get("user_id")
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "error", "message": "Auth timeout"})
            await websocket.close()
            return

        # Send initial data
        overview = await dashboard_service.get_overview_stats(db, user_id=user_id)
        await websocket.send_json({
            "type": "overview",
            "data": overview
        })

        # Periodic updates
        update_interval = 30  # seconds
        while True:
            try:
                # Check for incoming messages (non-blocking)
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=update_interval
                    )

                    # Handle refresh request
                    if message.get("type") == "refresh":
                        widget = message.get("widget", "all")
                        if widget == "overview" or widget == "all":
                            data = await dashboard_service.get_overview_stats(db, user_id=user_id)
                            await websocket.send_json({"type": "overview", "data": data})
                        if widget == "activity" or widget == "all":
                            data = await dashboard_service.get_activity_feed(db, user_id=user_id, limit=10)
                            await websocket.send_json({"type": "activity", "data": data})
                        if widget == "sources" or widget == "all":
                            data = await dashboard_service.get_source_health(db)
                            await websocket.send_json({"type": "sources_health", "data": data})

                except asyncio.TimeoutError:
                    # No message received, send periodic update
                    overview = await dashboard_service.get_overview_stats(db, user_id=user_id)
                    await websocket.send_json({
                        "type": "overview",
                        "data": overview,
                        "is_periodic": True
                    })

            except WebSocketDisconnect:
                logger.info("Dashboard WebSocket disconnected")
                break

    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
