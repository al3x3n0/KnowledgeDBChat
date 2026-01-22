"""
Notification API endpoints.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import asyncio
import json

from app.core.database import get_db
from app.core.config import settings
from app.models.user import User
from app.services.auth_service import get_current_user, require_admin
from app.services.notification_service import notification_service
from app.schemas.notification import (
    NotificationResponse,
    NotificationListResponse,
    NotificationPreferencesResponse,
    NotificationPreferencesUpdate,
    BroadcastNotificationRequest,
    UnreadCountResponse,
)
from app.models.notification import NotificationType
from app.utils.websocket_auth import require_websocket_auth

router = APIRouter()


@router.get("/", response_model=NotificationListResponse)
async def get_notifications(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    unread_only: bool = Query(False),
    notification_types: Optional[str] = Query(None, description="Comma-separated notification types"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated notifications for the current user."""
    types_list = notification_types.split(",") if notification_types else None

    notifications, total = await notification_service.get_notifications(
        db=db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        unread_only=unread_only,
        notification_types=types_list,
    )

    unread_count = await notification_service.get_unread_count(db, current_user.id)

    return NotificationListResponse(
        items=[NotificationResponse.model_validate(n) for n in notifications],
        total=total,
        page=page,
        page_size=page_size,
        unread_count=unread_count,
    )


@router.get("/unread-count", response_model=UnreadCountResponse)
async def get_unread_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get count of unread notifications."""
    count = await notification_service.get_unread_count(db, current_user.id)
    return UnreadCountResponse(unread_count=count)


@router.put("/{notification_id}/read")
async def mark_notification_read(
    notification_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark a single notification as read."""
    success = await notification_service.mark_as_read(db, current_user.id, notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification marked as read"}


@router.put("/read-all")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark all notifications as read."""
    count = await notification_service.mark_all_as_read(db, current_user.id)
    return {"message": f"Marked {count} notifications as read", "count": count}


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a notification."""
    success = await notification_service.delete_notification(db, current_user.id, notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification deleted"}


@router.put("/{notification_id}/dismiss")
async def dismiss_notification(
    notification_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Dismiss a notification (soft delete)."""
    success = await notification_service.dismiss_notification(db, current_user.id, notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"message": "Notification dismissed"}


# Preferences endpoints
@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get notification preferences for the current user."""
    prefs = await notification_service.get_user_preferences(db, current_user.id)

    if not prefs:
        # Return defaults by creating preferences
        prefs = await notification_service.update_user_preferences(db, current_user.id, {})

    return prefs


@router.put("/preferences", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    updates: NotificationPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update notification preferences for the current user."""
    update_data = updates.model_dump(exclude_unset=True)
    prefs = await notification_service.update_user_preferences(db, current_user.id, update_data)

    if not prefs:
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    return prefs


# Admin broadcast endpoint
@router.post("/broadcast")
async def broadcast_notification(
    request: BroadcastNotificationRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Broadcast a notification to all users (admin only)."""
    count = await notification_service.create_broadcast_notification(
        db=db,
        notification_type=NotificationType.ADMIN_BROADCAST,
        title=request.title,
        message=request.message,
        priority=request.priority,
        data={"broadcast_by": current_user.username},
        action_url=request.action_url,
        target_roles=request.target_roles,
    )

    return {"message": f"Broadcast sent to {count} users", "count": count}


# WebSocket endpoint for real-time notifications
@router.websocket("/ws")
async def notifications_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time notification delivery.

    Subscribes to the user's notification channel via Redis pub/sub.
    Messages are pushed in real-time when notifications are created.
    """
    try:
        user = await require_websocket_auth(websocket)
        logger.info(f"Notifications WebSocket connected for user {user.id}")
    except WebSocketDisconnect:
        logger.warning("Notifications WebSocket authentication failed")
        return

    # Subscribe to user's notification channel via Redis
    import redis.asyncio as aioredis

    try:
        redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        channel = f"notifications:{user.id}"
        await pubsub.subscribe(channel)

        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to notification stream",
            "channel": channel,
        })

        # Listen for messages
        async def reader():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        await websocket.send_text(message["data"])
                    except Exception as e:
                        logger.warning(f"Failed to send WebSocket message: {e}")
                        break

        # Handle incoming messages (ping/pong)
        async def writer():
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        msg = json.loads(data)
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass

        # Run both tasks concurrently
        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())

        try:
            await asyncio.gather(reader_task, writer_task)
        except Exception as e:
            logger.debug(f"WebSocket tasks ended: {e}")
        finally:
            reader_task.cancel()
            writer_task.cancel()

    except WebSocketDisconnect:
        logger.info(f"Notifications WebSocket disconnected for user {user.id}")
    except Exception as e:
        logger.error(f"Error in notifications WebSocket: {e}")
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            await redis_client.close()
        except Exception:
            pass
        logger.info(f"Notifications WebSocket closed for user {user.id}")
