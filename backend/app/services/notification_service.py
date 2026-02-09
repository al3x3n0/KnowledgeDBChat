"""
Notification service for creating and managing user notifications.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, and_, or_
from loguru import logger
import json
import redis

from app.models.notification import Notification, NotificationPreferences, NotificationType
from app.models.user import User
from app.core.config import settings


class NotificationService:
    """Service for managing notifications."""

    # Mapping of notification types to preference field names
    TYPE_TO_PREFERENCE = {
        NotificationType.DOCUMENT_PROCESSING_COMPLETE: "notify_document_processing",
        NotificationType.DOCUMENT_PROCESSING_ERROR: "notify_document_errors",
        NotificationType.DOCUMENT_SYNC_COMPLETE: "notify_sync_complete",
        NotificationType.DOCUMENT_SYNC_ERROR: "notify_document_errors",
        NotificationType.DOCUMENT_INGESTION_COMPLETE: "notify_ingestion_complete",
        NotificationType.DOCUMENT_INGESTION_ERROR: "notify_document_errors",
        NotificationType.DOCUMENT_TRANSCRIPTION_COMPLETE: "notify_transcription_complete",
        NotificationType.DOCUMENT_TRANSCRIPTION_ERROR: "notify_document_errors",
        NotificationType.DOCUMENT_SUMMARIZATION_COMPLETE: "notify_summarization_complete",
        NotificationType.SYSTEM_MAINTENANCE: "notify_maintenance",
        NotificationType.SYSTEM_QUOTA_WARNING: "notify_quota_warnings",
        NotificationType.ADMIN_BROADCAST: "notify_admin_broadcasts",
        NotificationType.RESEARCH_NOTE_CITATION_ISSUE: "notify_research_note_citation_issues",
        NotificationType.EXPERIMENT_RUN_UPDATE: "notify_experiment_run_updates",
        NotificationType.COLLABORATION_MENTION: "notify_mentions",
        NotificationType.COLLABORATION_SHARE: "notify_shares",
        NotificationType.COLLABORATION_COMMENT: "notify_comments",
    }

    def _get_redis_client(self):
        """Get a Redis client for publishing notifications."""
        try:
            return redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to create Redis client: {e}")
            return None

    async def create_notification(
        self,
        db: AsyncSession,
        user_id: UUID,
        notification_type: str,
        title: str,
        message: str,
        priority: str = "normal",
        related_entity_type: Optional[str] = None,
        related_entity_id: Optional[UUID] = None,
        data: Optional[Dict[str, Any]] = None,
        action_url: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Optional[Notification]:
        """Create a notification for a user, respecting their preferences."""
        try:
            # Check user preferences
            prefs = await self.get_user_preferences(db, user_id)

            # Check if user has opted out of this notification type
            pref_field = self.TYPE_TO_PREFERENCE.get(notification_type)
            if pref_field and prefs and not getattr(prefs, pref_field, True):
                logger.debug(f"User {user_id} has disabled {notification_type} notifications")
                return None

            # Create notification
            notification = Notification(
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                message=message,
                priority=priority,
                related_entity_type=related_entity_type,
                related_entity_id=related_entity_id,
                data=data,
                action_url=action_url,
                expires_at=expires_at,
                created_at=datetime.utcnow(),
            )

            db.add(notification)
            await db.commit()
            await db.refresh(notification)

            # Push via WebSocket
            self._push_notification(user_id, notification)

            logger.info(f"Created notification {notification.id} for user {user_id}: {title}")
            return notification

        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            await db.rollback()
            return None

    async def create_broadcast_notification(
        self,
        db: AsyncSession,
        notification_type: str,
        title: str,
        message: str,
        priority: str = "normal",
        data: Optional[Dict[str, Any]] = None,
        action_url: Optional[str] = None,
        target_roles: Optional[List[str]] = None,
    ) -> int:
        """Create a notification for all users (or filtered by role)."""
        try:
            # Get target users
            query = select(User.id).where(User.is_active == True)
            if target_roles:
                query = query.where(User.role.in_(target_roles))

            result = await db.execute(query)
            user_ids = [row[0] for row in result.fetchall()]

            count = 0
            for user_id in user_ids:
                notification = await self.create_notification(
                    db=db,
                    user_id=user_id,
                    notification_type=notification_type,
                    title=title,
                    message=message,
                    priority=priority,
                    data=data,
                    action_url=action_url,
                )
                if notification:
                    count += 1

            logger.info(f"Created {count} broadcast notifications: {title}")
            return count

        except Exception as e:
            logger.error(f"Error creating broadcast notifications: {e}")
            return 0

    async def get_notifications(
        self,
        db: AsyncSession,
        user_id: UUID,
        page: int = 1,
        page_size: int = 20,
        unread_only: bool = False,
        notification_types: Optional[List[str]] = None,
    ) -> Tuple[List[Notification], int]:
        """Get paginated notifications for a user."""
        try:
            # Build query
            query = select(Notification).where(
                Notification.user_id == user_id,
                Notification.is_dismissed == False,
            )

            # Filter expired notifications
            query = query.where(
                or_(
                    Notification.expires_at.is_(None),
                    Notification.expires_at > datetime.utcnow()
                )
            )

            if unread_only:
                query = query.where(Notification.is_read == False)

            if notification_types:
                query = query.where(Notification.notification_type.in_(notification_types))

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total = total_result.scalar() or 0

            # Apply pagination and ordering
            skip = (page - 1) * page_size
            query = query.order_by(Notification.created_at.desc()).offset(skip).limit(page_size)

            result = await db.execute(query)
            notifications = list(result.scalars().all())

            return notifications, total

        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return [], 0

    async def get_unread_count(self, db: AsyncSession, user_id: UUID) -> int:
        """Get count of unread notifications for a user."""
        try:
            query = select(func.count()).where(
                Notification.user_id == user_id,
                Notification.is_read == False,
                Notification.is_dismissed == False,
                or_(
                    Notification.expires_at.is_(None),
                    Notification.expires_at > datetime.utcnow()
                )
            )
            result = await db.execute(query)
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error getting unread count: {e}")
            return 0

    async def mark_as_read(
        self,
        db: AsyncSession,
        user_id: UUID,
        notification_id: UUID,
    ) -> bool:
        """Mark a notification as read."""
        try:
            result = await db.execute(
                update(Notification)
                .where(
                    Notification.id == notification_id,
                    Notification.user_id == user_id,
                )
                .values(is_read=True, read_at=datetime.utcnow())
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            await db.rollback()
            return False

    async def mark_all_as_read(self, db: AsyncSession, user_id: UUID) -> int:
        """Mark all notifications as read for a user."""
        try:
            result = await db.execute(
                update(Notification)
                .where(
                    Notification.user_id == user_id,
                    Notification.is_read == False,
                )
                .values(is_read=True, read_at=datetime.utcnow())
            )
            await db.commit()
            return result.rowcount
        except Exception as e:
            logger.error(f"Error marking all notifications as read: {e}")
            await db.rollback()
            return 0

    async def dismiss_notification(
        self,
        db: AsyncSession,
        user_id: UUID,
        notification_id: UUID,
    ) -> bool:
        """Dismiss (soft delete) a notification."""
        try:
            result = await db.execute(
                update(Notification)
                .where(
                    Notification.id == notification_id,
                    Notification.user_id == user_id,
                )
                .values(is_dismissed=True)
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error dismissing notification: {e}")
            await db.rollback()
            return False

    async def delete_notification(
        self,
        db: AsyncSession,
        user_id: UUID,
        notification_id: UUID,
    ) -> bool:
        """Permanently delete a notification."""
        try:
            result = await db.execute(
                select(Notification).where(
                    Notification.id == notification_id,
                    Notification.user_id == user_id,
                )
            )
            notification = result.scalar_one_or_none()
            if notification:
                await db.delete(notification)
                await db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting notification: {e}")
            await db.rollback()
            return False

    async def get_user_preferences(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> Optional[NotificationPreferences]:
        """Get notification preferences for a user."""
        try:
            result = await db.execute(
                select(NotificationPreferences).where(
                    NotificationPreferences.user_id == user_id
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting notification preferences: {e}")
            return None

    async def update_user_preferences(
        self,
        db: AsyncSession,
        user_id: UUID,
        preferences: Dict[str, Any],
    ) -> Optional[NotificationPreferences]:
        """Update notification preferences for a user."""
        try:
            prefs = await self.get_user_preferences(db, user_id)

            if not prefs:
                # Create default preferences
                prefs = NotificationPreferences(user_id=user_id)
                db.add(prefs)

            # Update fields
            for field, value in preferences.items():
                if hasattr(prefs, field):
                    setattr(prefs, field, value)

            prefs.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(prefs)
            return prefs

        except Exception as e:
            logger.error(f"Error updating notification preferences: {e}")
            await db.rollback()
            return None

    def _push_notification(self, user_id: UUID, notification: Notification) -> None:
        """Push notification to user via Redis pub/sub."""
        try:
            client = self._get_redis_client()
            if not client:
                return

            # Publish to user-specific channel
            channel = f"notifications:{user_id}"
            message = {
                "type": "notification",
                "notification": {
                    "id": str(notification.id),
                    "notification_type": notification.notification_type,
                    "title": notification.title,
                    "message": notification.message,
                    "priority": notification.priority,
                    "related_entity_type": notification.related_entity_type,
                    "related_entity_id": str(notification.related_entity_id) if notification.related_entity_id else None,
                    "data": notification.data,
                    "action_url": notification.action_url,
                    "is_read": notification.is_read,
                    "created_at": notification.created_at.isoformat() if notification.created_at else None,
                }
            }
            client.publish(channel, json.dumps(message))
            logger.debug(f"Pushed notification to channel {channel}")

        except Exception as e:
            logger.warning(f"Failed to push notification via Redis: {e}")

    async def cleanup_expired_notifications(self, db: AsyncSession) -> int:
        """Remove expired notifications (for maintenance task)."""
        try:
            from sqlalchemy import delete
            result = await db.execute(
                delete(Notification).where(
                    Notification.expires_at < datetime.utcnow()
                )
            )
            await db.commit()
            return result.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up expired notifications: {e}")
            await db.rollback()
            return 0


# Singleton instance
notification_service = NotificationService()
