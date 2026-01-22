"""
Notification-related database models.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Notification(Base):
    """Stores in-app notifications for users."""

    __tablename__ = "notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Notification content
    notification_type = Column(String(50), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(String(20), default="normal", nullable=False)  # low, normal, high, urgent

    # Related entities (polymorphic references)
    related_entity_type = Column(String(50), nullable=True)  # "document", "source", "user", etc.
    related_entity_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Additional data for the notification (action URLs, metadata, etc.)
    data = Column(JSONB, nullable=True)

    # Action URL for click navigation
    action_url = Column(String(500), nullable=True)

    # Read/dismiss state
    is_read = Column(Boolean, default=False, nullable=False, index=True)
    read_at = Column(DateTime(timezone=True), nullable=True)
    is_dismissed = Column(Boolean, default=False, nullable=False)

    # Expiration (for time-sensitive notifications)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", backref="notifications")

    # Composite index for efficient queries
    __table_args__ = (
        Index('ix_notifications_user_unread', 'user_id', 'is_read', 'is_dismissed', 'created_at'),
    )

    def __repr__(self):
        return f"<Notification(id={self.id}, type={self.notification_type}, user_id={self.user_id})>"


class NotificationPreferences(Base):
    """Per-user notification preferences."""

    __tablename__ = "notification_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Document event preferences
    notify_document_processing = Column(Boolean, default=True, nullable=False)
    notify_document_errors = Column(Boolean, default=True, nullable=False)
    notify_sync_complete = Column(Boolean, default=True, nullable=False)
    notify_ingestion_complete = Column(Boolean, default=True, nullable=False)
    notify_transcription_complete = Column(Boolean, default=True, nullable=False)
    notify_summarization_complete = Column(Boolean, default=False, nullable=False)

    # System event preferences
    notify_maintenance = Column(Boolean, default=True, nullable=False)
    notify_quota_warnings = Column(Boolean, default=True, nullable=False)
    notify_admin_broadcasts = Column(Boolean, default=True, nullable=False)

    # Collaboration event preferences (future)
    notify_mentions = Column(Boolean, default=True, nullable=False)
    notify_shares = Column(Boolean, default=True, nullable=False)
    notify_comments = Column(Boolean, default=True, nullable=False)

    # Delivery preferences
    play_sound = Column(Boolean, default=False, nullable=False)
    show_desktop_notification = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", backref="notification_preferences")

    def __repr__(self):
        return f"<NotificationPreferences(user_id={self.user_id})>"


# Notification type constants for reference
class NotificationType:
    """Notification type constants."""
    # Document events
    DOCUMENT_PROCESSING_COMPLETE = "document_processing_complete"
    DOCUMENT_PROCESSING_ERROR = "document_processing_error"
    DOCUMENT_SYNC_COMPLETE = "source_sync_complete"
    DOCUMENT_SYNC_ERROR = "source_sync_error"
    DOCUMENT_INGESTION_COMPLETE = "ingestion_complete"
    DOCUMENT_INGESTION_ERROR = "ingestion_error"
    DOCUMENT_TRANSCRIPTION_COMPLETE = "transcription_complete"
    DOCUMENT_TRANSCRIPTION_ERROR = "transcription_error"
    DOCUMENT_SUMMARIZATION_COMPLETE = "summarization_complete"

    # System events
    SYSTEM_MAINTENANCE = "system_maintenance"
    SYSTEM_QUOTA_WARNING = "quota_warning"
    ADMIN_BROADCAST = "admin_broadcast"

    # Collaboration events (future-proofing)
    COLLABORATION_MENTION = "mention"
    COLLABORATION_SHARE = "share"
    COLLABORATION_COMMENT = "comment"
