"""
Notification-related Pydantic schemas.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class NotificationBase(BaseModel):
    """Base notification schema."""
    id: UUID
    notification_type: str
    title: str
    message: str
    priority: str = "normal"
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[UUID] = None
    data: Optional[Dict[str, Any]] = None
    action_url: Optional[str] = None
    is_read: bool = False
    read_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class NotificationResponse(NotificationBase):
    """Notification response schema."""
    pass


class NotificationListResponse(BaseModel):
    """Paginated notification list response."""
    items: List[NotificationResponse]
    total: int
    page: int
    page_size: int
    unread_count: int


class NotificationCreate(BaseModel):
    """Schema for creating a notification (internal use)."""
    user_id: UUID
    notification_type: str
    title: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    related_entity_type: Optional[str] = None
    related_entity_id: Optional[UUID] = None
    data: Optional[Dict[str, Any]] = None
    action_url: Optional[str] = None
    expires_at: Optional[datetime] = None


class NotificationPreferencesBase(BaseModel):
    """Base notification preferences schema."""
    # Document event preferences
    notify_document_processing: bool = True
    notify_document_errors: bool = True
    notify_sync_complete: bool = True
    notify_ingestion_complete: bool = True
    notify_transcription_complete: bool = True
    notify_summarization_complete: bool = False

    # System event preferences
    notify_maintenance: bool = True
    notify_quota_warnings: bool = True
    notify_admin_broadcasts: bool = True

    # Collaboration event preferences (future)
    notify_mentions: bool = True
    notify_shares: bool = True
    notify_comments: bool = True

    # Delivery preferences
    play_sound: bool = False
    show_desktop_notification: bool = False


class NotificationPreferencesUpdate(BaseModel):
    """Schema for updating notification preferences."""
    notify_document_processing: Optional[bool] = None
    notify_document_errors: Optional[bool] = None
    notify_sync_complete: Optional[bool] = None
    notify_ingestion_complete: Optional[bool] = None
    notify_transcription_complete: Optional[bool] = None
    notify_summarization_complete: Optional[bool] = None
    notify_maintenance: Optional[bool] = None
    notify_quota_warnings: Optional[bool] = None
    notify_admin_broadcasts: Optional[bool] = None
    notify_mentions: Optional[bool] = None
    notify_shares: Optional[bool] = None
    notify_comments: Optional[bool] = None
    play_sound: Optional[bool] = None
    show_desktop_notification: Optional[bool] = None


class NotificationPreferencesResponse(NotificationPreferencesBase):
    """Notification preferences response schema."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BroadcastNotificationRequest(BaseModel):
    """Schema for admin broadcast notification."""
    title: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=2000)
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    action_url: Optional[str] = None
    target_roles: Optional[List[str]] = None  # If None, broadcast to all users


class UnreadCountResponse(BaseModel):
    """Response for unread count endpoint."""
    unread_count: int
