"""Add notifications tables.

Revision ID: 0021_add_notifications
Revises: 0020_add_specialized_agents
Create Date: 2026-01-21 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0021_add_notifications"
down_revision = "0020_add_specialized_agents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create notifications and notification_preferences tables."""

    # Create notifications table
    op.create_table(
        "notifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("notification_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("priority", sa.String(20), server_default="normal", nullable=False),
        sa.Column("related_entity_type", sa.String(50), nullable=True),
        sa.Column("related_entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("data", postgresql.JSONB(), nullable=True),
        sa.Column("action_url", sa.String(500), nullable=True),
        sa.Column("is_read", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_dismissed", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # Create indexes for notifications
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])
    op.create_index("ix_notifications_notification_type", "notifications", ["notification_type"])
    op.create_index("ix_notifications_related_entity_id", "notifications", ["related_entity_id"])
    op.create_index("ix_notifications_is_read", "notifications", ["is_read"])
    op.create_index("ix_notifications_created_at", "notifications", ["created_at"])

    # Composite index for efficient queries
    op.create_index(
        "ix_notifications_user_unread",
        "notifications",
        ["user_id", "is_read", "is_dismissed", "created_at"],
    )

    # Create notification_preferences table
    op.create_table(
        "notification_preferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True),
        # Document event preferences
        sa.Column("notify_document_processing", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_document_errors", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_sync_complete", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_ingestion_complete", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_transcription_complete", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_summarization_complete", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        # System event preferences
        sa.Column("notify_maintenance", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_quota_warnings", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_admin_broadcasts", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        # Collaboration event preferences (future)
        sa.Column("notify_mentions", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_shares", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("notify_comments", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        # Delivery preferences
        sa.Column("play_sound", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("show_desktop_notification", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # Create index for notification_preferences
    op.create_index("ix_notification_preferences_user_id", "notification_preferences", ["user_id"])


def downgrade() -> None:
    """Drop notifications tables."""
    op.drop_index("ix_notification_preferences_user_id", table_name="notification_preferences")
    op.drop_table("notification_preferences")

    op.drop_index("ix_notifications_user_unread", table_name="notifications")
    op.drop_index("ix_notifications_created_at", table_name="notifications")
    op.drop_index("ix_notifications_is_read", table_name="notifications")
    op.drop_index("ix_notifications_related_entity_id", table_name="notifications")
    op.drop_index("ix_notifications_notification_type", table_name="notifications")
    op.drop_index("ix_notifications_user_id", table_name="notifications")
    op.drop_table("notifications")
