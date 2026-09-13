"""Bounded polling subscriptions for CompOps-backed R&D evidence."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.sql import func

from app.core.database import Base


class CompOpsEvidenceSubscription(Base):
    __tablename__ = "compops_evidence_subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_tools.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    capability = Column(String(120), nullable=False)
    remote_id = Column(String(200), nullable=False)
    payload = Column(JSON, nullable=False, default=dict)
    interval_minutes = Column(Integer, nullable=False, default=15)
    is_enabled = Column(Boolean, nullable=False, default=True)
    status = Column(String(32), nullable=False, default="active")
    last_response_sha256 = Column(String(64), nullable=True)
    last_audit_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tool_execution_audits.id", ondelete="SET NULL"),
        nullable=True,
    )
    last_attempt_at = Column(DateTime(timezone=True), nullable=True)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    next_sync_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_error = Column(Text, nullable=True)
    webhook_secret_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_secrets.id", ondelete="SET NULL"),
        nullable=True,
    )
    webhook_enabled = Column(Boolean, nullable=False, default=False)
    last_webhook_at = Column(DateTime(timezone=True), nullable=True)
    last_webhook_event_id = Column(String(200), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "tool_id",
            "capability",
            "remote_id",
            name="uq_compops_evidence_subscription_target",
        ),
        Index(
            "ix_compops_evidence_subscriptions_due",
            "is_enabled",
            "next_sync_at",
        ),
    )


class CompOpsWebhookEvent(Base):
    """Replay-protection ledger containing no untrusted event payload."""

    __tablename__ = "compops_webhook_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id = Column(
        UUID(as_uuid=True),
        ForeignKey("compops_evidence_subscriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_id = Column(String(200), nullable=False)
    event_type = Column(String(120), nullable=True)
    payload_sha256 = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="queued")
    evidence_changed = Column(Boolean, nullable=True)
    error = Column(Text, nullable=True)
    received_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "subscription_id",
            "event_id",
            name="uq_compops_webhook_event",
        ),
        Index(
            "ix_compops_webhook_events_status_received",
            "status",
            "received_at",
        ),
    )
