"""Transactional outbox rows for autonomous external-system calls."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import (
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


class AgentExternalCallOutbox(Base):
    """Durable outbound request coupled to an agent's database transaction."""

    __tablename__ = "agent_external_call_outbox"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_tools.id", ondelete="SET NULL"),
        nullable=True,
    )

    capability = Column(String(160), nullable=False)
    payload = Column(JSON, nullable=False, default=dict)
    request_id = Column(String(200), nullable=False)
    idempotency_key = Column(String(128), nullable=False)
    correlation = Column(JSON, nullable=True)

    status = Column(String(32), nullable=False, default="pending")
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    next_attempt_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    claim_owner = Column(String(200), nullable=True)
    claim_token = Column(String(64), nullable=True)
    claim_expires_at = Column(DateTime(timezone=True), nullable=True)

    response = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    correlated_at = Column(DateTime(timezone=True), nullable=True)
    resume_claim_owner = Column(String(200), nullable=True)
    resume_claim_token = Column(String(64), nullable=True)
    resume_claim_expires_at = Column(DateTime(timezone=True), nullable=True)
    resume_enqueued_at = Column(DateTime(timezone=True), nullable=True)
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
            "idempotency_key",
            name="uq_agent_external_call_outbox_idempotency_key",
        ),
        UniqueConstraint(
            "request_id",
            name="uq_agent_external_call_outbox_request_id",
        ),
        Index(
            "ix_agent_external_call_outbox_due",
            "status",
            "next_attempt_at",
        ),
        Index(
            "ix_agent_external_call_outbox_resume_due",
            "status",
            "resume_enqueued_at",
            "resume_claim_expires_at",
        ),
        Index("ix_agent_external_call_outbox_job_id", "job_id"),
    )
