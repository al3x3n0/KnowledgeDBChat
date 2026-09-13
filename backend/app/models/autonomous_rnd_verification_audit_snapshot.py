"""Immutable registry entries for signed autonomous R&D verification audits."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base


class AutonomousRndVerificationAuditSnapshot(Base):
    """Append-only signed snapshot; deletion remains available for privacy erasure."""

    __tablename__ = "autonomous_rnd_verification_audit_snapshots"

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
    schema_version = Column(Integer, nullable=False)
    snapshot = Column(JSONB, nullable=False)
    canonicalization = Column(String(64), nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)
    signature_algorithm = Column(String(32), nullable=False)
    signature_encoding = Column(String(16), nullable=False)
    signature = Column(String(128), nullable=False)
    key_id = Column(String(120), nullable=False, index=True)
    public_key = Column(String(64), nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    __table_args__ = (
        Index(
            "ix_rnd_verification_audit_job_created",
            "job_id",
            "created_at",
        ),
    )
