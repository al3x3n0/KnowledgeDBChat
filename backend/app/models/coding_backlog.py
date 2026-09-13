from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class CodingBacklogItem(Base):
    __tablename__ = "coding_backlog_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    title = Column(String(200), nullable=False)
    portfolio_goal = Column(Text, nullable=False)
    status = Column(String(24), nullable=False, default="draft", index=True)
    priority = Column(Integer, nullable=False, default=50, index=True)
    scope = Column(String(32), nullable=True)

    failure_symptom = Column(Text, nullable=True)
    error_output = Column(Text, nullable=True)
    file_paths = Column(JSON, nullable=True)
    commands = Column(JSON, nullable=True)

    auto_apply_enabled = Column(Boolean, nullable=False, default=True)
    require_patch_pr = Column(Boolean, nullable=False, default=False)
    visibility = Column(String(24), nullable=False, default="private", index=True)
    shared_with_user_ids = Column(JSON, nullable=True)
    assigned_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    assigned_by_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    collaboration = Column(JSON, nullable=True)
    policy = Column(JSON, nullable=True)
    lineage = Column(JSON, nullable=True)
    decomposition = Column(JSON, nullable=True)
    child_job_ids = Column(JSON, nullable=True)
    latest_summary = Column(JSON, nullable=True)

    orchestrator_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    current_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    latest_apply_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    latest_proposal_id = Column(
        UUID(as_uuid=True),
        ForeignKey("code_patch_proposals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
