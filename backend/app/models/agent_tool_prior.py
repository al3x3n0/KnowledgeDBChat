"""
Persistent tool priors for autonomous agents.

Stores per-user, per-job-type historical tool outcomes used to bias
future autonomous decisions.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class AgentToolPrior(Base):
    __tablename__ = "agent_tool_priors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_type = Column(String(50), nullable=False, index=True)
    tool_name = Column(String(120), nullable=False, index=True)

    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "job_type", "tool_name", name="uq_agent_tool_priors_user_job_tool"),
        Index("ix_agent_tool_priors_user_job", "user_id", "job_type"),
    )

