from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ResearchPortfolio(Base):
    __tablename__ = "research_portfolios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    title = Column(String(200), nullable=False)
    objective = Column(Text, nullable=False)
    status = Column(String(24), nullable=False, default="draft", index=True)

    linked_profile_ids = Column(JSON, nullable=True)
    automation_profile = Column(String(24), nullable=False, default="balanced")
    automation_policy = Column(JSON, nullable=True)
    sandbox_profile_id = Column(String(80), nullable=True)
    opportunities = Column(JSON, nullable=True)
    latest_summary = Column(JSON, nullable=True)
    latest_note_ids = Column(JSON, nullable=True)
    latest_experiment_plan_ids = Column(JSON, nullable=True)
    latest_validation_run_ids = Column(JSON, nullable=True)
    child_job_ids = Column(JSON, nullable=True)

    active_job_id = Column(UUID(as_uuid=True), ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    latest_run_job_id = Column(UUID(as_uuid=True), ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    paused_at = Column(DateTime(timezone=True), nullable=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
