from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class DomainResearchProfile(Base):
    __tablename__ = "domain_research_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title = Column(String(200), nullable=False)
    domain = Column(String(300), nullable=False)
    objective = Column(Text, nullable=False)
    customer_context = Column(Text, nullable=True)
    status = Column(String(24), nullable=False, default="draft", index=True)

    source_scope = Column(String(32), nullable=False, default="kb_plus_arxiv")
    track_type = Column(String(32), nullable=False, default="generic")
    research_mode = Column(
        String(48), nullable=False, default="literature_to_hypothesis"
    )
    monitor_queries = Column(JSON, nullable=True)
    repo_source_ids = Column(JSON, nullable=True)
    benchmark_queries = Column(JSON, nullable=True)
    report_format = Column(String(32), nullable=False, default="brief_and_report")
    scoring_policy = Column(JSON, nullable=True)
    selection_policy = Column(JSON, nullable=True)
    validation_policy = Column(JSON, nullable=True)
    automation_profile = Column(String(24), nullable=False, default="balanced")
    automation_policy = Column(JSON, nullable=True)
    sandbox_profile_id = Column(String(80), nullable=True)
    interval_minutes = Column(Integer, nullable=False, default=1440)
    persist_artifacts = Column(Boolean, nullable=False, default=True)
    auto_launch_follow_up = Column(Boolean, nullable=False, default=True)
    auto_create_experiment_plans = Column(Boolean, nullable=False, default=True)
    confidence_threshold = Column(Float, nullable=False, default=0.7)
    max_documents = Column(Integer, nullable=False, default=10)
    max_papers = Column(Integer, nullable=False, default=8)

    latest_summary = Column(JSON, nullable=True)
    latest_note_ids = Column(JSON, nullable=True)
    latest_experiment_plan_ids = Column(JSON, nullable=True)
    latest_validation_run_ids = Column(JSON, nullable=True)

    latest_run_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    active_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
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
    paused_at = Column(DateTime(timezone=True), nullable=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
