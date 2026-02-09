"""
Experiment planning + run tracking.

An ExperimentPlan is typically derived from a ResearchNote (e.g., its Hypothesis section)
and describes datasets, metrics, ablations, and success criteria.

An ExperimentRun records an execution attempt of an experiment plan with results over time.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ExperimentPlan(Base):
    __tablename__ = "experiment_plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    research_note_id = Column(
        UUID(as_uuid=True),
        ForeignKey("research_notes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title = Column(String(500), nullable=False)
    hypothesis_text = Column(Text, nullable=True)
    plan = Column(JSON, nullable=False)  # Structured experiment plan JSON

    # Generation metadata (optional)
    generator = Column(String(100), nullable=True)  # e.g. "llm"
    generator_details = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", backref="experiment_plans")
    research_note = relationship("ResearchNote", backref="experiment_plans")
    runs = relationship("ExperimentRun", back_populates="plan", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ExperimentPlan(id={self.id}, note_id={self.research_note_id})>"


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    experiment_plan_id = Column(
        UUID(as_uuid=True),
        ForeignKey("experiment_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    name = Column(String(500), nullable=False)
    status = Column(String(32), nullable=False, default="planned")  # planned|running|completed|failed|cancelled

    # Run configuration + results
    config = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True)

    # Simple progress tracking
    progress = Column(Integer, default=0, nullable=False)  # 0-100

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", backref="experiment_runs")
    plan = relationship("ExperimentPlan", back_populates="runs")
    agent_job = relationship("AgentJob", backref="experiment_runs")

    def __repr__(self) -> str:
        return f"<ExperimentRun(id={self.id}, plan_id={self.experiment_plan_id}, status={self.status})>"
