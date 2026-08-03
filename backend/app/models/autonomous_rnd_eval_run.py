"""Persisted autonomous R&D evaluation runs and per-suite baselines."""

from __future__ import annotations

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base

EVAL_RUN_SOURCE_GRADE_JOBS = "grade_jobs"
EVAL_RUN_SOURCE_REPLAY = "replay"


class AutonomousRndEvalRun(Base):
    """One graded suite execution, kept so scores stay comparable over time.

    Trial detail lives inline in ``report``; the headline metrics are
    denormalized into columns so trend and baseline-comparison queries do not
    have to unpack JSON.
    """

    __tablename__ = "autonomous_rnd_eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    suite_id = Column(String(200), nullable=False, index=True)
    suite_name = Column(String(300), nullable=False)
    suite_version = Column(Integer, nullable=False)
    label = Column(String(200), nullable=True)
    source = Column(String(32), nullable=False, default=EVAL_RUN_SOURCE_GRADE_JOBS)
    is_baseline = Column(Boolean, nullable=False, default=False, index=True)

    task_count = Column(Integer, nullable=False, default=0)
    trial_count = Column(Integer, nullable=False, default=0)
    mean_score = Column(Float, nullable=False, default=0.0)
    pass_at_k = Column(Float, nullable=False, default=0.0)
    pass_pow_k = Column(Float, nullable=False, default=0.0)

    report = Column(JSONB, nullable=False)
    task_bindings = Column(JSONB, nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Migration 0080 additionally creates a partial unique index
    # (user_id, suite_id) WHERE is_baseline, so a suite cannot end up with two
    # comparison anchors. It is declared only in the migration because a
    # partial index has no portable SQLAlchemy spelling that degrades safely on
    # the SQLite test dialect.
    __table_args__ = (
        Index(
            "ix_rnd_eval_runs_user_suite_created",
            "user_id",
            "suite_id",
            "created_at",
        ),
    )
