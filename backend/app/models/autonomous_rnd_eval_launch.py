"""Fan-out of an autonomous R&D evaluation suite into real agent-job trials."""

from __future__ import annotations

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base

# A launch is created before its trial jobs exist, so that a crash mid-fan-out
# leaves an owner for the jobs already committed instead of orphans.
EVAL_LAUNCH_STATUS_PENDING = "pending"
EVAL_LAUNCH_STATUS_RUNNING = "running"
EVAL_LAUNCH_STATUS_COMPLETED = "completed"
EVAL_LAUNCH_STATUS_FAILED = "failed"
EVAL_LAUNCH_STATUS_CANCELLED = "cancelled"

EVAL_LAUNCH_TERMINAL_STATUSES = (
    EVAL_LAUNCH_STATUS_COMPLETED,
    EVAL_LAUNCH_STATUS_FAILED,
    EVAL_LAUNCH_STATUS_CANCELLED,
)


class AutonomousRndEvalLaunch(Base):
    """One suite fan-out: N agent jobs per task, graded once they all settle.

    ``task_bindings`` maps each suite task id to the trial job ids created for
    it, which is the same binding shape the grade-jobs endpoint accepts. The
    graded run is linked through ``run_id`` when finalization succeeds.
    """

    __tablename__ = "autonomous_rnd_eval_launches"

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
    status = Column(
        String(32), nullable=False, default=EVAL_LAUNCH_STATUS_RUNNING, index=True
    )
    trials_per_task = Column(Integer, nullable=False)
    job_count = Column(Integer, nullable=False, default=0)
    task_bindings = Column(JSONB, nullable=False)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("autonomous_rnd_eval_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "ix_rnd_eval_launches_user_status_created",
            "user_id",
            "status",
            "created_at",
        ),
    )
