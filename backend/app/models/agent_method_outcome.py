"""What became of the runs that carried a method.

A method is recorded with the evidence that established it, recalled into later
jobs, and until now never scored. So a method that misleads is recalled with
exactly the authority of one that works, and the loop that made this project's
*numbers* trustworthy -- predict, measure, score the error -- has no equivalent
for the procedure that produced them.

This is that equivalent. One row per (method, run): the method was in the run's
context, the run either satisfied its contract or did not, its predictions
either settled close or did not. Rows accumulate, and a method's standing is
the aggregate. A method followed by runs that keep failing should be recalled
with that attached, or not recalled first.

The weak link is deliberate and named in the schema. Being *injected* into a
run is not the same as being *followed* by it, and nothing here can watch a
model think. `cited` records the stronger claim -- the run said it was building
on this method -- and the two are counted separately rather than blurred,
because a standing built on the weaker signal would read as evidence it is not.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class AgentMethodOutcome(Base):
    __tablename__ = "agent_method_outcomes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # The memory holding the method record. Kept as a plain id rather than a
    # foreign key: a method may be deleted or re-recorded, and losing the
    # history of what happened under it would be the wrong repair.
    method_memory_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    # Denormalised so a report reads without a join, and so the history
    # survives the memory being removed.
    method_name = Column(String(200), nullable=False)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # True when the run named this method as something it was building on.
    # False when the method was merely in context, which is weaker evidence
    # and is counted apart from it.
    cited = Column(Boolean, default=False, nullable=False)

    # What became of the run. The contract is the deterministic half: it was
    # either satisfied or it was not, whatever the run concluded about itself.
    contract_enabled = Column(Boolean, default=False, nullable=False)
    contract_satisfied = Column(Boolean, default=False, nullable=False)
    unmet_requirements = Column(Text, nullable=True)

    # The calibration half: how many claims the run settled, and how far off
    # they were. A run that predicts nothing scores neither well nor badly.
    predictions_settled = Column(Integer, default=0, nullable=False)
    mean_relative_error = Column(Float, nullable=True)

    iterations = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_agent_method_outcomes_name", "method_name"),
        Index("ix_agent_method_outcomes_created_at", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<AgentMethodOutcome {self.method_name} "
            f"satisfied={self.contract_satisfied} cited={self.cited}>"
        )
