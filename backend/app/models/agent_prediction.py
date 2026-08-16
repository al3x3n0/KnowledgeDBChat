"""Predictions an agent made, and what actually happened.

An agent asked to invent its own methodology will produce something plausible
every time; plausible is not the same as good. This table is the difference:
the agent states what it expects to happen and how it reached that, the ground
truth is measured afterwards, and the error is the score. Methodology then
becomes something the agent can improve against a number rather than a story --
does sampling predict as well as a full run, is an assumed latency defensible,
is one candidate-ranking rule better than another.

The order matters more than the fields. A prediction recorded after its
measurement is unfalsifiable, so a row is created with its prediction and only
later updated with what was measured; see the service for the rules that keep
that true. Rows accumulate across runs, which is the point: the next run reads
the error history of the methodology it is about to reuse.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class AgentPrediction(Base):
    __tablename__ = "agent_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

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

    # What the prediction is about: a candidate instruction, a kernel, a
    # workload region. Free text because the agent chooses what it studies.
    subject = Column(String(300), nullable=False)
    # The quantity, e.g. "speedup", "cycles_per_iteration". Comparing errors
    # across different metrics is meaningless, so it is recorded, not assumed.
    metric = Column(String(120), nullable=False)

    # How the agent arrived at the number. The tags are what later runs group
    # by when asking which approach has been predicting well.
    methodology = Column(Text, nullable=False)
    methodology_tags = Column(JSON, nullable=True)
    prediction_basis = Column(Text, nullable=True)

    predicted_value = Column(Float, nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    measured_value = Column(Float, nullable=True)
    measured_at = Column(DateTime, nullable=True)
    # Which referee produced the ground truth, e.g. "gem5 O3 neoverse-n1".
    # A measurement without its source cannot be compared with another.
    measurement_source = Column(String(300), nullable=True)

    # Stored rather than computed on read so a query can rank by accuracy
    # without recomputing, and so the arithmetic is done once, in one place.
    error_absolute = Column(Float, nullable=True)
    error_relative = Column(Float, nullable=True)

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_agent_predictions_subject_metric", "subject", "metric"),
        Index("ix_agent_predictions_measured_at", "measured_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "open" if self.measured_value is None else "settled"
        return (
            f"<AgentPrediction({self.subject!r}, {self.metric!r}, "
            f"predicted={self.predicted_value}, {state})>"
        )
