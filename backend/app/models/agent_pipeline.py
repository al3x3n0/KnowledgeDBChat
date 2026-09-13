"""A saved pipeline spec.

The spec is stored verbatim as JSON rather than decomposed into stage rows.
That is deliberate: `agent_pipeline_spec.normalize` is the authority on what a
pipeline is, and a second definition in table columns would be a second
authority that has to agree with it. A spec that no longer validates — because
a tool was renamed, or an evidence type retired — should still be *readable*,
so its author can see what it said and repair it. Decomposed rows would have
refused to store it in the first place.

What is not in the JSON is what the database is actually for: whose it is,
what it is called, when it last ran, and how often. Those are questions about
the saved thing rather than about the pipeline.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class AgentPipeline(Base):
    """One saved pipeline, belonging to one user."""

    __tablename__ = "agent_pipelines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    #: The spec exactly as authored. See the module docstring for why it is not
    #: decomposed into stage rows.
    spec = Column(JSON, nullable=False)

    #: What the checker said when this was last saved. Cached so a list of
    #: twenty pipelines does not have to re-plan twenty specs to show which
    #: ones are broken — and never trusted as the answer: the tools and their
    #: costs change underneath a saved spec, so the studio re-checks on open.
    last_check_valid = Column(String(16), nullable=True)
    last_estimated_seconds = Column(Integer, nullable=True)

    #: Provenance for the runs it produced.
    launch_count = Column(Integer, nullable=False, default=0)
    last_launched_at = Column(DateTime(timezone=True), nullable=True)
    last_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    last_job = relationship("AgentJob", foreign_keys=[last_job_id])

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_agent_pipeline_user_name"),
    )

    def __repr__(self):
        return f"<AgentPipeline(id={self.id}, name='{self.name}')>"
