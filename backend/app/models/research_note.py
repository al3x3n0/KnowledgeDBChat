"""
Research Note model.

Research-native notes intended for labs: hypotheses, experiment plans, insights.
Typically created from synthesis jobs but can also be authored manually.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ResearchNote(Base):
    """Research note authored by a user."""

    __tablename__ = "research_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title = Column(String(500), nullable=False)
    content_markdown = Column(Text, nullable=False)

    # Provenance (optional)
    source_synthesis_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("synthesis_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_document_ids = Column(JSON, nullable=True)  # list[str UUID]

    tags = Column(JSON, nullable=True)  # list[str]
    attribution = Column(
        JSON, nullable=True
    )  # citation enforcement report + generated markdown (optional)
    structured_payload = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("User", backref="research_notes")
    source_synthesis_job = relationship(
        "SynthesisJob", foreign_keys=[source_synthesis_job_id]
    )

    def __repr__(self) -> str:
        return f"<ResearchNote(id={self.id}, user_id={self.user_id}, title={self.title!r})>"
