"""A result that was believed and then checked, and what it takes down with it.

The knowledge base is append-only, which is right for findings and wrong for
findings that turn out to be defective. This project retracted an entire
per-instruction measurement table -- four of nine classes had been timed on
chains that reached infinity within a few iterations, so the harness measured
exceptional-value arithmetic rather than the instructions named. Nothing in the
system noticed. Every method validated against those numbers kept its standing,
every finding derived from them kept its authority, and a campaign running
unattended would have gone on citing them for as long as it ran.

An autonomous programme accumulates poison faster than it accumulates results
unless retraction is a supported operation rather than a cleanup someone
remembers to do.

Two decisions worth naming.

**A reason is required.** "This was withdrawn" is not a record; a later run has
to be able to tell a measurement retracted for a harness defect from one
retracted because the question changed, and only the reason distinguishes them.

**Nothing is deleted.** The retraction is a row beside the thing, not a removal
of it, for the same reason a method with a poor record is demoted rather than
dropped: the evidence that something was once believed is itself evidence, and
a knowledge base that quietly loses its mistakes cannot be audited.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class RetractionKind:
    """What is being withdrawn.

    Three, because the real cases differ in scope. A whole run can be bad (its
    host was never verified). A *class* of measurement can be bad across every
    run that took it, which is what happened to reciprocal throughput here. And
    a recorded method can be bad on its own terms.
    """

    JOB = "job"
    FINDING_TYPE = "finding_type"
    METHOD = "method"

    ALL = (JOB, FINDING_TYPE, METHOD)


class AgentRetraction(Base):
    __tablename__ = "agent_retractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    subject_kind = Column(String(40), nullable=False)
    #: A job id, a finding type name, or a method memory id, depending on kind.
    #: Text rather than a foreign key: the subject may be deleted, and losing
    #: the record that it was retracted would be the wrong repair.
    subject_ref = Column(String(300), nullable=False)

    #: Required. See the module docstring.
    reason = Column(Text, nullable=False)

    #: Where the decision came from -- a job that discovered the defect, or an
    #: operator. Kept so a retraction can be traced the way a finding can.
    source = Column(String(200), nullable=True)
    source_job_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_agent_retractions_user_kind", "user_id", "subject_kind"),
        Index("ix_agent_retractions_subject", "subject_kind", "subject_ref"),
    )
