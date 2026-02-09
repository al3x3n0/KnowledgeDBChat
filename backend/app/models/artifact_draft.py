"""
Artifact drafts (human-in-the-loop review objects).

Generalizes the PatchPR concept to non-code artifacts such as:
- presentations (PPTX jobs)
- repo reports (DOCX/PDF/PPTX jobs)

Drafts are created from an existing job output and then go through:
draft -> in_review -> approved -> published

Approvals are recorded as JSON events and (by default) require:
- resource owner approval (draft.user_id)
- admin approval (any admin user)
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ArtifactDraft(Base):
    __tablename__ = "artifact_drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # presentation | repo_report | other
    artifact_type = Column(String(32), nullable=False, index=True)

    # Opaque source reference to the generating job.
    # We don't enforce a FK because different types map to different tables.
    source_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    status = Column(String(24), nullable=False, default="draft")  # draft|in_review|approved|published|rejected

    # Draft payload (JSON) must include enough to render a preview/diff.
    draft_payload = Column(JSON, nullable=False, default=dict)

    # Optional published snapshot; for now this mirrors the draft payload on publish.
    published_payload = Column(JSON, nullable=True)

    # Approvals list: [{"user_id":"...", "role":"owner|admin", "at":"...", "note":"..."}]
    approvals = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_artifact_drafts_user_status", "user_id", "status"),
        Index("ix_artifact_drafts_user_created", "user_id", "created_at"),
    )

