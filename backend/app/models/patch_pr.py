"""
Patch PRs (proposal review objects).

A PatchPR is a lightweight "PR-style" wrapper around one or more CodePatchProposal
objects. It supports:
- proposal history
- selected proposal
- checks/metadata
- approvals
- merge (apply selected proposal to KB code docs)
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text, Index
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class PatchPR(Base):
    __tablename__ = "patch_prs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id", ondelete="SET NULL"), nullable=True, index=True)

    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)

    status = Column(String(24), nullable=False, default="draft")  # draft | open | approved | merged | rejected

    # Selected proposal to merge/apply.
    selected_proposal_id = Column(UUID(as_uuid=True), ForeignKey("code_patch_proposals.id", ondelete="SET NULL"), nullable=True, index=True)

    # Proposal history (UUIDs as strings).
    proposal_ids = Column(JSON, nullable=True)

    # Checks/metadata: experiments, lint, compile, reviewer notes, etc.
    checks = Column(JSON, nullable=True)

    # Approvals list: [{"user_id": "...", "at": "...", "note": "..."}]
    approvals = Column(JSON, nullable=True)

    merged_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_patch_prs_user_status", "user_id", "status"),
        Index("ix_patch_prs_user_created", "user_id", "created_at"),
    )

