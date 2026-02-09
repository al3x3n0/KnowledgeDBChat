"""
Research Inbox models.

Stores discovered items from continuous/recurring monitoring jobs so users can
triage (accept/reject) and feed signals back into future monitoring.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ResearchInboxItem(Base):
    __tablename__ = "research_inbox_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True, index=True)

    # Customer/workspace tag (deployment-level profile name or a user-supplied label)
    customer = Column(String(255), nullable=True, index=True)

    # Item identity (dedupe per-user)
    item_type = Column(String(32), nullable=False)  # "document" | "arxiv" | ...
    item_key = Column(String(512), nullable=False)  # stable identifier (document_id, arxiv_id, url hash, etc.)

    # Display fields
    title = Column(String(1000), nullable=False)
    summary = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    discovered_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True)

    # Triage
    status = Column(String(16), nullable=False, default="new")  # new | accepted | rejected
    feedback = Column(Text, nullable=True)

    # Source-specific metadata
    item_metadata = Column("metadata", JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "item_type", "item_key", name="uq_research_inbox_item_once"),
        Index("ix_research_inbox_user_status", "user_id", "status"),
    )
