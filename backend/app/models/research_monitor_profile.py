"""
Research monitor profiles.

Stores lightweight, per-user (+ optional per-customer) preferences learned from
Research Inbox triage (accept/reject) so continuous monitors improve over time.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ResearchMonitorProfile(Base):
    __tablename__ = "research_monitor_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # customer tag used by inbox items; NULL means "global defaults for this user"
    customer = Column(String(255), nullable=True, index=True)

    # Token scores learned from triage (positive favors, negative suppresses)
    token_scores = Column(JSON, nullable=True)

    # Optional explicit mutes (tokens/patterns) set by user later
    muted_tokens = Column(JSON, nullable=True)
    muted_patterns = Column(JSON, nullable=True)

    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "customer", name="uq_research_monitor_profile_user_customer"),
        Index("ix_research_monitor_profiles_user_customer", "user_id", "customer"),
    )

