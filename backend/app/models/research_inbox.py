"""
Research Inbox models.

Stores discovered items from continuous/recurring monitoring jobs so users can
triage (accept/reject) and feed signals back into future monitoring.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ResearchInboxItem(Base):
    __tablename__ = "research_inbox_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Customer/workspace tag (deployment-level profile name or a user-supplied label)
    customer = Column(String(255), nullable=True, index=True)

    # Item identity (dedupe per-user)
    item_type = Column(String(32), nullable=False)  # "document" | "arxiv" | ...
    item_key = Column(
        String(512), nullable=False
    )  # stable identifier (document_id, arxiv_id, url hash, etc.)

    # Display fields
    title = Column(String(1000), nullable=False)
    summary = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    discovered_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True
    )

    # Triage
    status = Column(
        String(16), nullable=False, default="new"
    )  # new | accepted | rejected
    feedback = Column(Text, nullable=True)

    # Source-specific metadata
    item_metadata = Column("metadata", JSON, nullable=True)

    # Follow-up autonomy / execution state
    follow_up_decision = Column(String(32), nullable=True)
    follow_up_policy_mode = Column(String(32), nullable=True)
    follow_up_launch_status = Column(String(32), nullable=True)
    follow_up_block_reason = Column(Text, nullable=True)
    follow_up_budget_decision = Column(String(32), nullable=True)
    follow_up_budget_reason = Column(Text, nullable=True)
    follow_up_budget_throttle_state = Column(String(32), nullable=True)
    follow_up_customer_budget_decision = Column(String(32), nullable=True)
    follow_up_customer_budget_reason = Column(Text, nullable=True)
    follow_up_customer_budget_throttle_state = Column(String(32), nullable=True)
    follow_up_recommendation_key = Column(String(100), nullable=True)
    follow_up_operator_decision = Column(String(32), nullable=True)
    follow_up_operator_note = Column(Text, nullable=True)
    follow_up_operator_acted_at = Column(DateTime(timezone=True), nullable=True)
    follow_up_operator_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    follow_up_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    follow_up_chain_definition_id = Column(UUID(as_uuid=True), nullable=True)
    follow_up_launched_at = Column(DateTime(timezone=True), nullable=True)
    follow_up_outcome_status = Column(String(32), nullable=True)
    follow_up_outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)
    follow_up_outcome_summary = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "item_type", "item_key", name="uq_research_inbox_item_once"
        ),
        Index("ix_research_inbox_user_status", "user_id", "status"),
    )
