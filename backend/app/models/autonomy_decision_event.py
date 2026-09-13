"""
Persistent autonomy/operator decision events.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class AutonomyDecisionEvent(Base):
    __tablename__ = "autonomy_decision_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    event_time = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )
    event_type = Column(String(80), nullable=False, index=True)
    source_kind = Column(String(64), nullable=False, index=True)
    source_id = Column(String(128), nullable=True, index=True)
    source_label = Column(String(255), nullable=True)
    customer = Column(String(255), nullable=True, index=True)
    decision_type = Column(String(80), nullable=False, index=True)
    reason_code = Column(String(128), nullable=True, index=True)
    status = Column(String(64), nullable=True, index=True)
    severity = Column(String(32), nullable=True, index=True)
    actor_mode = Column(String(24), nullable=True, index=True)

    summary = Column(Text, nullable=False)
    operator_note = Column(Text, nullable=True)
    before_state = Column(JSON, nullable=True)
    after_state = Column(JSON, nullable=True)
    deep_link = Column(JSON, nullable=True)
    event_metadata = Column("metadata", JSON, nullable=True)

    is_derived = Column(Boolean, nullable=False, default=False)
    record_origin = Column(String(24), nullable=False, default="persisted")
    triage_status = Column(String(24), nullable=False, default="new", index=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolution_note = Column(Text, nullable=True)
    pinned = Column(Boolean, nullable=False, default=False, index=True)
    last_viewed_at = Column(DateTime(timezone=True), nullable=True)
    assigned_to_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    assigned_by_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    team_bucket = Column(String(64), nullable=True, index=True)
    due_at = Column(DateTime(timezone=True), nullable=True, index=True)
    escalation_state = Column(String(24), nullable=False, default="none", index=True)
    escalation_reason = Column(String(255), nullable=True)
    escalated_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
