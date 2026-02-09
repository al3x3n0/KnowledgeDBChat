"""
Audit log for agent tool calls and approvals.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
from sqlalchemy.sql import func

from app.core.database import Base


class ToolExecutionAudit(Base):
    __tablename__ = "tool_execution_audits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_definition_id = Column(UUID(as_uuid=True), ForeignKey("agent_definitions.id", ondelete="SET NULL"), nullable=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("agent_conversations.id", ondelete="SET NULL"), nullable=True, index=True)

    tool_name = Column(String(100), nullable=False, index=True)
    tool_input = Column(JSON, nullable=True)
    tool_output = Column(JSON, nullable=True)

    # Captures the policy evaluation result at the time the audit row is created/updated.
    # Useful for provenance, debugging, and evals.
    policy_decision = Column(JSONB, nullable=True)

    status = Column(String(32), nullable=False, default="pending")  # pending, running, requires_approval, completed, failed
    error = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)

    approval_required = Column(Boolean, nullable=False, default=False)
    # Approval model:
    # - approval_mode: "owner_or_admin" (legacy) or "owner_and_admin" (default)
    # - approval_status:
    #   - pending_owner / pending_admin for owner_and_admin mode
    #   - pending / approved / rejected for owner_or_admin mode
    approval_mode = Column(String(32), nullable=True, default="owner_and_admin")
    approval_status = Column(String(32), nullable=True)
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approval_note = Column(Text, nullable=True)

    owner_approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    owner_approved_at = Column(DateTime(timezone=True), nullable=True)
    admin_approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    admin_approved_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
