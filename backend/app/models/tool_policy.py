"""
Tool policies for governing tool execution across agents, MCP, and workflows.

Policy model is intentionally simple:
- allow-by-default
- explicit denies
- optional "require approval" flags
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from app.core.database import Base


class ToolPolicy(Base):
    __tablename__ = "tool_policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Who the policy applies to.
    # - global: applies to everyone
    # - role: applies to users with a role (subject_id unused; role stored in subject_key)
    # - user: applies to a specific user_id
    # - agent_definition: applies to a specific agent_definition_id
    # - api_key: applies to a specific api_key_id
    subject_type = Column(String(32), nullable=False, default="global")
    subject_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    subject_key = Column(String(128), nullable=True, index=True)

    # Tool selector
    # - exact tool name, or "*" for all tools
    tool_name = Column(String(120), nullable=False, index=True)

    # Policy effect: allow or deny
    effect = Column(String(8), nullable=False, default="deny")  # allow | deny

    # Optional "requires approval" gate.
    require_approval = Column(Boolean, nullable=False, default=False)

    # Optional constraints (JSONB)
    # Examples:
    # - {"allowed_domains": ["wiki.company.com"], "deny_private_networks": true}
    # - {"max_cost_tier": "low"}
    constraints = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_tool_policies_subject", "subject_type", "subject_id", "subject_key"),
        Index("ix_tool_policies_selector", "tool_name", "effect"),
    )

