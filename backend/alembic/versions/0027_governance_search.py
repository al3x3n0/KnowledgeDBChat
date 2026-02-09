"""Add governance + saved searches.

Revision ID: 0027_governance_search
Revises: 0026_add_reading_lists
Create Date: 2026-01-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0027_governance_search"
down_revision = "0026_add_reading_lists"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # AgentDefinition governance fields
    op.add_column("agent_definitions", sa.Column("owner_user_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("agent_definitions", sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")))
    op.add_column(
        "agent_definitions",
        sa.Column("lifecycle_status", sa.String(length=20), nullable=False, server_default=sa.text("'published'")),
    )
    op.create_index("ix_agent_definitions_owner_user_id", "agent_definitions", ["owner_user_id"])
    op.create_foreign_key(
        "fk_agent_definitions_owner_user_id_users",
        "agent_definitions",
        "users",
        ["owner_user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Secrets vault
    op.create_table(
        "user_secrets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_value", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "name", name="uq_user_secrets_user_name"),
    )
    op.create_index("ix_user_secrets_user_id", "user_secrets", ["user_id"])

    # Tool execution audit + approvals
    op.create_table(
        "tool_execution_audits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_definition_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("tool_input", sa.JSON(), nullable=True),
        sa.Column("tool_output", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("execution_time_ms", sa.Integer(), nullable=True),
        sa.Column("approval_required", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("approval_status", sa.String(length=32), nullable=True),
        sa.Column("approved_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["agent_definition_id"], ["agent_definitions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["conversation_id"], ["agent_conversations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["approved_by"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_tool_execution_audits_user_id", "tool_execution_audits", ["user_id"])
    op.create_index("ix_tool_execution_audits_tool_name", "tool_execution_audits", ["tool_name"])
    op.create_index("ix_tool_execution_audits_agent_definition_id", "tool_execution_audits", ["agent_definition_id"])
    op.create_index("ix_tool_execution_audits_conversation_id", "tool_execution_audits", ["conversation_id"])

    # Saved searches + share links
    op.create_table(
        "saved_searches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("filters", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_saved_searches_user_id", "saved_searches", ["user_id"])

    op.create_table(
        "search_shares",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("filters", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("token", name="uq_search_shares_token"),
    )
    op.create_index("ix_search_shares_token", "search_shares", ["token"])
    op.create_index("ix_search_shares_created_by", "search_shares", ["created_by"])


def downgrade() -> None:
    op.drop_index("ix_search_shares_created_by", table_name="search_shares")
    op.drop_index("ix_search_shares_token", table_name="search_shares")
    op.drop_table("search_shares")

    op.drop_index("ix_saved_searches_user_id", table_name="saved_searches")
    op.drop_table("saved_searches")

    op.drop_index("ix_tool_execution_audits_conversation_id", table_name="tool_execution_audits")
    op.drop_index("ix_tool_execution_audits_agent_definition_id", table_name="tool_execution_audits")
    op.drop_index("ix_tool_execution_audits_tool_name", table_name="tool_execution_audits")
    op.drop_index("ix_tool_execution_audits_user_id", table_name="tool_execution_audits")
    op.drop_table("tool_execution_audits")

    op.drop_index("ix_user_secrets_user_id", table_name="user_secrets")
    op.drop_table("user_secrets")

    op.drop_constraint("fk_agent_definitions_owner_user_id_users", "agent_definitions", type_="foreignkey")
    op.drop_index("ix_agent_definitions_owner_user_id", table_name="agent_definitions")
    op.drop_column("agent_definitions", "lifecycle_status")
    op.drop_column("agent_definitions", "version")
    op.drop_column("agent_definitions", "owner_user_id")

