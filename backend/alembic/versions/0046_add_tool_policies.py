"""Add tool_policies table.

Revision ID: 0046_add_tool_policies
Revises: 0045_add_chat_groundedness_score
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0046_add_tool_policies"
down_revision = "0045_add_chat_groundedness_score"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_policies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("subject_type", sa.String(length=32), nullable=False, server_default="global"),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("subject_key", sa.String(length=128), nullable=True),
        sa.Column("tool_name", sa.String(length=120), nullable=False),
        sa.Column("effect", sa.String(length=8), nullable=False, server_default="deny"),
        sa.Column("require_approval", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("constraints", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_tool_policies_subject_id", "tool_policies", ["subject_id"])
    op.create_index("ix_tool_policies_subject_key", "tool_policies", ["subject_key"])
    op.create_index("ix_tool_policies_tool_name", "tool_policies", ["tool_name"])
    op.create_index("ix_tool_policies_subject", "tool_policies", ["subject_type", "subject_id", "subject_key"])
    op.create_index("ix_tool_policies_selector", "tool_policies", ["tool_name", "effect"])


def downgrade() -> None:
    op.drop_index("ix_tool_policies_selector", table_name="tool_policies")
    op.drop_index("ix_tool_policies_subject", table_name="tool_policies")
    op.drop_index("ix_tool_policies_tool_name", table_name="tool_policies")
    op.drop_index("ix_tool_policies_subject_key", table_name="tool_policies")
    op.drop_index("ix_tool_policies_subject_id", table_name="tool_policies")
    op.drop_table("tool_policies")

