"""Add persistent agent tool priors table.

Revision ID: 0051_add_agent_tool_priors
Revises: 0050_add_presentation_retrieval_trace_id
Create Date: 2026-02-06 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0051_add_agent_tool_priors"
down_revision = "0050_add_presentation_retrieval_trace_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_tool_priors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("tool_name", sa.String(length=120), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("user_id", "job_type", "tool_name", name="uq_agent_tool_priors_user_job_tool"),
    )
    op.create_index("ix_agent_tool_priors_user_id", "agent_tool_priors", ["user_id"])
    op.create_index("ix_agent_tool_priors_job_type", "agent_tool_priors", ["job_type"])
    op.create_index("ix_agent_tool_priors_tool_name", "agent_tool_priors", ["tool_name"])
    op.create_index("ix_agent_tool_priors_user_job", "agent_tool_priors", ["user_id", "job_type"])


def downgrade() -> None:
    op.drop_index("ix_agent_tool_priors_user_job", table_name="agent_tool_priors")
    op.drop_index("ix_agent_tool_priors_tool_name", table_name="agent_tool_priors")
    op.drop_index("ix_agent_tool_priors_job_type", table_name="agent_tool_priors")
    op.drop_index("ix_agent_tool_priors_user_id", table_name="agent_tool_priors")
    op.drop_table("agent_tool_priors")

