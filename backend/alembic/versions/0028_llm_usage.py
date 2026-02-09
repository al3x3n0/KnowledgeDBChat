"""Add LLM usage events table.

Revision ID: 0028_llm_usage
Revises: 0027_governance_search
Create Date: 2026-01-26
"""

from alembic import op
import sqlalchemy as sa


revision = "0028_llm_usage"
down_revision = "0027_governance_search"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_events",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("task_type", sa.String(length=32), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("input_chars", sa.Integer(), nullable=True),
        sa.Column("output_chars", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.String(length=255), nullable=True),
        sa.Column("extra", sa.dialects.postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_llm_usage_events_user_id", "llm_usage_events", ["user_id"])
    op.create_index("ix_llm_usage_events_provider", "llm_usage_events", ["provider"])
    op.create_index("ix_llm_usage_events_model", "llm_usage_events", ["model"])
    op.create_index("ix_llm_usage_events_task_type", "llm_usage_events", ["task_type"])
    op.create_index("ix_llm_usage_events_created_at", "llm_usage_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_events_created_at", table_name="llm_usage_events")
    op.drop_index("ix_llm_usage_events_task_type", table_name="llm_usage_events")
    op.drop_index("ix_llm_usage_events_model", table_name="llm_usage_events")
    op.drop_index("ix_llm_usage_events_provider", table_name="llm_usage_events")
    op.drop_index("ix_llm_usage_events_user_id", table_name="llm_usage_events")
    op.drop_table("llm_usage_events")

