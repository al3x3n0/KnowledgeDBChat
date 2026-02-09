"""Add AI Hub recommendation feedback table.

Revision ID: 0039_ai_hub_recommendation_feedback
Revises: 0038
Create Date: 2026-01-30
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0039_ai_hub_recommendation_feedback"
down_revision = "0038a_widen_alembic_version_num"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_hub_recommendation_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("customer_profile_name", sa.String(length=200), nullable=True),
        sa.Column("workflow", sa.String(length=32), nullable=False),
        sa.Column("item_type", sa.String(length=32), nullable=False),
        sa.Column("item_id", sa.String(length=200), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("customer_keywords", postgresql.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["agent_job_id"], ["agent_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_ai_hub_recommendation_feedback_user_id", "ai_hub_recommendation_feedback", ["user_id"])
    op.create_index("ix_ai_hub_recommendation_feedback_agent_job_id", "ai_hub_recommendation_feedback", ["agent_job_id"])
    op.create_index("ix_ai_hub_recommendation_feedback_customer_profile_name", "ai_hub_recommendation_feedback", ["customer_profile_name"])
    op.create_index("ix_ai_hub_recommendation_feedback_workflow", "ai_hub_recommendation_feedback", ["workflow"])
    op.create_index("ix_ai_hub_recommendation_feedback_item_type", "ai_hub_recommendation_feedback", ["item_type"])
    op.create_index("ix_ai_hub_recommendation_feedback_item_id", "ai_hub_recommendation_feedback", ["item_id"])
    op.create_index("ix_ai_hub_recommendation_feedback_decision", "ai_hub_recommendation_feedback", ["decision"])

    op.create_index(
        "ix_ai_hub_reco_feedback_profile_item",
        "ai_hub_recommendation_feedback",
        ["customer_profile_name", "workflow", "item_type", "item_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_ai_hub_reco_feedback_profile_item", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_decision", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_item_id", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_item_type", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_workflow", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_customer_profile_name", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_agent_job_id", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_user_id", table_name="ai_hub_recommendation_feedback")
    op.drop_table("ai_hub_recommendation_feedback")
