"""Add customer_profile_id to AI Hub recommendation feedback.

Revision ID: 0040_ai_hub_feedback_profile_id
Revises: 0039_ai_hub_recommendation_feedback
Create Date: 2026-01-30
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0040_ai_hub_feedback_profile_id"
down_revision = "0039_ai_hub_recommendation_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ai_hub_recommendation_feedback",
        sa.Column("customer_profile_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_ai_hub_recommendation_feedback_customer_profile_id",
        "ai_hub_recommendation_feedback",
        ["customer_profile_id"],
    )

    # Replace the composite index to prefer profile_id.
    try:
        op.drop_index("ix_ai_hub_reco_feedback_profile_item", table_name="ai_hub_recommendation_feedback")
    except Exception:
        pass
    op.create_index(
        "ix_ai_hub_reco_feedback_profile_item",
        "ai_hub_recommendation_feedback",
        ["customer_profile_id", "workflow", "item_type", "item_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_ai_hub_reco_feedback_profile_item", table_name="ai_hub_recommendation_feedback")
    op.drop_index("ix_ai_hub_recommendation_feedback_customer_profile_id", table_name="ai_hub_recommendation_feedback")
    op.drop_column("ai_hub_recommendation_feedback", "customer_profile_id")

