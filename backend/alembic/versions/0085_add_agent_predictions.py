"""Add agent_predictions: what an agent expected, and what was measured

The table exists to score an agent's methodology by prediction error, so its
integrity rule is about order rather than shape: a row is created carrying a
prediction and only later updated with the measurement. A prediction written
after its outcome is unfalsifiable, and an error column computed from one
measures nothing.

Revision ID: 0085_add_agent_predictions
Revises: 0084_rename_sandbox_research_images
Create Date: 2026-08-16

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0085_add_agent_predictions"
down_revision = "0084_rename_sandbox_research_images"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_jobs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("subject", sa.String(length=300), nullable=False),
        sa.Column("metric", sa.String(length=120), nullable=False),
        sa.Column("methodology", sa.Text(), nullable=False),
        sa.Column("methodology_tags", sa.JSON(), nullable=True),
        sa.Column("prediction_basis", sa.Text(), nullable=True),
        sa.Column("predicted_value", sa.Float(), nullable=False),
        sa.Column("predicted_at", sa.DateTime(), nullable=False),
        sa.Column("measured_value", sa.Float(), nullable=True),
        sa.Column("measured_at", sa.DateTime(), nullable=True),
        sa.Column("measurement_source", sa.String(length=300), nullable=True),
        sa.Column("error_absolute", sa.Float(), nullable=True),
        sa.Column("error_relative", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_agent_predictions_user_id", "agent_predictions", ["user_id"])
    op.create_index("ix_agent_predictions_job_id", "agent_predictions", ["job_id"])
    op.create_index(
        "ix_agent_predictions_subject_metric",
        "agent_predictions",
        ["subject", "metric"],
    )
    op.create_index(
        "ix_agent_predictions_measured_at", "agent_predictions", ["measured_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_agent_predictions_measured_at", table_name="agent_predictions")
    op.drop_index("ix_agent_predictions_subject_metric", table_name="agent_predictions")
    op.drop_index("ix_agent_predictions_job_id", table_name="agent_predictions")
    op.drop_index("ix_agent_predictions_user_id", table_name="agent_predictions")
    op.drop_table("agent_predictions")
