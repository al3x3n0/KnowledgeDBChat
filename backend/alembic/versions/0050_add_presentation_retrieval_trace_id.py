"""Add retrieval_trace_id to presentation_jobs.

Revision ID: 0050_add_presentation_retrieval_trace_id
Revises: 0049_add_artifact_drafts_and_retrieval_traces
Create Date: 2026-02-05 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0050_add_presentation_retrieval_trace_id"
down_revision = "0049_add_artifact_drafts_and_retrieval_traces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "presentation_jobs",
        sa.Column(
            "retrieval_trace_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval_traces.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_presentation_jobs_retrieval_trace_id", "presentation_jobs", ["retrieval_trace_id"])


def downgrade() -> None:
    op.drop_index("ix_presentation_jobs_retrieval_trace_id", table_name="presentation_jobs")
    op.drop_column("presentation_jobs", "retrieval_trace_id")

