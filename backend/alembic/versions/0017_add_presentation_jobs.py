"""Add presentation_jobs table for AI-powered presentation generation.

Revision ID: 0017_add_presentation_jobs
Revises: 0016_add_workflows
Create Date: 2024-06-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0017_add_presentation_jobs"
down_revision = "0016_add_workflows"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create presentation_jobs table
    op.create_table(
        "presentation_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Presentation metadata
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        # Generation settings
        sa.Column("source_document_ids", postgresql.JSON(), nullable=True, server_default=sa.text("'[]'::json")),
        sa.Column("slide_count", sa.Integer(), nullable=False, server_default=sa.text("10")),
        sa.Column("style", sa.String(length=50), nullable=False, server_default=sa.text("'professional'")),
        sa.Column("include_diagrams", sa.Integer(), nullable=False, server_default=sa.text("1")),
        # Status tracking
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("progress", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("current_stage", sa.String(length=100), nullable=True),
        # Generated data
        sa.Column("generated_outline", postgresql.JSON(), nullable=True),
        # File storage
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        # Error handling
        sa.Column("error", sa.Text(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        # Foreign key
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # Create indexes for common queries
    op.create_index("ix_presentation_jobs_user_id", "presentation_jobs", ["user_id"])
    op.create_index("ix_presentation_jobs_status", "presentation_jobs", ["status"])
    op.create_index("ix_presentation_jobs_created_at", "presentation_jobs", ["created_at"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_presentation_jobs_created_at", table_name="presentation_jobs")
    op.drop_index("ix_presentation_jobs_status", table_name="presentation_jobs")
    op.drop_index("ix_presentation_jobs_user_id", table_name="presentation_jobs")
    # Drop table
    op.drop_table("presentation_jobs")
