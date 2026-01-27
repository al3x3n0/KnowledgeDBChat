"""Add repo_report_jobs table for repository report and presentation generation.

Revision ID: 0032_add_repo_report_jobs
Revises: 0031_add_export_jobs
Create Date: 2026-01-27 14:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0032_add_repo_report_jobs"
down_revision = "0031_add_export_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create repo_report_jobs table
    op.create_table(
        "repo_report_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Source reference - either synced DocumentSource or ad-hoc URL
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("adhoc_url", sa.String(length=500), nullable=True),
        sa.Column("adhoc_token", sa.String(length=500), nullable=True),  # Encrypted token for ad-hoc
        # Repository information
        sa.Column("repo_name", sa.String(length=255), nullable=False),
        sa.Column("repo_url", sa.String(length=500), nullable=False),
        sa.Column("repo_type", sa.String(length=20), nullable=False),  # 'github', 'gitlab'
        # Output configuration
        sa.Column("output_format", sa.String(length=20), nullable=False),  # 'docx', 'pdf', 'pptx'
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("sections", postgresql.JSON(), nullable=False),  # List of section names
        sa.Column("slide_count", sa.Integer(), nullable=True),  # For PPTX
        sa.Column("include_diagrams", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # Style
        sa.Column("style", sa.String(length=50), nullable=False, server_default=sa.text("'professional'")),
        sa.Column("custom_theme", postgresql.JSON(), nullable=True),
        # Cached analysis data
        sa.Column("analysis_data", postgresql.JSON(), nullable=True),
        # Status tracking
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("progress", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("current_stage", sa.String(length=100), nullable=True),
        # File storage
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        # Error handling
        sa.Column("error", sa.Text(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        # Foreign keys
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["document_sources.id"], ondelete="SET NULL"),
    )

    # Create indexes for common queries
    op.create_index("ix_repo_report_jobs_user_id", "repo_report_jobs", ["user_id"])
    op.create_index("ix_repo_report_jobs_status", "repo_report_jobs", ["status"])
    op.create_index("ix_repo_report_jobs_created_at", "repo_report_jobs", ["created_at"])
    op.create_index("ix_repo_report_jobs_source_id", "repo_report_jobs", ["source_id"])
    op.create_index("ix_repo_report_jobs_repo_type", "repo_report_jobs", ["repo_type"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_repo_report_jobs_repo_type", table_name="repo_report_jobs")
    op.drop_index("ix_repo_report_jobs_source_id", table_name="repo_report_jobs")
    op.drop_index("ix_repo_report_jobs_created_at", table_name="repo_report_jobs")
    op.drop_index("ix_repo_report_jobs_status", table_name="repo_report_jobs")
    op.drop_index("ix_repo_report_jobs_user_id", table_name="repo_report_jobs")
    # Drop table
    op.drop_table("repo_report_jobs")
