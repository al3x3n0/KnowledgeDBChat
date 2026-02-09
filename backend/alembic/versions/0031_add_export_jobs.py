"""Add export_jobs table for DOCX/PDF document generation.

Revision ID: 0031_add_export_jobs
Revises: 0030_compiler_expert
Create Date: 2026-01-27 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0031_add_export_jobs"
down_revision = "0030_compiler_expert"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create export_jobs table
    op.create_table(
        "export_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Export configuration
        sa.Column("export_type", sa.String(length=50), nullable=False),  # 'chat', 'document_summary', 'custom'
        sa.Column("output_format", sa.String(length=20), nullable=False),  # 'docx', 'pdf'
        # Source reference
        sa.Column("source_type", sa.String(length=50), nullable=False),  # 'chat_session', 'document', 'llm_content'
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=True),
        # Content for custom exports
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_format", sa.String(length=20), nullable=True, server_default=sa.text("'markdown'")),
        # Metadata
        sa.Column("title", sa.String(length=255), nullable=False),
        # Style
        sa.Column("style", sa.String(length=50), nullable=False, server_default=sa.text("'professional'")),
        sa.Column("custom_theme", postgresql.JSON(), nullable=True),
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
        # Foreign key
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # Create indexes for common queries
    op.create_index("ix_export_jobs_user_id", "export_jobs", ["user_id"])
    op.create_index("ix_export_jobs_status", "export_jobs", ["status"])
    op.create_index("ix_export_jobs_created_at", "export_jobs", ["created_at"])
    op.create_index("ix_export_jobs_source_id", "export_jobs", ["source_id"])


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_export_jobs_source_id", table_name="export_jobs")
    op.drop_index("ix_export_jobs_created_at", table_name="export_jobs")
    op.drop_index("ix_export_jobs_status", table_name="export_jobs")
    op.drop_index("ix_export_jobs_user_id", table_name="export_jobs")
    # Drop table
    op.drop_table("export_jobs")
