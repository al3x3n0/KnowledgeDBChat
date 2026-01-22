"""Add template jobs table.

Revision ID: 0012_add_template_jobs
Revises: 0011_add_persona_edit_requests
Create Date: 2024-06-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0012_add_template_jobs"
down_revision = "0011_add_persona_edit_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "template_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Template file info
        sa.Column("template_file_path", sa.String(length=1000), nullable=False),
        sa.Column("template_filename", sa.String(length=500), nullable=False),
        # Detected sections
        sa.Column("sections", postgresql.JSON(), nullable=True),
        # Source documents
        sa.Column("source_document_ids", postgresql.JSON(), nullable=False),
        # Processing status
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("progress", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("current_section", sa.String(length=500), nullable=True),
        # Result
        sa.Column("filled_file_path", sa.String(length=1000), nullable=True),
        sa.Column("filled_filename", sa.String(length=500), nullable=True),
        # Error tracking
        sa.Column("error_message", sa.Text(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_template_jobs_user_id", "template_jobs", ["user_id"])
    op.create_index("ix_template_jobs_status", "template_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_template_jobs_status", table_name="template_jobs")
    op.drop_index("ix_template_jobs_user_id", table_name="template_jobs")
    op.drop_table("template_jobs")
