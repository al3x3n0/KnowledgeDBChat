"""Add LaTeX Studio project files table.

Revision ID: 0043_add_latex_project_files
Revises: 0042_add_latex_projects
Create Date: 2026-02-03 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0043_add_latex_project_files"
down_revision = "0042_add_latex_projects"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "latex_project_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("latex_projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("file_path", sa.String(length=1000), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_latex_project_files_project_id", "latex_project_files", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_latex_project_files_project_id", table_name="latex_project_files")
    op.drop_table("latex_project_files")

