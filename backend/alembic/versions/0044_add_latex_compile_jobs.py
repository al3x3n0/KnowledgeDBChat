"""Add LaTeX Studio compile jobs table.

Revision ID: 0044_add_latex_compile_jobs
Revises: 0043_add_latex_project_files
Create Date: 2026-02-03 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0044_add_latex_compile_jobs"
down_revision = "0043_add_latex_project_files"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "latex_compile_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("latex_projects.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="queued"),
        sa.Column("safe_mode", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("preferred_engine", sa.String(length=50), nullable=True),
        sa.Column("engine", sa.String(length=50), nullable=True),
        sa.Column("log", sa.Text(), nullable=True),
        sa.Column("violations", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("pdf_file_path", sa.String(length=1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_latex_compile_jobs_user_id", "latex_compile_jobs", ["user_id"])
    op.create_index("ix_latex_compile_jobs_project_id", "latex_compile_jobs", ["project_id"])
    op.create_index("ix_latex_compile_jobs_celery_task_id", "latex_compile_jobs", ["celery_task_id"])


def downgrade() -> None:
    op.drop_index("ix_latex_compile_jobs_celery_task_id", table_name="latex_compile_jobs")
    op.drop_index("ix_latex_compile_jobs_project_id", table_name="latex_compile_jobs")
    op.drop_index("ix_latex_compile_jobs_user_id", table_name="latex_compile_jobs")
    op.drop_table("latex_compile_jobs")

