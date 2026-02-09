"""Add LaTeX Studio projects table.

Revision ID: 0042_add_latex_projects
Revises: 0041_enable_url_ingest_and_web_scrape_tools_for_document_expert
Create Date: 2026-02-03 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0042_add_latex_projects"
down_revision = "0041_enable_url_ingest_and_web_scrape_tools_for_document_expert"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "latex_projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False, server_default="Untitled LaTeX Project"),
        sa.Column("tex_source", sa.Text(), nullable=False, server_default=""),
        sa.Column("tex_file_path", sa.String(length=1000), nullable=True),
        sa.Column("pdf_file_path", sa.String(length=1000), nullable=True),
        sa.Column("last_compile_engine", sa.String(length=50), nullable=True),
        sa.Column("last_compile_log", sa.Text(), nullable=True),
        sa.Column("last_compiled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_latex_projects_user_id", "latex_projects", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_latex_projects_user_id", table_name="latex_projects")
    op.drop_table("latex_projects")

