"""add synthesis job paper ids

Revision ID: 0060_add_synthesis_job_paper_ids
Revises: 0059_add_domain_research_memo_fields
Create Date: 2026-03-27 14:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0060_add_synthesis_job_paper_ids"
down_revision = "0059_add_domain_research_memo_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "synthesis_jobs",
        sa.Column("paper_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.alter_column("synthesis_jobs", "paper_ids", server_default=None)


def downgrade() -> None:
    op.drop_column("synthesis_jobs", "paper_ids")
