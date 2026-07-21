"""add synthesis job research note id

Revision ID: 0066_add_synthesis_job_research_note_id
Revises: 0065_add_coding_backlog_collaboration, 0060_add_synthesis_job_paper_ids
Create Date: 2026-03-27 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0066_add_synthesis_job_research_note_id"
down_revision = ("0065_add_coding_backlog_collaboration", "0060_add_synthesis_job_paper_ids")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("synthesis_jobs", sa.Column("research_note_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_synthesis_jobs_research_note_id",
        "synthesis_jobs",
        "research_notes",
        ["research_note_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_synthesis_jobs_research_note_id", "synthesis_jobs", type_="foreignkey")
    op.drop_column("synthesis_jobs", "research_note_id")
