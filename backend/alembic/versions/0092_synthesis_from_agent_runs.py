"""Let a synthesis take its sources from agent runs.

A run records measurements; a document is written from them. Until now the two
had no way to reference each other, so the numbers were retyped by hand into
the synthesis — which is the exact step where a measured value becomes a
remembered one.

Revision ID: 0092_synthesis_from_agent_runs
Revises: 0091_add_llm_snapshot_reasoning
Create Date: 2026-09-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0092_synthesis_from_agent_runs"
down_revision = "0091_add_llm_snapshot_reasoning"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A list, like document_ids and paper_ids beside it: one synthesis may draw
    # on several runs, and a run may feed several documents. Not a foreign key
    # for the same reason those are not — the column holds ids, and a deleted
    # run should leave the document standing with a source it can name.
    op.add_column(
        "synthesis_jobs",
        sa.Column(
            "agent_job_ids",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )


def downgrade() -> None:
    op.drop_column("synthesis_jobs", "agent_job_ids")
