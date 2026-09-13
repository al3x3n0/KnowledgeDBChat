"""Make a coding workspace addressable from any process.

A workspace was a `tempfile.mkdtemp()` directory in a process-local dict inside
the celery worker that ran the job. The API could not see it, it did not
survive the worker, and it was deleted on completion with only changed files
pushed to object storage.

For a research pipeline that last part is the problem: the environment is part
of the evidence, and a benchmark whose workspace no longer exists is a number
nobody can re-derive.

This table holds the identity and shape of a workspace; the files live on the
volume both the API and the workers already mount at /app/data. The row
deliberately outlives the directory -- `status='discarded'` says a workspace
existed and its files are gone, which is different information from never
having heard of it.

Revision ID: 0096_coding_workspaces
Revises: 0095_llm_cache_tokens
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0096_coding_workspaces"
down_revision = "0095_llm_cache_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "coding_workspaces",
        # The id the agent's tools already use, not a new one: the run state,
        # the tool parameters and the job results all name this already.
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # No foreign key: deleting the run should orphan the workspace record,
        # not destroy the account of how a measurement was produced.
        sa.Column("owner_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("base_path", sa.Text(), nullable=False),
        sa.Column("source_id", sa.String(length=200), nullable=True),
        sa.Column("repo_url", sa.Text(), nullable=True),
        sa.Column("branch", sa.String(length=200), nullable=True),
        sa.Column("state", sa.JSON(), nullable=True),
        sa.Column(
            "status", sa.String(length=24), nullable=False, server_default="active"
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_coding_workspaces_user_id", "coding_workspaces", ["user_id"])
    op.create_index(
        "ix_coding_workspaces_owner_job_id", "coding_workspaces", ["owner_job_id"]
    )
    op.create_index("ix_coding_workspaces_status", "coding_workspaces", ["status"])
    op.create_index(
        "ix_coding_workspaces_user_status", "coding_workspaces", ["user_id", "status"]
    )


def downgrade() -> None:
    op.drop_index("ix_coding_workspaces_user_status", table_name="coding_workspaces")
    op.drop_index("ix_coding_workspaces_status", table_name="coding_workspaces")
    op.drop_index("ix_coding_workspaces_owner_job_id", table_name="coding_workspaces")
    op.drop_index("ix_coding_workspaces_user_id", table_name="coding_workspaces")
    op.drop_table("coding_workspaces")
