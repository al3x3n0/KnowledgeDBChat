"""Saved pipeline specs.

One table. The spec is stored as JSON exactly as authored — see the model's
docstring for why it is not decomposed into stage rows — and everything else
here is about the saved thing rather than the pipeline: whose it is, what it is
called, and what it has launched.

Revision ID: 0094_agent_pipelines
Revises: 0093_document_folders
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0094_agent_pipelines"
down_revision = "0093_document_folders"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_pipelines",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("last_check_valid", sa.String(length=16), nullable=True),
        sa.Column("last_estimated_seconds", sa.Integer(), nullable=True),
        sa.Column("launch_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_launched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # SET NULL rather than CASCADE: deleting the run a pipeline produced
        # must not delete the pipeline that produced it.
        sa.ForeignKeyConstraint(
            ["last_job_id"], ["agent_jobs.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint("user_id", "name", name="uq_agent_pipeline_user_name"),
    )
    op.create_index(
        "ix_agent_pipelines_user_id", "agent_pipelines", ["user_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_agent_pipelines_user_id", "agent_pipelines")
    op.drop_table("agent_pipelines")
