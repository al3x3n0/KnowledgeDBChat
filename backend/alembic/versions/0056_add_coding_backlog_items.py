"""Add coding backlog items table.

Revision ID: 0056_add_coding_backlog_items
Revises: 0055_add_agent_jobs_relaunch_indexes
Create Date: 2026-03-23 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0056_add_coding_backlog_items"
down_revision = "0055a_create_tables_missing_from_history"
branch_labels = None
depends_on = None


def _uuid_type():
    bind = op.get_bind()
    if str(getattr(bind.dialect, "name", "") or "").lower() == "postgresql":
        return postgresql.UUID(as_uuid=True)
    return sa.String(length=36)


def upgrade() -> None:
    uuid_type = _uuid_type()
    op.create_table(
        "coding_backlog_items",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("user_id", uuid_type, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_id", uuid_type, sa.ForeignKey("document_sources.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("portfolio_goal", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="draft"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("scope", sa.String(length=32), nullable=True),
        sa.Column("failure_symptom", sa.Text(), nullable=True),
        sa.Column("error_output", sa.Text(), nullable=True),
        sa.Column("file_paths", sa.JSON(), nullable=True),
        sa.Column("commands", sa.JSON(), nullable=True),
        sa.Column("auto_apply_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("require_patch_pr", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("policy", sa.JSON(), nullable=True),
        sa.Column("decomposition", sa.JSON(), nullable=True),
        sa.Column("child_job_ids", sa.JSON(), nullable=True),
        sa.Column("latest_summary", sa.JSON(), nullable=True),
        sa.Column("orchestrator_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("current_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("latest_apply_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("latest_proposal_id", uuid_type, sa.ForeignKey("code_patch_proposals.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_coding_backlog_items_user_id", "coding_backlog_items", ["user_id"])
    op.create_index("ix_coding_backlog_items_source_id", "coding_backlog_items", ["source_id"])
    op.create_index("ix_coding_backlog_items_status", "coding_backlog_items", ["status"])
    op.create_index("ix_coding_backlog_items_priority", "coding_backlog_items", ["priority"])
    op.create_index("ix_coding_backlog_items_orchestrator_job_id", "coding_backlog_items", ["orchestrator_job_id"])
    op.create_index("ix_coding_backlog_items_current_job_id", "coding_backlog_items", ["current_job_id"])
    op.create_index("ix_coding_backlog_items_latest_apply_job_id", "coding_backlog_items", ["latest_apply_job_id"])
    op.create_index("ix_coding_backlog_items_latest_proposal_id", "coding_backlog_items", ["latest_proposal_id"])


def downgrade() -> None:
    op.drop_index("ix_coding_backlog_items_latest_proposal_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_latest_apply_job_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_current_job_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_orchestrator_job_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_priority", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_status", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_source_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_user_id", table_name="coding_backlog_items")
    op.drop_table("coding_backlog_items")
