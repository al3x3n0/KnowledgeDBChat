"""Add domain research profiles table.

Revision ID: 0057_add_domain_research_profiles
Revises: 0056_add_coding_backlog_items
Create Date: 2026-03-24 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0057_add_domain_research_profiles"
down_revision = "0056_add_coding_backlog_items"
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
        "domain_research_profiles",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("user_id", uuid_type, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("domain", sa.String(length=300), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("customer_context", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="draft"),
        sa.Column("source_scope", sa.String(length=32), nullable=False, server_default="kb_plus_arxiv"),
        sa.Column("monitor_queries", sa.JSON(), nullable=True),
        sa.Column("report_format", sa.String(length=32), nullable=False, server_default="brief_and_report"),
        sa.Column("interval_minutes", sa.Integer(), nullable=False, server_default="1440"),
        sa.Column("persist_artifacts", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("auto_launch_follow_up", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("auto_create_experiment_plans", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("confidence_threshold", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("max_documents", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("max_papers", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("latest_summary", sa.JSON(), nullable=True),
        sa.Column("latest_note_ids", sa.JSON(), nullable=True),
        sa.Column("latest_experiment_plan_ids", sa.JSON(), nullable=True),
        sa.Column("latest_run_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("active_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_domain_research_profiles_user_id", "domain_research_profiles", ["user_id"])
    op.create_index("ix_domain_research_profiles_status", "domain_research_profiles", ["status"])
    op.create_index("ix_domain_research_profiles_latest_run_job_id", "domain_research_profiles", ["latest_run_job_id"])
    op.create_index("ix_domain_research_profiles_active_job_id", "domain_research_profiles", ["active_job_id"])


def downgrade() -> None:
    op.drop_index("ix_domain_research_profiles_active_job_id", table_name="domain_research_profiles")
    op.drop_index("ix_domain_research_profiles_latest_run_job_id", table_name="domain_research_profiles")
    op.drop_index("ix_domain_research_profiles_status", table_name="domain_research_profiles")
    op.drop_index("ix_domain_research_profiles_user_id", table_name="domain_research_profiles")
    op.drop_table("domain_research_profiles")
