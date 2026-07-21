"""Add research portfolios table.

Revision ID: 0058_add_research_portfolios
Revises: 0057_add_domain_research_profiles
Create Date: 2026-03-24 00:10:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0058_add_research_portfolios"
down_revision = "0057_add_domain_research_profiles"
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
        "research_portfolios",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("user_id", uuid_type, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="draft"),
        sa.Column("linked_profile_ids", sa.JSON(), nullable=True),
        sa.Column("automation_policy", sa.JSON(), nullable=True),
        sa.Column("opportunities", sa.JSON(), nullable=True),
        sa.Column("latest_summary", sa.JSON(), nullable=True),
        sa.Column("latest_note_ids", sa.JSON(), nullable=True),
        sa.Column("latest_experiment_plan_ids", sa.JSON(), nullable=True),
        sa.Column("child_job_ids", sa.JSON(), nullable=True),
        sa.Column("active_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("latest_run_job_id", uuid_type, sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_research_portfolios_user_id", "research_portfolios", ["user_id"])
    op.create_index("ix_research_portfolios_status", "research_portfolios", ["status"])
    op.create_index("ix_research_portfolios_active_job_id", "research_portfolios", ["active_job_id"])
    op.create_index("ix_research_portfolios_latest_run_job_id", "research_portfolios", ["latest_run_job_id"])


def downgrade() -> None:
    op.drop_index("ix_research_portfolios_latest_run_job_id", table_name="research_portfolios")
    op.drop_index("ix_research_portfolios_active_job_id", table_name="research_portfolios")
    op.drop_index("ix_research_portfolios_status", table_name="research_portfolios")
    op.drop_index("ix_research_portfolios_user_id", table_name="research_portfolios")
    op.drop_table("research_portfolios")
