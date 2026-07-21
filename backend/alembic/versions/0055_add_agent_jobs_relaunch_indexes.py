"""Add agent_jobs relaunch expression indexes for lineage filters/counts.

Revision ID: 0055_add_agent_jobs_relaunch_indexes
Revises: 0054_normalize_scope_keys_to_source_id
Create Date: 2026-02-26 00:00:00.000000
"""

from __future__ import annotations

from alembic import op


revision = "0055_add_agent_jobs_relaunch_indexes"
down_revision = "0054_normalize_scope_keys_to_source_id"
branch_labels = None
depends_on = None


IDX_RELAUNCH_EXPR = "ix_agent_jobs_relaunch_from_job_id_expr"
IDX_USER_RELAUNCH_EXPR = "ix_agent_jobs_user_relaunch_from_job_id_expr"


def _is_postgres() -> bool:
    bind = op.get_bind()
    return str(getattr(bind.dialect, "name", "") or "").lower() == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return

    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {IDX_RELAUNCH_EXPR}
        ON agent_jobs ((config->>'relaunch_from_job_id'))
        WHERE (config->>'relaunch_from_job_id') IS NOT NULL
          AND (config->>'relaunch_from_job_id') <> ''
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {IDX_USER_RELAUNCH_EXPR}
        ON agent_jobs (user_id, (config->>'relaunch_from_job_id'))
        WHERE (config->>'relaunch_from_job_id') IS NOT NULL
          AND (config->>'relaunch_from_job_id') <> ''
        """
    )


def downgrade() -> None:
    if not _is_postgres():
        return

    op.execute(f"DROP INDEX IF EXISTS {IDX_USER_RELAUNCH_EXPR}")
    op.execute(f"DROP INDEX IF EXISTS {IDX_RELAUNCH_EXPR}")

