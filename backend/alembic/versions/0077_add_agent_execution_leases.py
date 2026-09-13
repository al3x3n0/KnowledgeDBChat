"""add agent execution leases

Revision ID: 0077_add_agent_execution_leases
Revises: 0076_add_compops_signed_webhooks
Create Date: 2026-07-29 00:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "0077_add_agent_execution_leases"
down_revision = "0076_add_compops_signed_webhooks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_jobs",
        sa.Column("execution_lease_owner", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "agent_jobs",
        sa.Column("execution_lease_token", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agent_jobs",
        sa.Column(
            "execution_lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_jobs",
        sa.Column(
            "execution_lease_heartbeat_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_jobs",
        sa.Column(
            "execution_fence",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_agent_jobs_execution_lease_expires_at",
        "agent_jobs",
        ["execution_lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_jobs_execution_lease_expires_at",
        table_name="agent_jobs",
    )
    for column in (
        "execution_fence",
        "execution_lease_heartbeat_at",
        "execution_lease_expires_at",
        "execution_lease_token",
        "execution_lease_owner",
    ):
        op.drop_column("agent_jobs", column)
