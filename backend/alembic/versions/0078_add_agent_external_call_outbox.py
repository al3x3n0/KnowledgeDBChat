"""add agent external call outbox

Revision ID: 0078_add_agent_external_call_outbox
Revises: 0077_add_agent_execution_leases
Create Date: 2026-07-29 00:30:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0078_add_agent_external_call_outbox"
down_revision = "0077_add_agent_execution_leases"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_external_call_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tool_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_tools.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("capability", sa.String(length=160), nullable=False),
        sa.Column("payload", postgresql.JSON(), nullable=False),
        sa.Column("request_id", sa.String(length=200), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="5", nullable=False),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("claim_owner", sa.String(length=200), nullable=True),
        sa.Column("claim_token", sa.String(length=64), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response", postgresql.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_agent_external_call_outbox_idempotency_key",
        ),
        sa.UniqueConstraint(
            "request_id",
            name="uq_agent_external_call_outbox_request_id",
        ),
    )
    op.create_index(
        "ix_agent_external_call_outbox_due",
        "agent_external_call_outbox",
        ["status", "next_attempt_at"],
    )
    op.create_index(
        "ix_agent_external_call_outbox_job_id",
        "agent_external_call_outbox",
        ["job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_external_call_outbox_job_id",
        table_name="agent_external_call_outbox",
    )
    op.drop_index(
        "ix_agent_external_call_outbox_due",
        table_name="agent_external_call_outbox",
    )
    op.drop_table("agent_external_call_outbox")
