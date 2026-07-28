"""add CompOps evidence subscriptions

Revision ID: 0075_add_compops_evidence_subscriptions
Revises: 0074_add_rnd_verification_audit_snapshots
Create Date: 2026-07-28 00:00:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0075_add_compops_evidence_subscriptions"
down_revision = "0074_add_rnd_verification_audit_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compops_evidence_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tool_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_tools.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("capability", sa.String(length=120), nullable=False),
        sa.Column("remote_id", sa.String(length=200), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), nullable=False),
        sa.Column("is_enabled", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("last_response_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "last_audit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tool_execution_audits.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
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
            "job_id",
            "tool_id",
            "capability",
            "remote_id",
            name="uq_compops_evidence_subscription_target",
        ),
    )
    for column in ("user_id", "job_id", "tool_id", "next_sync_at"):
        op.create_index(
            f"ix_compops_evidence_subscriptions_{column}",
            "compops_evidence_subscriptions",
            [column],
        )
    op.create_index(
        "ix_compops_evidence_subscriptions_due",
        "compops_evidence_subscriptions",
        ["is_enabled", "next_sync_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compops_evidence_subscriptions_due",
        table_name="compops_evidence_subscriptions",
    )
    for column in ("next_sync_at", "tool_id", "job_id", "user_id"):
        op.drop_index(
            f"ix_compops_evidence_subscriptions_{column}",
            table_name="compops_evidence_subscriptions",
        )
    op.drop_table("compops_evidence_subscriptions")
