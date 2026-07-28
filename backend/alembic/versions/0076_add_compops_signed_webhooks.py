"""add CompOps signed webhooks

Revision ID: 0076_add_compops_signed_webhooks
Revises: 0075_add_compops_evidence_subscriptions
Create Date: 2026-07-28 00:00:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0076_add_compops_signed_webhooks"
down_revision = "0075_add_compops_evidence_subscriptions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "compops_evidence_subscriptions",
        sa.Column(
            "webhook_secret_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "compops_evidence_subscriptions",
        sa.Column(
            "webhook_enabled",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        "compops_evidence_subscriptions",
        sa.Column("last_webhook_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "compops_evidence_subscriptions",
        sa.Column("last_webhook_event_id", sa.String(length=200), nullable=True),
    )
    op.create_foreign_key(
        "fk_compops_subscription_webhook_secret",
        "compops_evidence_subscriptions",
        "user_secrets",
        ["webhook_secret_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_table(
        "compops_webhook_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "subscription_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "compops_evidence_subscriptions.id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("event_id", sa.String(length=200), nullable=False),
        sa.Column("event_type", sa.String(length=120), nullable=True),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("evidence_changed", sa.Boolean(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "subscription_id",
            "event_id",
            name="uq_compops_webhook_event",
        ),
    )
    op.create_index(
        "ix_compops_webhook_events_subscription_id",
        "compops_webhook_events",
        ["subscription_id"],
    )
    op.create_index(
        "ix_compops_webhook_events_status_received",
        "compops_webhook_events",
        ["status", "received_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compops_webhook_events_status_received",
        table_name="compops_webhook_events",
    )
    op.drop_index(
        "ix_compops_webhook_events_subscription_id",
        table_name="compops_webhook_events",
    )
    op.drop_table("compops_webhook_events")
    op.drop_constraint(
        "fk_compops_subscription_webhook_secret",
        "compops_evidence_subscriptions",
        type_="foreignkey",
    )
    for column in (
        "last_webhook_event_id",
        "last_webhook_at",
        "webhook_enabled",
        "webhook_secret_id",
    ):
        op.drop_column("compops_evidence_subscriptions", column)
