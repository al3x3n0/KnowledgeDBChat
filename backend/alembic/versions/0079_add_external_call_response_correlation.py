"""add external call response correlation

Revision ID: 0079_add_external_call_response_correlation
Revises: 0078_add_agent_external_call_outbox
Create Date: 2026-07-29 01:00:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0079_add_external_call_response_correlation"
down_revision = "0078_add_agent_external_call_outbox"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_external_call_outbox",
        sa.Column("correlation", postgresql.JSON(), nullable=True),
    )
    op.add_column(
        "agent_external_call_outbox",
        sa.Column("correlated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "agent_external_call_outbox",
        sa.Column("resume_claim_owner", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "agent_external_call_outbox",
        sa.Column("resume_claim_token", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agent_external_call_outbox",
        sa.Column(
            "resume_claim_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_external_call_outbox",
        sa.Column("resume_enqueued_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_agent_external_call_outbox_resume_due",
        "agent_external_call_outbox",
        ["status", "resume_enqueued_at", "resume_claim_expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_external_call_outbox_resume_due",
        table_name="agent_external_call_outbox",
    )
    for column in (
        "resume_enqueued_at",
        "resume_claim_expires_at",
        "resume_claim_token",
        "resume_claim_owner",
        "correlated_at",
        "correlation",
    ):
        op.drop_column("agent_external_call_outbox", column)
