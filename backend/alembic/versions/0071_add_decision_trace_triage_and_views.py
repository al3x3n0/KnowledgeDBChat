"""add decision trace triage fields and saved views

Revision ID: 0071_add_decision_trace_triage_and_views
Revises: 0070_add_autonomy_decision_events
Create Date: 2026-04-02 18:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0071_add_decision_trace_triage_and_views"
down_revision = "0070_add_autonomy_decision_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "autonomy_decision_events",
        sa.Column("triage_status", sa.String(length=24), nullable=False, server_default="new"),
    )
    op.add_column("autonomy_decision_events", sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "autonomy_decision_events",
        sa.Column(
            "acknowledged_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("autonomy_decision_events", sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "autonomy_decision_events",
        sa.Column(
            "resolved_by_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column("autonomy_decision_events", sa.Column("resolution_note", sa.Text(), nullable=True))
    op.add_column(
        "autonomy_decision_events",
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column("autonomy_decision_events", sa.Column("last_viewed_at", sa.DateTime(timezone=True), nullable=True))

    op.create_index(
        "ix_autonomy_decision_events_triage_status",
        "autonomy_decision_events",
        ["triage_status"],
        unique=False,
    )
    op.create_index(
        "ix_autonomy_decision_events_pinned",
        "autonomy_decision_events",
        ["pinned"],
        unique=False,
    )

    op.create_table(
        "autonomy_decision_trace_views",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("filters", sa.JSON(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_autonomy_decision_trace_views_user_id",
        "autonomy_decision_trace_views",
        ["user_id"],
        unique=False,
    )

    op.alter_column("autonomy_decision_events", "triage_status", server_default=None)
    op.alter_column("autonomy_decision_events", "pinned", server_default=None)
    op.alter_column("autonomy_decision_trace_views", "is_default", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_autonomy_decision_trace_views_user_id", table_name="autonomy_decision_trace_views")
    op.drop_table("autonomy_decision_trace_views")

    op.drop_index("ix_autonomy_decision_events_pinned", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_triage_status", table_name="autonomy_decision_events")
    op.drop_column("autonomy_decision_events", "last_viewed_at")
    op.drop_column("autonomy_decision_events", "pinned")
    op.drop_column("autonomy_decision_events", "resolution_note")
    op.drop_column("autonomy_decision_events", "resolved_by_user_id")
    op.drop_column("autonomy_decision_events", "resolved_at")
    op.drop_column("autonomy_decision_events", "acknowledged_by_user_id")
    op.drop_column("autonomy_decision_events", "acknowledged_at")
    op.drop_column("autonomy_decision_events", "triage_status")
