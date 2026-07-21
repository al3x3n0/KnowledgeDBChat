"""add decision trace assignment and escalation fields

Revision ID: 0072_add_decision_trace_assignment_and_escalation
Revises: 0071_add_decision_trace_triage_and_views
Create Date: 2026-04-06 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0072_add_decision_trace_assignment_and_escalation"
down_revision = "0071_add_decision_trace_triage_and_views"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "autonomy_decision_events",
        sa.Column("assigned_to_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("assigned_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("team_bucket", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("escalation_state", sa.String(length=24), nullable=False, server_default="none"),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("escalation_reason", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "autonomy_decision_events",
        sa.Column("escalated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_foreign_key(
        "fk_autonomy_decision_events_assigned_to_user_id",
        "autonomy_decision_events",
        "users",
        ["assigned_to_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_autonomy_decision_events_assigned_by_user_id",
        "autonomy_decision_events",
        "users",
        ["assigned_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_autonomy_decision_events_assigned_to_user_id",
        "autonomy_decision_events",
        ["assigned_to_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_autonomy_decision_events_team_bucket",
        "autonomy_decision_events",
        ["team_bucket"],
        unique=False,
    )
    op.create_index(
        "ix_autonomy_decision_events_due_at",
        "autonomy_decision_events",
        ["due_at"],
        unique=False,
    )
    op.create_index(
        "ix_autonomy_decision_events_escalation_state",
        "autonomy_decision_events",
        ["escalation_state"],
        unique=False,
    )

    op.execute("UPDATE autonomy_decision_events SET escalation_state = 'none' WHERE escalation_state IS NULL")
    op.alter_column("autonomy_decision_events", "escalation_state", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_autonomy_decision_events_escalation_state", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_due_at", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_team_bucket", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_assigned_to_user_id", table_name="autonomy_decision_events")
    op.drop_constraint("fk_autonomy_decision_events_assigned_by_user_id", "autonomy_decision_events", type_="foreignkey")
    op.drop_constraint("fk_autonomy_decision_events_assigned_to_user_id", "autonomy_decision_events", type_="foreignkey")
    op.drop_column("autonomy_decision_events", "escalated_at")
    op.drop_column("autonomy_decision_events", "escalation_reason")
    op.drop_column("autonomy_decision_events", "escalation_state")
    op.drop_column("autonomy_decision_events", "due_at")
    op.drop_column("autonomy_decision_events", "team_bucket")
    op.drop_column("autonomy_decision_events", "assigned_by_user_id")
    op.drop_column("autonomy_decision_events", "assigned_at")
    op.drop_column("autonomy_decision_events", "assigned_to_user_id")
