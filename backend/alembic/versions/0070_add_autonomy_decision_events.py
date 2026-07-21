"""add autonomy decision events

Revision ID: 0070_add_autonomy_decision_events
Revises: 0069_add_domain_research_profile_automation_fields
Create Date: 2026-04-02 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0070_add_autonomy_decision_events"
down_revision = "0069_add_domain_research_profile_automation_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autonomy_decision_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=80), nullable=False),
        sa.Column("source_kind", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=128), nullable=True),
        sa.Column("source_label", sa.String(length=255), nullable=True),
        sa.Column("customer", sa.String(length=255), nullable=True),
        sa.Column("decision_type", sa.String(length=80), nullable=False),
        sa.Column("reason_code", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=True),
        sa.Column("severity", sa.String(length=32), nullable=True),
        sa.Column("actor_mode", sa.String(length=24), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("operator_note", sa.Text(), nullable=True),
        sa.Column("before_state", sa.JSON(), nullable=True),
        sa.Column("after_state", sa.JSON(), nullable=True),
        sa.Column("deep_link", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("is_derived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("record_origin", sa.String(length=24), nullable=False, server_default="persisted"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_autonomy_decision_events_user_id", "autonomy_decision_events", ["user_id"])
    op.create_index("ix_autonomy_decision_events_event_time", "autonomy_decision_events", ["event_time"])
    op.create_index("ix_autonomy_decision_events_source_kind", "autonomy_decision_events", ["source_kind"])
    op.create_index("ix_autonomy_decision_events_source_id", "autonomy_decision_events", ["source_id"])
    op.create_index("ix_autonomy_decision_events_customer", "autonomy_decision_events", ["customer"])
    op.create_index("ix_autonomy_decision_events_decision_type", "autonomy_decision_events", ["decision_type"])
    op.create_index("ix_autonomy_decision_events_reason_code", "autonomy_decision_events", ["reason_code"])
    op.create_index("ix_autonomy_decision_events_status", "autonomy_decision_events", ["status"])
    op.create_index("ix_autonomy_decision_events_severity", "autonomy_decision_events", ["severity"])
    op.create_index("ix_autonomy_decision_events_actor_mode", "autonomy_decision_events", ["actor_mode"])
    op.alter_column("autonomy_decision_events", "is_derived", server_default=None)
    op.alter_column("autonomy_decision_events", "record_origin", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_autonomy_decision_events_actor_mode", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_severity", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_status", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_reason_code", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_decision_type", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_customer", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_source_id", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_source_kind", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_event_time", table_name="autonomy_decision_events")
    op.drop_index("ix_autonomy_decision_events_user_id", table_name="autonomy_decision_events")
    op.drop_table("autonomy_decision_events")

