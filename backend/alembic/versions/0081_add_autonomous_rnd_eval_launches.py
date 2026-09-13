"""add autonomous R&D evaluation launches

Revision ID: 0081_add_autonomous_rnd_eval_launches
Revises: 0080_add_autonomous_rnd_eval_runs
Create Date: 2026-08-03 00:30:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0081_add_autonomous_rnd_eval_launches"
down_revision = "0080_add_autonomous_rnd_eval_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autonomous_rnd_eval_launches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("suite_id", sa.String(length=200), nullable=False),
        sa.Column("suite_name", sa.String(length=300), nullable=False),
        sa.Column("suite_version", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("trials_per_task", sa.Integer(), nullable=False),
        sa.Column("job_count", sa.Integer(), nullable=False),
        sa.Column("task_bindings", postgresql.JSONB(), nullable=False),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("autonomous_rnd_eval_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_autonomous_rnd_eval_launches_user_id",
        "autonomous_rnd_eval_launches",
        ["user_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_launches_suite_id",
        "autonomous_rnd_eval_launches",
        ["suite_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_launches_status",
        "autonomous_rnd_eval_launches",
        ["status"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_launches_run_id",
        "autonomous_rnd_eval_launches",
        ["run_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_launches_created_at",
        "autonomous_rnd_eval_launches",
        ["created_at"],
    )
    op.create_index(
        "ix_rnd_eval_launches_user_status_created",
        "autonomous_rnd_eval_launches",
        ["user_id", "status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_rnd_eval_launches_user_status_created",
        table_name="autonomous_rnd_eval_launches",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_launches_created_at",
        table_name="autonomous_rnd_eval_launches",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_launches_run_id",
        table_name="autonomous_rnd_eval_launches",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_launches_status",
        table_name="autonomous_rnd_eval_launches",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_launches_suite_id",
        table_name="autonomous_rnd_eval_launches",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_launches_user_id",
        table_name="autonomous_rnd_eval_launches",
    )
    op.drop_table("autonomous_rnd_eval_launches")
