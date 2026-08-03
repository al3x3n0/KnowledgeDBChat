"""add persisted autonomous R&D evaluation runs

Revision ID: 0080_add_autonomous_rnd_eval_runs
Revises: 0079_add_external_call_response_correlation
Create Date: 2026-08-03 00:00:00.000000
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0080_add_autonomous_rnd_eval_runs"
down_revision = "0079_add_external_call_response_correlation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autonomous_rnd_eval_runs",
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
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "is_baseline",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("task_count", sa.Integer(), nullable=False),
        sa.Column("trial_count", sa.Integer(), nullable=False),
        sa.Column("mean_score", sa.Float(), nullable=False),
        sa.Column("pass_at_k", sa.Float(), nullable=False),
        sa.Column("pass_pow_k", sa.Float(), nullable=False),
        sa.Column("report", postgresql.JSONB(), nullable=False),
        sa.Column("task_bindings", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_autonomous_rnd_eval_runs_user_id",
        "autonomous_rnd_eval_runs",
        ["user_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_runs_suite_id",
        "autonomous_rnd_eval_runs",
        ["suite_id"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_runs_is_baseline",
        "autonomous_rnd_eval_runs",
        ["is_baseline"],
    )
    op.create_index(
        "ix_autonomous_rnd_eval_runs_created_at",
        "autonomous_rnd_eval_runs",
        ["created_at"],
    )
    op.create_index(
        "ix_rnd_eval_runs_user_suite_created",
        "autonomous_rnd_eval_runs",
        ["user_id", "suite_id", "created_at"],
    )
    # At most one baseline per owner and suite; enforced in the database so a
    # concurrent promotion cannot leave two comparison anchors behind.
    op.create_index(
        "uq_rnd_eval_runs_single_baseline",
        "autonomous_rnd_eval_runs",
        ["user_id", "suite_id"],
        unique=True,
        postgresql_where=sa.text("is_baseline"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_rnd_eval_runs_single_baseline",
        table_name="autonomous_rnd_eval_runs",
    )
    op.drop_index(
        "ix_rnd_eval_runs_user_suite_created",
        table_name="autonomous_rnd_eval_runs",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_runs_created_at",
        table_name="autonomous_rnd_eval_runs",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_runs_is_baseline",
        table_name="autonomous_rnd_eval_runs",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_runs_suite_id",
        table_name="autonomous_rnd_eval_runs",
    )
    op.drop_index(
        "ix_autonomous_rnd_eval_runs_user_id",
        table_name="autonomous_rnd_eval_runs",
    )
    op.drop_table("autonomous_rnd_eval_runs")
