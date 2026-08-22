"""Add agent_method_outcomes: what became of the runs that carried a method

Methods were recorded, recalled and never scored, so one that misleads carried
exactly the authority of one that works. This table scores them the way
agent_predictions scores numbers: one row per method per run, the contract
result and the settled-prediction error attached, aggregated on read.

`cited` separates the two strengths of evidence rather than blurring them.
A method that was merely in a run's context is weak evidence about that run;
a method the run named as what it was building on is stronger. Counting them
together would produce a standing that reads as more than it is.

Revision ID: 0086_add_agent_method_outcomes
Revises: 0085_add_agent_predictions
Create Date: 2026-08-22

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0086_add_agent_method_outcomes"
down_revision = "0085_add_agent_predictions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_method_outcomes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # Not a foreign key on purpose: a method may be deleted or re-recorded,
        # and dropping the history of what happened under it is the wrong
        # repair for that.
        sa.Column("method_memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("method_name", sa.String(length=200), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agent_jobs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("cited", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "contract_enabled", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "contract_satisfied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("unmet_requirements", sa.Text(), nullable=True),
        sa.Column(
            "predictions_settled", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("mean_relative_error", sa.Float(), nullable=True),
        sa.Column("iterations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_agent_method_outcomes_method_memory_id",
        "agent_method_outcomes",
        ["method_memory_id"],
    )
    op.create_index(
        "ix_agent_method_outcomes_user_id", "agent_method_outcomes", ["user_id"]
    )
    op.create_index(
        "ix_agent_method_outcomes_job_id", "agent_method_outcomes", ["job_id"]
    )
    op.create_index(
        "ix_agent_method_outcomes_name", "agent_method_outcomes", ["method_name"]
    )
    op.create_index(
        "ix_agent_method_outcomes_created_at", "agent_method_outcomes", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_method_outcomes_created_at", table_name="agent_method_outcomes"
    )
    op.drop_index("ix_agent_method_outcomes_name", table_name="agent_method_outcomes")
    op.drop_index("ix_agent_method_outcomes_job_id", table_name="agent_method_outcomes")
    op.drop_index(
        "ix_agent_method_outcomes_user_id", table_name="agent_method_outcomes"
    )
    op.drop_index(
        "ix_agent_method_outcomes_method_memory_id", table_name="agent_method_outcomes"
    )
    op.drop_table("agent_method_outcomes")
