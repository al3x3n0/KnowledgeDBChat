"""add llm call snapshots table

Revision ID: 0073_add_llm_call_snapshots
Revises: 0072_add_decision_trace_assignment_and_escalation
Create Date: 2026-07-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0073_add_llm_call_snapshots"
down_revision = "0072_add_decision_trace_assignment_and_escalation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_call_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
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
        sa.Column("iteration", sa.Integer(), nullable=True),
        sa.Column("phase", sa.String(length=50), nullable=True),
        sa.Column("provider", sa.String(length=50), nullable=True),
        sa.Column("model", sa.String(length=200), nullable=True),
        sa.Column("task_type", sa.String(length=50), nullable=True),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("structured", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_llm_call_snapshots_user_id", "llm_call_snapshots", ["user_id"])
    op.create_index("ix_llm_call_snapshots_job_id", "llm_call_snapshots", ["job_id"])
    op.create_index("ix_llm_call_snapshots_created_at", "llm_call_snapshots", ["created_at"])
    op.create_index(
        "ix_llm_call_snapshots_job_created", "llm_call_snapshots", ["job_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_llm_call_snapshots_job_created", table_name="llm_call_snapshots")
    op.drop_index("ix_llm_call_snapshots_created_at", table_name="llm_call_snapshots")
    op.drop_index("ix_llm_call_snapshots_job_id", table_name="llm_call_snapshots")
    op.drop_index("ix_llm_call_snapshots_user_id", table_name="llm_call_snapshots")
    op.drop_table("llm_call_snapshots")
