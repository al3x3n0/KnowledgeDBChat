"""A column for whether a job may belong in the operator queue.

Two of the reasons a job becomes a queue row are inside JSON: an approval
checkpoint in `results`, and a completed-with-contract-unmet entry in
`execution_log`. Neither can be asked of the database cheaply, so the queue
loaded every job a user had ever run, whole, on each call. Two flags carry
them, each kept by mapper events on `AgentJob` from its own source; a partial
index holds only the jobs where one is set.

Revision ID: 0108_job_operator_flags
Revises: 0107_job_collaborator_index
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "0108_job_operator_flags"
down_revision = "0107_job_collaborator_index"
branch_labels = None
depends_on = None


def _decoded(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def _flags(results, execution_log):
    # A copy of the model's rule as it was when this migration was written.
    results, execution_log = _decoded(results), _decoded(execution_log)
    checkpoint = False
    if isinstance(results, dict):
        direct = results.get("approval_checkpoint")
        execution = results.get("execution_strategy")
        approvals = (
            execution.get("approval_checkpoints")
            if isinstance(execution, dict)
            else None
        )
        pending = approvals.get("pending") if isinstance(approvals, dict) else None
        checkpoint = bool(
            (isinstance(direct, dict) and direct)
            or (isinstance(pending, dict) and pending)
        )
    unmet = isinstance(execution_log, list) and any(
        isinstance(entry, dict) and entry.get("phase") == "completed_contract_unmet"
        for entry in execution_log
    )
    return checkpoint, unmet


def upgrade() -> None:
    for name in ("has_approval_checkpoint", "has_contract_unmet_entry"):
        op.add_column(
            "agent_jobs",
            sa.Column(name, sa.Boolean(), nullable=False, server_default=sa.false()),
        )

    jobs = sa.table(
        "agent_jobs",
        sa.column("id"),
        sa.column("results", sa.JSON),
        sa.column("execution_log", sa.JSON),
        sa.column("has_approval_checkpoint", sa.Boolean),
        sa.column("has_contract_unmet_entry", sa.Boolean),
    )
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(jobs.c.id, jobs.c.results, jobs.c.execution_log).where(
            sa.or_(
                sa.cast(jobs.c.results, sa.Text).like("%approval_checkpoint%"),
                sa.cast(jobs.c.execution_log, sa.Text).like(
                    "%completed_contract_unmet%"
                ),
            )
        )
    ).fetchall()
    for job_id, results, execution_log in rows:
        checkpoint, unmet = _flags(results, execution_log)
        if checkpoint or unmet:
            bind.execute(
                jobs.update()
                .where(jobs.c.id == job_id)
                .values(
                    has_approval_checkpoint=checkpoint,
                    has_contract_unmet_entry=unmet,
                )
            )

    op.create_index(
        "ix_agent_jobs_wants_operator",
        "agent_jobs",
        ["user_id"],
        postgresql_where=sa.text("has_approval_checkpoint OR has_contract_unmet_entry"),
        sqlite_where=sa.text("has_approval_checkpoint OR has_contract_unmet_entry"),
    )


def downgrade() -> None:
    op.drop_index("ix_agent_jobs_wants_operator", table_name="agent_jobs")
    op.drop_column("agent_jobs", "has_contract_unmet_entry")
    op.drop_column("agent_jobs", "has_approval_checkpoint")
