"""A column for who a swarm job is shared with.

Sharing lives in `results["swarm_collaboration"]`, and finding the jobs shared
with a user meant casting every job's `results` to text and pattern-matching
it -- the whole table, on every request for the job list. `collaborator_index`
holds the assignee and the people it is shared with as "|id|id|", kept by
mapper events on `AgentJob`, and a partial index holds only the jobs that have
a collaboration block at all.

Revision ID: 0107_job_collaborator_index
Revises: 0106_manual_kg_relationships
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "0107_job_collaborator_index"
down_revision = "0106_manual_kg_relationships"
branch_labels = None
depends_on = None


def _index_for(results):
    # A copy of app.models.agent_job.collaborator_index_for as it was when this
    # migration was written; a migration must not change when the model does.
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except ValueError:
            return None
    block = results.get("swarm_collaboration") if isinstance(results, dict) else None
    if not isinstance(block, dict):
        return None
    shared = block.get("shared_with_user_ids")
    names = [block.get("assigned_user_id")]
    names.extend(shared[:200] if isinstance(shared, list) else [])
    seen = []
    for raw in names:
        name = str(raw or "").strip().lower()
        if name and "|" not in name and name not in seen:
            seen.append(name)
    return "|" + "".join(f"{name}|" for name in seen)


def upgrade() -> None:
    op.add_column(
        "agent_jobs", sa.Column("collaborator_index", sa.Text(), nullable=True)
    )

    jobs = sa.table(
        "agent_jobs",
        sa.column("id"),
        sa.column("results", sa.JSON),
        sa.column("collaborator_index", sa.Text),
    )
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(jobs.c.id, jobs.c.results).where(
            sa.cast(jobs.c.results, sa.Text).like("%swarm_collaboration%")
        )
    ).fetchall()
    for job_id, results in rows:
        value = _index_for(results)
        if value is not None:
            bind.execute(
                jobs.update()
                .where(jobs.c.id == job_id)
                .values(collaborator_index=value)
            )

    op.create_index(
        "ix_agent_jobs_has_collaborators",
        "agent_jobs",
        ["user_id"],
        postgresql_where=sa.text("collaborator_index IS NOT NULL"),
        sqlite_where=sa.text("collaborator_index IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_agent_jobs_has_collaborators", table_name="agent_jobs")
    op.drop_column("agent_jobs", "collaborator_index")
