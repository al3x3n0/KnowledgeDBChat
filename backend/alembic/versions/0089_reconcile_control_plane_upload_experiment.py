"""create the last two tables and three columns the models need and migrations never made

The same class of drift 0055a and 0082 were written to repair, found by the
check those migrations exist to make possible. `agent_control_plane_views` and
`upload_sessions` are declared in the models and appear nowhere in the history,
and `experiment_runs` is missing `parent_run_id`, `latest_child_run_id` and
`retry_count`. On a database built only from migrations -- which is what CI
builds and what a new deployment gets -- the control-plane view presets and
chunked upload resume would fail at the first query, and experiment retries and
run lineage with them.

Idempotent in the style of 0082, and for the same reason: databases that
predate the removal of `create_all` already have these objects, and this must
complete on a fresh database while changing nothing on those. Hence CREATE
TABLE / ADD COLUMN / CREATE INDEX ... IF NOT EXISTS throughout.

Only the missing objects are addressed. The drift check also reports index,
nullability and type differences on other tables; those are declaration
mismatches rather than absences, the application runs against them, and folding
them into a migration about missing objects would make both harder to review.

Revision ID: 0089_reconcile_control_plane_upload_experiment
Revises: 0088_add_campaign_item_lineage
Create Date: 2026-08-23

"""

from alembic import op

revision = "0089_reconcile_control_plane_upload_experiment"
down_revision = "0088_add_campaign_item_lineage"
branch_labels = None
depends_on = None


MISSING_TABLES = [
    """
CREATE TABLE IF NOT EXISTS agent_control_plane_views (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    filters JSON,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
)
    """,
    """
CREATE TABLE IF NOT EXISTS upload_sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    filename VARCHAR(500) NOT NULL,
    file_size INTEGER NOT NULL,
    file_type VARCHAR(50),
    content_type VARCHAR(100),
    chunk_size INTEGER NOT NULL DEFAULT 5242880,
    total_chunks INTEGER NOT NULL,
    uploaded_chunks JSON NOT NULL DEFAULT '[]',
    uploaded_bytes INTEGER NOT NULL DEFAULT 0,
    minio_upload_id VARCHAR(200),
    minio_part_etags JSON,
    title VARCHAR(500),
    tags JSON,
    extra_metadata JSON,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message VARCHAR(1000),
    document_id UUID REFERENCES documents(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    expires_at TIMESTAMP WITH TIME ZONE
)
    """,
]

# Self-referencing, so a run knows the run it was retried from and the newest
# retry of itself. ON DELETE SET NULL: deleting a run must not cascade away the
# other runs of the same experiment.
MISSING_COLUMNS = [
    """
ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS parent_run_id UUID
    REFERENCES experiment_runs(id) ON DELETE SET NULL
    """,
    """
ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS latest_child_run_id UUID
    REFERENCES experiment_runs(id) ON DELETE SET NULL
    """,
    """
ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0
    """,
]

MISSING_INDEXES = [
    """
CREATE INDEX IF NOT EXISTS ix_agent_control_plane_views_user_id
    ON agent_control_plane_views(user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_upload_sessions_user_id ON upload_sessions(user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_parent_run_id
    ON experiment_runs(parent_run_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_latest_child_run_id
    ON experiment_runs(latest_child_run_id)
    """,
]


def upgrade() -> None:
    for statement in MISSING_TABLES + MISSING_COLUMNS + MISSING_INDEXES:
        op.execute(statement)


def downgrade() -> None:
    # Intentionally a no-op, as in 0055a and 0082. These objects predate this
    # migration on any database built by the old create_all path, so dropping
    # them on the way down would destroy data this migration never created.
    pass
