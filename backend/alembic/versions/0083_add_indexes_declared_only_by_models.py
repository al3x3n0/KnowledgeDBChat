"""create the indexes the models declare but no migration created

Revision ID: 0083_add_indexes_declared_only_by_models
Revises: 0082_reconcile_schema_with_models
Create Date: 2026-08-05 00:00:00.000000

Found by running every migration against an empty database and diffing the
result against the models. These indexes existed in practice only because
``create_all`` built them at startup, so a database built from migrations alone
was missing them — including ``ix_users_email`` and ``ix_users_username``.

Idempotent, so it is a no-op on databases that already have them.

Two related differences are deliberately NOT handled here, because neither is a
simple omission and both deserve their own review:

- foreign keys that exist with different ``ondelete`` behaviour than the models
  declare; adding another constraint would duplicate rather than correct them
- unique constraints the models declare and the database lacks, which can fail
  outright against existing duplicate rows
"""

from alembic import op

revision = "0083_add_indexes_declared_only_by_models"
down_revision = "0082_reconcile_schema_with_models"
branch_labels = None
depends_on = None

MISSING_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_agent_definitions_name ON agent_definitions (name)",
    "CREATE INDEX IF NOT EXISTS ix_autonomy_decision_events_event_type ON autonomy_decision_events (event_type)",
    "CREATE INDEX IF NOT EXISTS ix_coding_swarm_profiles_latest_job_id ON coding_swarm_profiles (latest_job_id)",
    "CREATE INDEX IF NOT EXISTS ix_document_persona_detections_document_id ON document_persona_detections (document_id)",
    "CREATE INDEX IF NOT EXISTS ix_document_persona_detections_persona_id ON document_persona_detections (persona_id)",
    "CREATE INDEX IF NOT EXISTS ix_document_source_sync_logs_source_id ON document_source_sync_logs (source_id)",
    "CREATE INDEX IF NOT EXISTS ix_documents_content_hash ON documents (content_hash)",
    "CREATE INDEX IF NOT EXISTS ix_documents_owner_persona_id ON documents (owner_persona_id)",
    "CREATE INDEX IF NOT EXISTS ix_git_branch_diffs_source_id ON git_branch_diffs (source_id)",
    "CREATE INDEX IF NOT EXISTS ix_git_branches_source_id ON git_branches (source_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_notification_preferences_user_id ON notification_preferences (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_persona_edit_requests_persona_id ON persona_edit_requests (persona_id)",
    "CREATE INDEX IF NOT EXISTS ix_scientific_sandbox_profiles_created_by_user_id ON scientific_sandbox_profiles (created_by_user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_search_shares_token ON search_shares (token)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_user_preferences_user_id ON user_preferences (user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)",
]


def upgrade() -> None:
    for statement in MISSING_INDEXES:
        op.execute(statement)


def downgrade() -> None:
    # Intentionally a no-op: these indexes predate this revision on every
    # database built the old way, so dropping them would remove objects this
    # revision did not create.
    pass
