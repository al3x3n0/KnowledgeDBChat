"""reconcile the remaining columns and indexes with the models

Revision ID: 0082_reconcile_schema_with_models
Revises: 0081_add_autonomous_rnd_eval_launches
Create Date: 2026-08-04 00:00:00.000000

Companion to 0055a, which created the tables missing from the history. These are
the columns and standalone indexes that also only ever existed because of the
hand-written DDL replayed at startup, on tables migrations did create.

Idempotent, so it completes a fresh database and changes nothing on an existing
one.
"""

from alembic import op

revision = "0082_reconcile_schema_with_models"
down_revision = "0081_add_autonomous_rnd_eval_launches"
branch_labels = None
depends_on = None

MISSING_COLUMNS = [
    """
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS llm_task_providers JSON
    """,
    """
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS paper_algo_default_run_demo_check BOOLEAN NOT NULL DEFAULT FALSE
    """,
    """
ALTER TABLE agent_definitions ADD COLUMN IF NOT EXISTS routing_defaults JSON
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_research_note_citation_issues BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_experiment_run_updates BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_hypothesis_reevaluation_updates BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_queue_urgency_alerts BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_follow_up_outcome_alerts BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_policy_guardrail_alerts BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_autonomy_budget_alerts BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_customer_autonomy_budget_alerts BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_coverage_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.7
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_cooldown_hours INTEGER NOT NULL DEFAULT 12
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS queue_urgency_alert_reminder_cooldown_hours INTEGER NOT NULL DEFAULT 6
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_on_unknown_keys BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_on_low_coverage BOOLEAN NOT NULL DEFAULT TRUE
    """,
    """
ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_on_missing_bibliography BOOLEAN NOT NULL DEFAULT TRUE
    """,
]

MISSING_INDEXES = [
    """
CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_list_user_name ON reading_lists(user_id, name)
    """,
    """
CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_list_item_document_once ON reading_list_items(reading_list_id, document_id)
    """,
]


def upgrade() -> None:
    for statement in MISSING_COLUMNS + MISSING_INDEXES:
        op.execute(statement)


def downgrade() -> None:
    # Intentionally a no-op; see 0055a.
    pass
