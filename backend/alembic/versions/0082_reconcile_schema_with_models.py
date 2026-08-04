"""reconcile migration history with the schema the models require

Revision ID: 0082_reconcile_schema_with_models
Revises: 0081_add_autonomous_rnd_eval_launches
Create Date: 2026-08-04 00:00:00.000000

Until now the working schema came from ``Base.metadata.create_all`` plus a block
of hand-written DDL replayed on every startup, so the migration history could
not build a working database on its own: 12 tables, 17 columns and
2 standalone indexes existed nowhere in it.

This migration folds all of that in. Every statement is idempotent, so it
completes a fresh database and changes nothing on an existing one. After it,
Alembic is the only source of schema truth.
"""

from alembic import op

revision = "0082_reconcile_schema_with_models"
down_revision = "0081_add_autonomous_rnd_eval_launches"
branch_labels = None
depends_on = None

# Compiled from the SQLAlchemy models at this revision and then frozen: a
# migration must describe the schema as of its own revision, never follow the
# models forward.
MISSING_TABLES = [
    """
CREATE TABLE IF NOT EXISTS research_notes (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	title VARCHAR(500) NOT NULL, 
	content_markdown TEXT NOT NULL, 
	source_synthesis_job_id UUID, 
	source_document_ids JSON, 
	tags JSON, 
	attribution JSON, 
	structured_payload JSON, 
	created_at TIMESTAMP WITH TIME ZONE, 
	updated_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(source_synthesis_job_id) REFERENCES synthesis_jobs (id) ON DELETE SET NULL
)
    """,
    """
CREATE TABLE IF NOT EXISTS agent_control_plane_views (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	filters JSON, 
	is_default BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
)
    """,
    """
CREATE TABLE IF NOT EXISTS experiment_plans (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	research_note_id UUID NOT NULL, 
	title VARCHAR(500) NOT NULL, 
	hypothesis_text TEXT, 
	plan JSON NOT NULL, 
	generator VARCHAR(100), 
	generator_details JSON, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(research_note_id) REFERENCES research_notes (id) ON DELETE CASCADE
)
    """,
    """
CREATE TABLE IF NOT EXISTS research_monitor_profiles (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	customer VARCHAR(255), 
	token_scores JSON, 
	phrase_scores JSON, 
	recommendation_scores JSON, 
	source_type_scores JSON, 
	outcome_counters JSON, 
	customer_budget_config JSON, 
	customer_rebalance_history JSON, 
	muted_tokens JSON, 
	muted_patterns JSON, 
	notes TEXT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_research_monitor_profile_user_customer UNIQUE (user_id, customer), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
)
    """,
    """
CREATE TABLE IF NOT EXISTS code_patch_proposals (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	job_id UUID, 
	source_id UUID, 
	title VARCHAR(500) NOT NULL, 
	summary TEXT, 
	diff_unified TEXT NOT NULL, 
	metadata JSON, 
	status VARCHAR(24) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_code_patch_proposals_job_id UNIQUE (job_id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(job_id) REFERENCES agent_jobs (id) ON DELETE SET NULL, 
	FOREIGN KEY(source_id) REFERENCES document_sources (id) ON DELETE SET NULL
)
    """,
    """
CREATE TABLE IF NOT EXISTS experiment_runs (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	experiment_plan_id UUID NOT NULL, 
	agent_job_id UUID, 
	parent_run_id UUID, 
	latest_child_run_id UUID, 
	name VARCHAR(500) NOT NULL, 
	status VARCHAR(32) NOT NULL, 
	config JSON, 
	results JSON, 
	summary TEXT, 
	progress INTEGER NOT NULL, 
	retry_count INTEGER NOT NULL, 
	started_at TIMESTAMP WITH TIME ZONE, 
	completed_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(experiment_plan_id) REFERENCES experiment_plans (id) ON DELETE CASCADE, 
	FOREIGN KEY(agent_job_id) REFERENCES agent_jobs (id) ON DELETE SET NULL, 
	FOREIGN KEY(parent_run_id) REFERENCES experiment_runs (id) ON DELETE SET NULL, 
	FOREIGN KEY(latest_child_run_id) REFERENCES experiment_runs (id) ON DELETE SET NULL
)
    """,
    """
CREATE TABLE IF NOT EXISTS research_inbox_items (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	job_id UUID, 
	customer VARCHAR(255), 
	item_type VARCHAR(32) NOT NULL, 
	item_key VARCHAR(512) NOT NULL, 
	title VARCHAR(1000) NOT NULL, 
	summary TEXT, 
	url TEXT, 
	published_at TIMESTAMP WITH TIME ZONE, 
	discovered_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	feedback TEXT, 
	metadata JSON, 
	follow_up_decision VARCHAR(32), 
	follow_up_policy_mode VARCHAR(32), 
	follow_up_launch_status VARCHAR(32), 
	follow_up_block_reason TEXT, 
	follow_up_budget_decision VARCHAR(32), 
	follow_up_budget_reason TEXT, 
	follow_up_budget_throttle_state VARCHAR(32), 
	follow_up_customer_budget_decision VARCHAR(32), 
	follow_up_customer_budget_reason TEXT, 
	follow_up_customer_budget_throttle_state VARCHAR(32), 
	follow_up_recommendation_key VARCHAR(100), 
	follow_up_operator_decision VARCHAR(32), 
	follow_up_operator_note TEXT, 
	follow_up_operator_acted_at TIMESTAMP WITH TIME ZONE, 
	follow_up_operator_user_id UUID, 
	follow_up_job_id UUID, 
	follow_up_chain_definition_id UUID, 
	follow_up_launched_at TIMESTAMP WITH TIME ZONE, 
	follow_up_outcome_status VARCHAR(32), 
	follow_up_outcome_recorded_at TIMESTAMP WITH TIME ZONE, 
	follow_up_outcome_summary TEXT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_research_inbox_item_once UNIQUE (user_id, item_type, item_key), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(job_id) REFERENCES agent_jobs (id) ON DELETE SET NULL, 
	FOREIGN KEY(follow_up_operator_user_id) REFERENCES users (id) ON DELETE SET NULL, 
	FOREIGN KEY(follow_up_job_id) REFERENCES agent_jobs (id) ON DELETE SET NULL
)
    """,
    """
CREATE TABLE IF NOT EXISTS research_papers (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	source_id UUID, 
	arxiv_id VARCHAR(128) NOT NULL, 
	title VARCHAR(500) NOT NULL, 
	authors JSON, 
	abstract TEXT, 
	published_at TIMESTAMP WITH TIME ZONE, 
	categories JSON, 
	paper_url VARCHAR(1000), 
	pdf_url VARCHAR(1000), 
	extraction_status VARCHAR(32) NOT NULL, 
	extracted_at TIMESTAMP WITH TIME ZONE, 
	extractor_version VARCHAR(64), 
	summary TEXT, 
	mechanisms JSON, 
	assumptions JSON, 
	benchmarks JSON, 
	metrics JSON, 
	limitations JSON, 
	raw_extraction_payload JSON, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE, 
	FOREIGN KEY(source_id) REFERENCES document_sources (id) ON DELETE SET NULL
)
    """,
    """
CREATE TABLE IF NOT EXISTS upload_sessions (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	filename VARCHAR(500) NOT NULL, 
	file_size INTEGER NOT NULL, 
	file_type VARCHAR(50), 
	content_type VARCHAR(100), 
	chunk_size INTEGER NOT NULL, 
	total_chunks INTEGER NOT NULL, 
	uploaded_chunks JSON NOT NULL, 
	uploaded_bytes INTEGER NOT NULL, 
	minio_upload_id VARCHAR(200), 
	minio_part_etags JSON, 
	title VARCHAR(500), 
	tags JSON, 
	extra_metadata JSON, 
	status VARCHAR(50) NOT NULL, 
	error_message VARCHAR(1000), 
	document_id UUID, 
	created_at TIMESTAMP WITH TIME ZONE, 
	updated_at TIMESTAMP WITH TIME ZONE, 
	expires_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	FOREIGN KEY(document_id) REFERENCES documents (id)
)
    """,
    """
CREATE TABLE IF NOT EXISTS paper_claims (
	id UUID NOT NULL, 
	paper_id UUID NOT NULL, 
	kind VARCHAR(32) NOT NULL, 
	statement TEXT NOT NULL, 
	mechanism VARCHAR(255), 
	target_layer VARCHAR(32) NOT NULL, 
	conditions JSON, 
	assumptions JSON, 
	expected_effect TEXT, 
	evidence_summary TEXT, 
	confidence FLOAT, 
	tags JSON, 
	rank INTEGER, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(paper_id) REFERENCES research_papers (id) ON DELETE CASCADE
)
    """,
    """
CREATE TABLE IF NOT EXISTS paper_extraction_jobs (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	document_id UUID NOT NULL, 
	source_id UUID, 
	paper_id UUID, 
	status VARCHAR(32) NOT NULL, 
	extractor_version VARCHAR(64), 
	error TEXT, 
	request_payload JSON, 
	result_summary JSON, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	started_at TIMESTAMP WITH TIME ZONE, 
	completed_at TIMESTAMP WITH TIME ZONE, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE, 
	FOREIGN KEY(source_id) REFERENCES document_sources (id) ON DELETE SET NULL, 
	FOREIGN KEY(paper_id) REFERENCES research_papers (id) ON DELETE SET NULL
)
    """,
    """
CREATE TABLE IF NOT EXISTS patch_prs (
	id UUID NOT NULL, 
	user_id UUID NOT NULL, 
	source_id UUID, 
	title VARCHAR(500) NOT NULL, 
	description TEXT, 
	status VARCHAR(24) NOT NULL, 
	selected_proposal_id UUID, 
	proposal_ids JSON, 
	checks JSON, 
	approvals JSON, 
	merged_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
	FOREIGN KEY(source_id) REFERENCES document_sources (id) ON DELETE SET NULL, 
	FOREIGN KEY(selected_proposal_id) REFERENCES code_patch_proposals (id) ON DELETE SET NULL
)
    """,
]

MISSING_TABLE_INDEXES = [
    """
CREATE INDEX IF NOT EXISTS ix_research_notes_user_id ON research_notes (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_agent_control_plane_views_user_id ON agent_control_plane_views (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_plans_research_note_id ON experiment_plans (research_note_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_plans_user_id ON experiment_plans (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_plans_created_at ON experiment_plans (created_at)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_monitor_profiles_user_id ON research_monitor_profiles (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_monitor_profiles_user_customer ON research_monitor_profiles (user_id, customer)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_monitor_profiles_customer ON research_monitor_profiles (customer)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_user_status ON code_patch_proposals (user_id, status)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_job_id ON code_patch_proposals (job_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_user_id ON code_patch_proposals (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_source_id ON code_patch_proposals (source_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_user_id ON experiment_runs (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_experiment_plan_id ON experiment_runs (experiment_plan_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_created_at ON experiment_runs (created_at)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_latest_child_run_id ON experiment_runs (latest_child_run_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_parent_run_id ON experiment_runs (parent_run_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_experiment_runs_agent_job_id ON experiment_runs (agent_job_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_inbox_items_user_id ON research_inbox_items (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_inbox_items_follow_up_job_id ON research_inbox_items (follow_up_job_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_inbox_items_discovered_at ON research_inbox_items (discovered_at)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_inbox_items_job_id ON research_inbox_items (job_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_inbox_items_customer ON research_inbox_items (customer)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_inbox_user_status ON research_inbox_items (user_id, status)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_papers_document_id ON research_papers (document_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_papers_arxiv_id ON research_papers (arxiv_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_papers_source_id ON research_papers (source_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_papers_user_id ON research_papers (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_research_papers_extraction_status ON research_papers (extraction_status)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_upload_sessions_user_id ON upload_sessions (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_paper_claims_paper_id ON paper_claims (paper_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_document_id ON paper_extraction_jobs (document_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_status ON paper_extraction_jobs (status)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_paper_id ON paper_extraction_jobs (paper_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_source_id ON paper_extraction_jobs (source_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_user_id ON paper_extraction_jobs (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_patch_prs_user_id ON patch_prs (user_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_patch_prs_user_status ON patch_prs (user_id, status)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_patch_prs_source_id ON patch_prs (source_id)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_patch_prs_user_created ON patch_prs (user_id, created_at)
    """,
    """
CREATE INDEX IF NOT EXISTS ix_patch_prs_selected_proposal_id ON patch_prs (selected_proposal_id)
    """,
]

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
    for statement in (
        MISSING_TABLES + MISSING_TABLE_INDEXES + MISSING_COLUMNS + MISSING_INDEXES
    ):
        op.execute(statement)


def downgrade() -> None:
    # Intentionally a no-op. Upgrade is idempotent and does nothing on databases
    # that predate it, so a symmetric downgrade would drop tables, columns and
    # indexes this migration never created — together with their data.
    pass
