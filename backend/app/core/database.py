"""
Database configuration and connection management.
"""

import asyncio
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import NullPool
from loguru import logger

from .config import settings

# Convert sync database URL to async for async operations
async_database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
engine = create_async_engine(
    async_database_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT_SECONDS,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create declarative base
Base = declarative_base()

# Metadata for table creation
metadata = MetaData()

_db_session_semaphore: asyncio.Semaphore | None = None


def _get_db_session_semaphore() -> asyncio.Semaphore:
    global _db_session_semaphore
    if _db_session_semaphore is None:
        limit = settings.DB_SESSION_CONCURRENCY_LIMIT
        if limit is None:
            limit = max(1, int(settings.DB_POOL_SIZE) + int(settings.DB_MAX_OVERFLOW))
        else:
            limit = max(1, int(limit))
        _db_session_semaphore = asyncio.Semaphore(limit)
        logger.info(f"DB session concurrency limit: {limit}")
    return _db_session_semaphore


async def get_db() -> AsyncSession:
    """
    Get database session.
    
    Yields:
        AsyncSession: Database session
    """
    semaphore = _get_db_session_semaphore()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=float(settings.DB_SESSION_ACQUIRE_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        # Avoid dogpiling the pool; fail fast under load.
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is busy (DB concurrency limit reached). Please retry in a moment.",
            headers={"Retry-After": "2"},
        )

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            # Don't log HTTPException as database errors - they're expected API responses
            from fastapi import HTTPException
            from fastapi.exceptions import RequestValidationError
            if not isinstance(e, (HTTPException, RequestValidationError)):
                logger.opt(exception=True).error("Database session error: {}", str(e))
            await session.rollback()
            raise
        finally:
            await session.close()
            try:
                semaphore.release()
            except Exception:
                pass


async def create_tables():
    """Create database tables."""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            import app.models  # noqa: F401
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

            # Apply lightweight migrations for columns added after initial release
            await _apply_minimal_migrations(conn)
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


async def apply_minimal_migrations() -> None:
    """
    Apply idempotent schema updates that are safe to run on every startup.

    This is intended for deployments that don't run Alembic migrations automatically.
    """
    try:
        async with engine.begin() as conn:
            await _apply_minimal_migrations(conn)
    except Exception as e:
        logger.warning(f"Error applying minimal migrations: {e}")


async def drop_tables():
    """Drop all database tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


def create_celery_session():
    """
    Create a fresh async session for Celery tasks.

    Celery workers fork from the main process, and the global engine's connection pool
    is bound to the parent's event loop. When asyncio.run() creates a new event loop
    in the worker, the old engine is incompatible, causing "Future attached to a
    different loop" errors.

    This function creates a fresh engine and session factory for each task invocation.

    Returns:
        AsyncSession factory (sessionmaker instance)
    """
    if settings.CELERY_DB_USE_NULLPOOL:
        # NullPool doesn't support QueuePool tuning kwargs like pool_size/max_overflow/pool_timeout.
        # It's also pointless to pre-ping/recycle when every checkout creates a fresh connection.
        kwargs = dict(
            echo=settings.DEBUG,
            poolclass=NullPool,
        )
    else:
        kwargs = dict(
            echo=settings.DEBUG,
            pool_pre_ping=True,
            pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
            pool_timeout=settings.CELERY_DB_POOL_TIMEOUT_SECONDS,
            pool_size=settings.CELERY_DB_POOL_SIZE,
            max_overflow=settings.CELERY_DB_MAX_OVERFLOW,
        )

    fresh_engine = create_async_engine(async_database_url, **kwargs)
    return sessionmaker(
        fresh_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def _apply_minimal_migrations(conn) -> None:
    """
    Apply idempotent schema updates that are safe to run on every startup.
    This supplements Alembic for environments that only rely on create_all.
    """
    statements = [
        # Document summarization columns (introduced after initial schema)
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_model VARCHAR(100)",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_generated_at TIMESTAMPTZ",
        # User LLM per-task provider overrides
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS llm_task_providers JSON",
        # Paper algorithm defaults
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS paper_algo_default_run_demo_check BOOLEAN NOT NULL DEFAULT FALSE",

        # Agent routing defaults
        "ALTER TABLE agent_definitions ADD COLUMN IF NOT EXISTS routing_defaults JSON",

        # Notifications
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_research_note_citation_issues BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_experiment_run_updates BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_hypothesis_reevaluation_updates BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_queue_urgency_alerts BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_follow_up_outcome_alerts BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_policy_guardrail_alerts BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_autonomy_budget_alerts BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS notify_customer_autonomy_budget_alerts BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_coverage_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.7",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_cooldown_hours INTEGER NOT NULL DEFAULT 12",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS queue_urgency_alert_reminder_cooldown_hours INTEGER NOT NULL DEFAULT 6",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_on_unknown_keys BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_on_low_coverage BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE notification_preferences ADD COLUMN IF NOT EXISTS research_note_citation_notify_on_missing_bibliography BOOLEAN NOT NULL DEFAULT TRUE",

        # Research inbox follow-up autonomy state
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_decision VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_policy_mode VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_launch_status VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_block_reason TEXT",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_budget_decision VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_budget_reason TEXT",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_budget_throttle_state VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_customer_budget_decision VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_customer_budget_reason TEXT",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_customer_budget_throttle_state VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_recommendation_key VARCHAR(100)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_operator_decision VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_operator_note TEXT",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_operator_acted_at TIMESTAMPTZ",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_operator_user_id UUID",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_job_id UUID",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_chain_definition_id UUID",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_launched_at TIMESTAMPTZ",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_outcome_status VARCHAR(32)",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_outcome_recorded_at TIMESTAMPTZ",
        "ALTER TABLE research_inbox_items ADD COLUMN IF NOT EXISTS follow_up_outcome_summary TEXT",

        # Tool audits: policy provenance
        "ALTER TABLE tool_execution_audits ADD COLUMN IF NOT EXISTS policy_decision JSONB",
        "ALTER TABLE tool_execution_audits ADD COLUMN IF NOT EXISTS approval_mode VARCHAR(32) DEFAULT 'owner_and_admin'",
        "ALTER TABLE tool_execution_audits ADD COLUMN IF NOT EXISTS owner_approved_by UUID",
        "ALTER TABLE tool_execution_audits ADD COLUMN IF NOT EXISTS owner_approved_at TIMESTAMPTZ",
        "ALTER TABLE tool_execution_audits ADD COLUMN IF NOT EXISTS admin_approved_by UUID",
        "ALTER TABLE tool_execution_audits ADD COLUMN IF NOT EXISTS admin_approved_at TIMESTAMPTZ",
        "ALTER TABLE research_monitor_profiles ADD COLUMN IF NOT EXISTS customer_budget_config JSON",
        "ALTER TABLE research_monitor_profiles ADD COLUMN IF NOT EXISTS customer_rebalance_history JSON",

        # Persistent agent tool priors
        """
        CREATE TABLE IF NOT EXISTS agent_tool_priors (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_type VARCHAR(50) NOT NULL,
            tool_name VARCHAR(120) NOT NULL,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_tool_priors_user_job_tool ON agent_tool_priors(user_id, job_type, tool_name)",
        "CREATE INDEX IF NOT EXISTS ix_agent_tool_priors_user_id ON agent_tool_priors(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_agent_tool_priors_job_type ON agent_tool_priors(job_type)",
        "CREATE INDEX IF NOT EXISTS ix_agent_tool_priors_tool_name ON agent_tool_priors(tool_name)",
        "CREATE INDEX IF NOT EXISTS ix_agent_tool_priors_user_job ON agent_tool_priors(user_id, job_type)",

        # External auth fields (LDAP)
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) NOT NULL DEFAULT 'local'",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_subject VARCHAR(512)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_metadata JSON",
        "CREATE INDEX IF NOT EXISTS ix_users_auth_provider ON users(auth_provider)",
        "CREATE INDEX IF NOT EXISTS ix_users_auth_subject ON users(auth_subject)",

        # Reading lists (collections) - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS reading_lists (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT NULL,
            source_id UUID NULL REFERENCES document_sources(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_list_user_name ON reading_lists(user_id, name)",
        "CREATE INDEX IF NOT EXISTS ix_reading_lists_user_id ON reading_lists(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_reading_lists_source_id ON reading_lists(source_id)",
        """
        CREATE TABLE IF NOT EXISTS reading_list_items (
            id UUID PRIMARY KEY,
            reading_list_id UUID NOT NULL REFERENCES reading_lists(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            status VARCHAR(16) NOT NULL DEFAULT 'to-read',
            priority INTEGER NOT NULL DEFAULT 0,
            position INTEGER NOT NULL DEFAULT 0,
            notes TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_list_item_document_once ON reading_list_items(reading_list_id, document_id)",
        "CREATE INDEX IF NOT EXISTS ix_reading_list_items_reading_list_id ON reading_list_items(reading_list_id)",
        "CREATE INDEX IF NOT EXISTS ix_reading_list_items_document_id ON reading_list_items(document_id)",
        "CREATE INDEX IF NOT EXISTS ix_reading_list_items_list_position ON reading_list_items(reading_list_id, position)",
        # Research notes - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS research_notes (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(500) NOT NULL,
            content_markdown TEXT NOT NULL,
            source_synthesis_job_id UUID NULL REFERENCES synthesis_jobs(id) ON DELETE SET NULL,
            source_document_ids JSON NULL,
            tags JSON NULL,
            attribution JSON NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "ALTER TABLE research_notes ADD COLUMN IF NOT EXISTS attribution JSON",
        "ALTER TABLE research_notes ADD COLUMN IF NOT EXISTS structured_payload JSON",
        "CREATE INDEX IF NOT EXISTS ix_research_notes_user_id ON research_notes(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_notes_created_at ON research_notes(created_at)",
        "ALTER TABLE synthesis_jobs ADD COLUMN IF NOT EXISTS paper_ids JSON NOT NULL DEFAULT '[]'",
        "ALTER TABLE synthesis_jobs ADD COLUMN IF NOT EXISTS research_note_id UUID NULL REFERENCES research_notes(id) ON DELETE SET NULL",

        # Structured research papers - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS research_papers (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            source_id UUID NULL REFERENCES document_sources(id) ON DELETE SET NULL,
            arxiv_id VARCHAR(128) NOT NULL,
            title VARCHAR(500) NOT NULL,
            authors JSON NULL,
            abstract TEXT NULL,
            published_at TIMESTAMPTZ NULL,
            categories JSON NULL,
            paper_url VARCHAR(1000) NULL,
            pdf_url VARCHAR(1000) NULL,
            extraction_status VARCHAR(32) NOT NULL DEFAULT 'pending',
            extracted_at TIMESTAMPTZ NULL,
            extractor_version VARCHAR(64) NULL,
            summary TEXT NULL,
            mechanisms JSON NULL,
            assumptions JSON NULL,
            benchmarks JSON NULL,
            metrics JSON NULL,
            limitations JSON NULL,
            raw_extraction_payload JSON NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_papers_document_id ON research_papers(document_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_papers_user_id ON research_papers(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_papers_source_id ON research_papers(source_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_papers_arxiv_id ON research_papers(arxiv_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_papers_extraction_status ON research_papers(extraction_status)",
        """
        CREATE TABLE IF NOT EXISTS paper_claims (
            id UUID PRIMARY KEY,
            paper_id UUID NOT NULL REFERENCES research_papers(id) ON DELETE CASCADE,
            kind VARCHAR(32) NOT NULL DEFAULT 'other',
            statement TEXT NOT NULL,
            mechanism VARCHAR(255) NULL,
            target_layer VARCHAR(32) NOT NULL DEFAULT 'unknown',
            conditions JSON NULL,
            assumptions JSON NULL,
            expected_effect TEXT NULL,
            evidence_summary TEXT NULL,
            confidence DOUBLE PRECISION NULL,
            tags JSON NULL,
            rank INTEGER NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_paper_claims_paper_id ON paper_claims(paper_id)",
        """
        CREATE TABLE IF NOT EXISTS paper_extraction_jobs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            source_id UUID NULL REFERENCES document_sources(id) ON DELETE SET NULL,
            paper_id UUID NULL REFERENCES research_papers(id) ON DELETE SET NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'pending',
            extractor_version VARCHAR(64) NULL,
            error TEXT NULL,
            request_payload JSON NULL,
            result_summary JSON NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_user_id ON paper_extraction_jobs(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_document_id ON paper_extraction_jobs(document_id)",
        "CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_source_id ON paper_extraction_jobs(source_id)",
        "CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_paper_id ON paper_extraction_jobs(paper_id)",
        "CREATE INDEX IF NOT EXISTS ix_paper_extraction_jobs_status ON paper_extraction_jobs(status)",

        # Experiments - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS experiment_plans (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            research_note_id UUID NOT NULL REFERENCES research_notes(id) ON DELETE CASCADE,
            title VARCHAR(500) NOT NULL,
            hypothesis_text TEXT NULL,
            plan JSON NOT NULL,
            generator VARCHAR(100) NULL,
            generator_details JSON NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_experiment_plans_user_id ON experiment_plans(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_experiment_plans_note_id ON experiment_plans(research_note_id)",
        "CREATE INDEX IF NOT EXISTS ix_experiment_plans_created_at ON experiment_plans(created_at)",
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            experiment_plan_id UUID NOT NULL REFERENCES experiment_plans(id) ON DELETE CASCADE,
            agent_job_id UUID NULL REFERENCES agent_jobs(id) ON DELETE SET NULL,
            name VARCHAR(500) NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'planned',
            config JSON NULL,
            results JSON NULL,
            summary TEXT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMPTZ NULL,
            completed_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "ALTER TABLE experiment_runs ADD COLUMN IF NOT EXISTS agent_job_id UUID",
        "DO $$ BEGIN ALTER TABLE experiment_runs ADD CONSTRAINT experiment_runs_agent_job_id_fkey FOREIGN KEY (agent_job_id) REFERENCES agent_jobs(id) ON DELETE SET NULL; EXCEPTION WHEN duplicate_object THEN NULL; END $$;",
        "CREATE INDEX IF NOT EXISTS ix_experiment_runs_user_id ON experiment_runs(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_experiment_runs_plan_id ON experiment_runs(experiment_plan_id)",
        "CREATE INDEX IF NOT EXISTS ix_experiment_runs_agent_job_id ON experiment_runs(agent_job_id)",
        "CREATE INDEX IF NOT EXISTS ix_experiment_runs_created_at ON experiment_runs(created_at)",

        # Benchmark harness - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS benchmark_suites (
            id VARCHAR(120) PRIMARY KEY,
            user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT NULL,
            track_type VARCHAR(32) NOT NULL DEFAULT 'compiler',
            benchmark_family VARCHAR(64) NOT NULL DEFAULT 'compiler_regression',
            suite_version INTEGER NOT NULL DEFAULT 1,
            tags JSON NOT NULL DEFAULT '[]',
            metadata_json JSON NOT NULL DEFAULT '{}',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            system_managed BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_benchmark_suites_user_id ON benchmark_suites(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_suites_track_type ON benchmark_suites(track_type)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_suites_benchmark_family ON benchmark_suites(benchmark_family)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_suites_enabled ON benchmark_suites(enabled)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_suites_system_managed ON benchmark_suites(system_managed)",
        """
        CREATE TABLE IF NOT EXISTS benchmark_cases (
            id VARCHAR(120) PRIMARY KEY,
            suite_id VARCHAR(120) NOT NULL REFERENCES benchmark_suites(id) ON DELETE CASCADE,
            name VARCHAR(200) NOT NULL,
            description TEXT NULL,
            rank INTEGER NOT NULL DEFAULT 0,
            source_ref VARCHAR(500) NULL,
            benchmark_query VARCHAR(500) NULL,
            compile_command_template TEXT NULL,
            run_command_template TEXT NULL,
            expected_artifacts JSON NOT NULL DEFAULT '[]',
            metrics JSON NOT NULL DEFAULT '[]',
            metadata_json JSON NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_benchmark_cases_suite_id ON benchmark_cases(suite_id)",
        """
        CREATE TABLE IF NOT EXISTS benchmark_baselines (
            id VARCHAR(120) PRIMARY KEY,
            suite_id VARCHAR(120) NOT NULL REFERENCES benchmark_suites(id) ON DELETE CASCADE,
            case_id VARCHAR(120) NULL REFERENCES benchmark_cases(id) ON DELETE SET NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT NULL,
            compiler_revision VARCHAR(120) NULL,
            toolchain_id VARCHAR(120) NULL,
            sandbox_profile_id VARCHAR(120) NULL,
            measurements JSON NOT NULL DEFAULT '{}',
            environment_snapshot JSON NOT NULL DEFAULT '{}',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            system_managed BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_benchmark_baselines_suite_id ON benchmark_baselines(suite_id)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_baselines_case_id ON benchmark_baselines(case_id)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_baselines_enabled ON benchmark_baselines(enabled)",
        "CREATE INDEX IF NOT EXISTS ix_benchmark_baselines_system_managed ON benchmark_baselines(system_managed)",

        # Research inbox items - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS research_inbox_items (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_id UUID NULL REFERENCES agent_jobs(id) ON DELETE SET NULL,
            customer VARCHAR(255) NULL,
            item_type VARCHAR(32) NOT NULL,
            item_key VARCHAR(512) NOT NULL,
            title VARCHAR(1000) NOT NULL,
            summary TEXT NULL,
            url TEXT NULL,
            published_at TIMESTAMPTZ NULL,
            discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status VARCHAR(16) NOT NULL DEFAULT 'new',
            feedback TEXT NULL,
            metadata JSON NULL,
            follow_up_decision VARCHAR(32) NULL,
            follow_up_policy_mode VARCHAR(32) NULL,
            follow_up_launch_status VARCHAR(32) NULL,
            follow_up_block_reason TEXT NULL,
            follow_up_budget_decision VARCHAR(32) NULL,
            follow_up_budget_reason TEXT NULL,
            follow_up_budget_throttle_state VARCHAR(32) NULL,
            follow_up_recommendation_key VARCHAR(100) NULL,
            follow_up_operator_decision VARCHAR(32) NULL,
            follow_up_operator_note TEXT NULL,
            follow_up_operator_acted_at TIMESTAMPTZ NULL,
            follow_up_operator_user_id UUID NULL,
            follow_up_job_id UUID NULL,
            follow_up_chain_definition_id UUID NULL,
            follow_up_launched_at TIMESTAMPTZ NULL,
            follow_up_outcome_status VARCHAR(32) NULL,
            follow_up_outcome_recorded_at TIMESTAMPTZ NULL,
            follow_up_outcome_summary TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_inbox_item_once ON research_inbox_items(user_id, item_type, item_key)",
        "CREATE INDEX IF NOT EXISTS ix_research_inbox_items_user_id ON research_inbox_items(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_inbox_items_job_id ON research_inbox_items(job_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_inbox_items_customer ON research_inbox_items(customer)",
        "CREATE INDEX IF NOT EXISTS ix_research_inbox_items_discovered_at ON research_inbox_items(discovered_at)",
        "CREATE INDEX IF NOT EXISTS ix_research_inbox_user_status ON research_inbox_items(user_id, status)",

        # Research monitor profiles - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS research_monitor_profiles (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            customer VARCHAR(255) NULL,
            token_scores JSON NULL,
            phrase_scores JSON NULL,
            recommendation_scores JSON NULL,
            source_type_scores JSON NULL,
            outcome_counters JSON NULL,
            muted_tokens JSON NULL,
            muted_patterns JSON NULL,
            notes TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_research_monitor_profile_user_customer ON research_monitor_profiles(user_id, customer)",
        "CREATE INDEX IF NOT EXISTS ix_research_monitor_profiles_user_id ON research_monitor_profiles(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_research_monitor_profiles_customer ON research_monitor_profiles(customer)",
        "CREATE INDEX IF NOT EXISTS ix_research_monitor_profiles_user_customer ON research_monitor_profiles(user_id, customer)",
        "ALTER TABLE research_monitor_profiles ADD COLUMN IF NOT EXISTS phrase_scores JSON",
        "ALTER TABLE research_monitor_profiles ADD COLUMN IF NOT EXISTS recommendation_scores JSON",
        "ALTER TABLE research_monitor_profiles ADD COLUMN IF NOT EXISTS source_type_scores JSON",
        "ALTER TABLE research_monitor_profiles ADD COLUMN IF NOT EXISTS outcome_counters JSON",

        # Code patch proposals - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS code_patch_proposals (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            job_id UUID NULL REFERENCES agent_jobs(id) ON DELETE SET NULL,
            source_id UUID NULL REFERENCES document_sources(id) ON DELETE SET NULL,
            title VARCHAR(500) NOT NULL,
            summary TEXT NULL,
            diff_unified TEXT NOT NULL,
            metadata JSON NULL,
            status VARCHAR(24) NOT NULL DEFAULT 'proposed',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_code_patch_proposals_job_id ON code_patch_proposals(job_id)",
        "CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_user_id ON code_patch_proposals(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_job_id ON code_patch_proposals(job_id)",
        "CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_source_id ON code_patch_proposals(source_id)",
        "CREATE INDEX IF NOT EXISTS ix_code_patch_proposals_user_status ON code_patch_proposals(user_id, status)",

        # Patch PRs - for environments without Alembic
        """
        CREATE TABLE IF NOT EXISTS patch_prs (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            source_id UUID NULL REFERENCES document_sources(id) ON DELETE SET NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT NULL,
            status VARCHAR(24) NOT NULL DEFAULT 'draft',
            selected_proposal_id UUID NULL REFERENCES code_patch_proposals(id) ON DELETE SET NULL,
            proposal_ids JSON NULL,
            checks JSON NULL,
            approvals JSON NULL,
            merged_at TIMESTAMPTZ NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_patch_prs_user_id ON patch_prs(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_patch_prs_source_id ON patch_prs(source_id)",
        "CREATE INDEX IF NOT EXISTS ix_patch_prs_selected_proposal_id ON patch_prs(selected_proposal_id)",
        "CREATE INDEX IF NOT EXISTS ix_patch_prs_user_status ON patch_prs(user_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_patch_prs_user_created ON patch_prs(user_id, created_at)",
    ]

    for stmt in statements:
        try:
            await conn.execute(text(stmt))
        except Exception as exc:
            logger.warning(f"Minimal migration statement failed ({stmt}): {exc}")
