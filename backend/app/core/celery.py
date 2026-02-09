"""
Celery configuration for background task processing.
"""

from celery import Celery
from celery.schedules import crontab
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "knowledge_db",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ingestion_tasks",
        "app.tasks.url_ingestion_tasks",
        "app.tasks.processing_tasks",
        "app.tasks.sync_tasks",
        "app.tasks.chat_tasks",
        "app.tasks.transcription_tasks",
        "app.tasks.transcode_tasks",
        "app.tasks.summarization_tasks",
        "app.tasks.monitoring_tasks",
        "app.tasks.presentation_tasks",
        "app.tasks.git_compare_tasks",
        "app.tasks.research_tasks",
        "app.tasks.paper_kg_tasks",
        "app.tasks.paper_enrichment_tasks",
        "app.tasks.maintenance_tasks",
        "app.tasks.repo_report_tasks",
        "app.tasks.template_tasks",
        "app.tasks.agent_job_tasks",
        "app.tasks.training_tasks",
        "app.tasks.latex_tasks",
        "app.tasks.latex_maintenance_tasks",
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        # Route only the heavy LaTeX compile task to a dedicated queue by default.
        "app.tasks.latex_tasks.compile_latex_project_job": {"queue": getattr(settings, "LATEX_COMPILER_CELERY_QUEUE", "latex") or "latex"},
    },
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    # Sync GitLab sources every hour
    "sync-gitlab-sources": {
        "task": "app.tasks.sync_tasks.sync_all_gitlab_sources",
        "schedule": crontab(minute=0),  # Every hour
    },
    
    # Sync Confluence sources every 2 hours
    "sync-confluence-sources": {
        "task": "app.tasks.sync_tasks.sync_all_confluence_sources",
        "schedule": crontab(minute=0, hour="*/2"),  # Every 2 hours
    },
    
    # Sync web sources daily at 2 AM
    "sync-web-sources": {
        "task": "app.tasks.sync_tasks.sync_all_web_sources",
        "schedule": crontab(minute=0, hour=2),  # Daily at 2 AM
    },
    
    # Clean up old logs and temporary files weekly
    "cleanup-old-data": {
        "task": "app.tasks.maintenance_tasks.cleanup_old_data",
        "schedule": crontab(minute=0, hour=3, day_of_week=0),  # Weekly on Sunday at 3 AM
    },
    
    # Health check every 15 minutes
    "health-check": {
        "task": "app.tasks.monitoring_tasks.health_check",
        "schedule": crontab(minute="*/15"),
    },

    # Sync experiment runs from linked agent jobs (every 5 minutes)
    "sync-experiment-runs": {
        "task": "app.tasks.monitoring_tasks.sync_experiment_runs",
        "schedule": crontab(minute="*/5"),
    },

    # Lint citations in recently-updated research notes (every hour)
    "lint-research-note-citations": {
        "task": "app.tasks.monitoring_tasks.lint_recent_research_notes_citations",
        "schedule": crontab(minute=0),  # Every hour
    },
    
    # Per-source scheduling scan (every 5 minutes)
    "scan-scheduled-sources": {
        "task": "app.tasks.sync_tasks.scan_scheduled_sources",
        "schedule": crontab(minute="*/5"),
    },

    # Process scheduled agent jobs (every 5 minutes)
    "process-scheduled-agent-jobs": {
        "task": "app.tasks.agent_job_tasks.process_scheduled_agent_jobs",
        "schedule": crontab(minute="*/5"),
    },

    # Check for stalled agent jobs (every 10 minutes)
    "check-stalled-agent-jobs": {
        "task": "app.tasks.agent_job_tasks.check_stalled_agent_jobs",
        "schedule": crontab(minute="*/10"),
    },

    # Resume paused agent jobs (every 15 minutes)
    "resume-paused-agent-jobs": {
        "task": "app.tasks.agent_job_tasks.resume_paused_agent_jobs",
        "schedule": crontab(minute="*/15"),
    },

    # Cleanup old agent jobs (weekly on Sunday at 4 AM)
    "cleanup-old-agent-jobs": {
        "task": "app.tasks.agent_job_tasks.cleanup_old_agent_jobs",
        "schedule": crontab(minute=0, hour=4, day_of_week=0),
    },

    # Fail stale LaTeX compile jobs (every 5 minutes)
    "fail-stale-latex-compile-jobs": {
        "task": "app.tasks.latex_maintenance_tasks.fail_stale_latex_compile_jobs",
        "schedule": crontab(minute="*/5"),
    },
}

celery_app.conf.timezone = "UTC"
