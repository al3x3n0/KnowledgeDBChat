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
        "app.tasks.paper_extraction_tasks",
        "app.tasks.maintenance_tasks",
        "app.tasks.repo_report_tasks",
        "app.tasks.template_tasks",
        "app.tasks.agent_job_tasks",
        "app.tasks.training_tasks",
        "app.tasks.latex_tasks",
        "app.tasks.latex_maintenance_tasks",
        "app.tasks.compops_sync_tasks",
        "app.tasks.agent_external_call_outbox_tasks",
        "app.tasks.autonomous_rnd_eval_tasks",
    ],
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
        "app.tasks.latex_tasks.compile_latex_project_job": {
            "queue": getattr(settings, "LATEX_COMPILER_CELERY_QUEUE", "latex")
            or "latex"
        },
    },
    task_annotations={
        # Transcription may include model download/init + long media decode on CPU.
        # Keep a generous default so calls using .delay() do not die at the global 25m soft limit.
        "app.tasks.transcription_tasks.transcribe_document": {
            "soft_time_limit": 5 * 60 * 60,  # 5 hours
            "time_limit": 6 * 60 * 60,  # 6 hours
        },
        # An agent job is the longest-running task here and the only one whose
        # own limits promise more than the global one allows. max_runtime_minutes
        # accepts up to 480 and defaults to 60, and the execution lease runs to
        # 30 -- while the global soft limit killed the task at 25, so no job
        # could ever reach the budget it was configured with. A chained
        # simulation study died at 13 iterations with three ceilings disagreeing
        # and only the invisible one binding. Sized to the 480-minute maximum
        # the schema permits, so the job's own limit is what stops it.
        "app.tasks.agent_job_tasks.execute_agent_job_task": {
            "soft_time_limit": 8 * 60 * 60 + 10 * 60,  # 8h10m
            "time_limit": 8 * 60 * 60 + 20 * 60,  # 8h20m
        },
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
        "schedule": crontab(
            minute=0, hour=3, day_of_week=0
        ),  # Weekly on Sunday at 3 AM
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
    # Refresh bounded CompOps evidence subscriptions (every 5 minutes)
    "sync-compops-evidence": {
        "task": "app.tasks.compops_sync_tasks.sync_due_compops_evidence",
        "schedule": crontab(minute="*/5"),
    },
    # Deliver committed external-agent outbox calls (every minute)
    "deliver-agent-external-call-outbox": {
        "task": (
            "app.tasks.agent_external_call_outbox_tasks." "deliver_external_call_outbox"
        ),
        "schedule": crontab(minute="*"),
    },
    # Emit queue urgency notifications from the derived checkpoint queue (every 10 minutes)
    "emit-queue-urgency-alerts": {
        "task": "app.tasks.monitoring_tasks.emit_queue_urgency_alerts",
        "schedule": crontab(minute="*/10"),
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
    # Advance research campaigns (every 5 minutes).
    #
    # A campaign is a line of enquiry across many jobs, and it keeps all its
    # state in the database rather than in a process. Running one is therefore
    # a matter of asking it to take a step often enough: five minutes is short
    # against jobs that take tens of minutes, and each tick launches at most
    # one job per campaign, so the cost is bounded by how many are active.
    "advance-research-campaigns": {
        "task": "app.tasks.agent_job_tasks.advance_research_campaigns",
        "schedule": crontab(minute="*/5"),
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
