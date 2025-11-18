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
        "app.tasks.processing_tasks",
        "app.tasks.sync_tasks"
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
}

celery_app.conf.timezone = "UTC"







