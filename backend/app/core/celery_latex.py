"""
Celery configuration for the dedicated LaTeX compilation worker.

This intentionally avoids importing the full task include list from `app.core.celery`
so the LaTeX worker image can use a much smaller Python dependency set.
"""

from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "knowledge_db_latex",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.latex_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "app.tasks.latex_tasks.compile_latex_project_job": {
            "queue": getattr(settings, "LATEX_COMPILER_CELERY_QUEUE", "latex") or "latex"
        },
    },
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

celery_app.conf.timezone = "UTC"

