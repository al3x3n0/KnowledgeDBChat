"""
Maintenance tasks for LaTeX Studio compile jobs.

These should run on the *default* Celery queue so they keep working even if the
dedicated LaTeX compile worker is down.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict

from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.config import settings
from app.core.database import create_celery_session
from app.models.latex_compile_job import LatexCompileJob


@celery_app.task(bind=True, name="app.tasks.latex_maintenance_tasks.fail_stale_latex_compile_jobs")
def fail_stale_latex_compile_jobs(self) -> Dict[str, Any]:
    return asyncio.run(_async_fail_stale_latex_compile_jobs())


async def _async_fail_stale_latex_compile_jobs() -> Dict[str, Any]:
    now = datetime.utcnow()
    queued_seconds = int(getattr(settings, "LATEX_COMPILER_JOB_QUEUED_STALE_SECONDS", 600) or 600)
    running_seconds = int(getattr(settings, "LATEX_COMPILER_JOB_RUNNING_STALE_SECONDS", 300) or 300)

    queued_cutoff = now - timedelta(seconds=queued_seconds)
    running_cutoff = now - timedelta(seconds=running_seconds)

    updated = 0
    async with create_celery_session()() as db:
        # Queued jobs that never started
        result = await db.execute(
            select(LatexCompileJob).where(
                (LatexCompileJob.status == "queued") & (LatexCompileJob.created_at < queued_cutoff)
            )
        )
        for job in result.scalars().all():
            job.status = "failed"
            job.log = (job.log or "").strip() or "Compile job timed out in queue (no worker picked it up)."
            job.finished_at = now
            updated += 1

        # Running jobs that exceeded expected runtime
        result = await db.execute(
            select(LatexCompileJob).where(
                (LatexCompileJob.status == "running") & ((LatexCompileJob.started_at < running_cutoff) | ((LatexCompileJob.started_at == None) & (LatexCompileJob.created_at < running_cutoff)))  # noqa: E711
            )
        )
        for job in result.scalars().all():
            job.status = "failed"
            job.log = (job.log or "").strip() or "Compile job timed out (worker exceeded limits)."
            job.finished_at = now
            updated += 1

        if updated:
            await db.commit()

    if updated:
        logger.warning(f"Marked {updated} stale LaTeX compile job(s) as failed.")
    return {"updated": updated, "queued_stale_seconds": queued_seconds, "running_stale_seconds": running_seconds}

