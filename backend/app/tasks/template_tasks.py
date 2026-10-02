"""
Background tasks for template filling.
"""

import asyncio
from typing import Any, Dict
from uuid import UUID

from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.template import TemplateJob
from app.services.template_fill_service import TemplateFillService
from app.tasks import job_support


@celery_app.task(bind=True, name="app.tasks.template_tasks.fill_template")
def fill_template(self, job_id: str) -> Dict[str, Any]:
    """
    Background task to fill a template document.

    Args:
        job_id: UUID string of the TemplateJob

    Returns:
        Dictionary with success status and result info
    """
    return asyncio.run(_async_fill_template(self, job_id))


async def _async_fill_template(task, job_id: str) -> Dict[str, Any]:
    """Async implementation of template filling."""
    async with create_celery_session()() as db:
        try:
            logger.info(f"Starting template fill job {job_id}")

            # Load job
            job = await db.get(TemplateJob, UUID(job_id))
            if not job:
                error_msg = f"Template job {job_id} not found"
                logger.error(error_msg)
                _publish_error(job_id, error_msg)
                return {"success": False, "job_id": job_id, "error": error_msg}

            # Initialize service
            service = TemplateFillService()

            # Define progress callback
            def progress_callback(data: dict):
                _publish_progress(job_id, data)

            # Process the job
            result = await service.process_template_job(job, db, progress_callback)

            if result.get("success"):
                _publish_complete(
                    job_id,
                    {
                        "filled_filename": result.get("filled_filename"),
                        "filled_file_path": result.get("filled_file_path"),
                    },
                )
            else:
                _publish_error(job_id, result.get("error", "Unknown error"))

            return result

        except Exception as e:
            logger.error(f"Template fill task failed for job {job_id}: {e}")
            _publish_error(job_id, str(e))

            # Update job status
            try:
                job = await db.get(TemplateJob, UUID(job_id))
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    await db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update job status: {db_error}")

            return {"success": False, "job_id": job_id, "error": str(e)}


def _publish_progress(job_id: str, progress: dict):
    """Publish progress update via Redis."""
    job_support.publish_sync(
        f"template_progress:{job_id}",
        {"type": "progress", "job_id": job_id, "data": progress},
    )


def _publish_complete(job_id: str, result: dict):
    """Publish completion event via Redis."""
    job_support.publish_sync(
        f"template_progress:{job_id}",
        {"type": "complete", "job_id": job_id, "result": result},
    )


def _publish_error(job_id: str, error: str):
    """Publish error event via Redis."""
    job_support.publish_sync(
        f"template_progress:{job_id}",
        {"type": "error", "job_id": job_id, "error": error},
    )
