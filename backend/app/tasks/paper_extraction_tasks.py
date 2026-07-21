"""
Celery tasks for structured paper extraction.
"""

import asyncio
from typing import Any, Dict
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document
from app.models.research_paper import PaperExtractionJob
from app.services.paper_extraction_service import paper_extraction_service


@celery_app.task(bind=True, name="app.tasks.paper_extraction_tasks.extract_paper_job")
def extract_paper_job(self, job_id: str) -> Dict[str, Any]:
    return asyncio.run(_async_extract_paper_job(job_id))


async def _async_extract_paper_job(job_id: str) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        stmt = select(PaperExtractionJob).where(PaperExtractionJob.id == UUID(job_id))
        job = (await db.execute(stmt)).scalar_one_or_none()
        if not job:
            return {"success": False, "error": "Job not found", "job_id": job_id}
        document = await db.get(Document, job.document_id)
        if document is None:
            await paper_extraction_service.mark_failed(db, job=job, error="Document not found")
            return {"success": False, "job_id": job_id, "error": "Document not found"}
        await db.refresh(document, ["source"])
        try:
            paper = await paper_extraction_service.extract_document(
                db,
                document=document,
                user_id=job.user_id,
                job=job,
            )
            return {
                "success": True,
                "job_id": job_id,
                "paper_id": str(paper.id),
                "claims_count": len(paper.claims),
            }
        except Exception as exc:
            logger.error(f"Paper extraction failed for job {job_id}: {exc}")
            await paper_extraction_service.mark_failed(db, job=job, error=str(exc))
            return {"success": False, "job_id": job_id, "error": str(exc)}
