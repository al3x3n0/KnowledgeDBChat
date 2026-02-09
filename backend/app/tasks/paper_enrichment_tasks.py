"""
Tasks for paper metadata enrichment (BibTeX/DOI/venue/keywords).
"""

import asyncio
from typing import Any, Dict
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document, DocumentSource
from app.services.paper_enrichment_service import PaperEnrichmentService


@celery_app.task(bind=True, name="app.tasks.paper_enrichment_tasks.enrich_arxiv_document")
def enrich_arxiv_document(self, document_id: str, force: bool = False) -> Dict[str, Any]:
    return asyncio.run(_async_enrich_document(self, document_id, force))


async def _async_enrich_document(task, document_id: str, force: bool) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            svc = PaperEnrichmentService()
            return await svc.enrich_arxiv_document(db, UUID(document_id), force=force)
        except Exception as exc:
            logger.error(f"Paper enrichment failed for {document_id}: {exc}")
            return {"success": False, "document_id": document_id, "error": str(exc)}


@celery_app.task(bind=True, name="app.tasks.paper_enrichment_tasks.enrich_arxiv_source")
def enrich_arxiv_source(self, source_id: str, force: bool = False, limit: int = 500) -> Dict[str, Any]:
    return asyncio.run(_async_enrich_source(self, source_id, force, limit))


async def _async_enrich_source(task, source_id: str, force: bool, limit: int) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            src = await db.get(DocumentSource, UUID(source_id))
            if not src or src.source_type != "arxiv":
                return {"success": False, "source_id": source_id, "error": "Source not found or not arXiv"}

            result = await db.execute(
                select(Document.id).where(Document.source_id == src.id).order_by(Document.created_at.desc()).limit(limit)
            )
            doc_ids = [str(r[0]) for r in result.all()]
            for did in doc_ids:
                enrich_arxiv_document.delay(did, force)

            return {"success": True, "source_id": source_id, "queued": len(doc_ids)}
        except Exception as exc:
            logger.error(f"Source enrichment failed for {source_id}: {exc}")
            return {"success": False, "source_id": source_id, "error": str(exc)}

