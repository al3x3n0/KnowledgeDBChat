"""
Tasks for converting extracted paper insights into knowledge graph nodes/edges.
"""

import asyncio
from typing import Any, Dict
from uuid import UUID

from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.services.paper_kg_service import PaperKnowledgeGraphService


@celery_app.task(bind=True, name="app.tasks.paper_kg_tasks.upsert_paper_insights_to_kg")
def upsert_paper_insights_to_kg(self, document_id: str, force: bool = False) -> Dict[str, Any]:
    return asyncio.run(_async_upsert(self, document_id, force))


async def _async_upsert(task, document_id: str, force: bool) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            svc = PaperKnowledgeGraphService()
            return await svc.upsert_from_document(db, UUID(document_id), force=force)
        except Exception as exc:
            logger.error(f"Paper→KG extraction failed for {document_id}: {exc}")
            return {"success": False, "document_id": document_id, "error": str(exc)}

