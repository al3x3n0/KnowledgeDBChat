"""
Background task to summarize a document.
"""

import asyncio
from typing import Dict, Any
from uuid import UUID
from loguru import logger

from app.core.celery import celery_app
import json
import redis
from app.core.config import settings
from app.core.database import create_celery_session
from app.services.document_service import DocumentService


@celery_app.task(bind=True, name="app.tasks.summarization_tasks.summarize_document")
def summarize_document(self, document_id: str, force: bool = False) -> Dict[str, Any]:
    return asyncio.run(_async_summarize_document(self, document_id, force))


async def _async_summarize_document(task, document_id: str, force: bool) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            logger.info(f"Summarizing document {document_id}, force={force}")
            # Mark document as summarizing and publish status
            try:
                from sqlalchemy import select
                from app.models.document import Document as _Document
                result = await db.execute(select(_Document).where(_Document.id == UUID(document_id)))
                doc = result.scalar_one_or_none()
                if doc:
                    meta = doc.extra_metadata or {}
                    meta.update({"is_summarizing": True})
                    doc.extra_metadata = meta
                    await db.commit()
                    _publish_sum_status(document_id, {"is_summarizing": True})
            except Exception as e:
                logger.debug(f"Unable to set is_summarizing flag pre-run: {e}")
            svc = DocumentService()
            _publish_sum_progress(document_id, {"stage": "start", "progress": 0})
            summary = await svc.summarize_document(UUID(document_id), db, force=force)
            _publish_sum_complete(document_id, {"summary_length": len(summary or '')})
            # Clear summarizing flag and broadcast
            try:
                from sqlalchemy import select
                from app.models.document import Document as _Document
                result = await db.execute(select(_Document).where(_Document.id == UUID(document_id)))
                doc = result.scalar_one_or_none()
                if doc:
                    meta = doc.extra_metadata or {}
                    meta.update({"is_summarizing": False})
                    doc.extra_metadata = meta
                    await db.commit()
                    _publish_sum_status(document_id, {"is_summarizing": False})
            except Exception as e:
                logger.debug(f"Unable to clear is_summarizing flag: {e}")
            return {"success": True, "document_id": document_id, "summary": summary}
        except Exception as e:
            logger.error(f"Summarization failed for {document_id}: {e}")
            _publish_sum_error(document_id, str(e))
            # Clear summarizing flag and broadcast error status
            try:
                from sqlalchemy import select
                from app.models.document import Document as _Document
                result = await db.execute(select(_Document).where(_Document.id == UUID(document_id)))
                doc = result.scalar_one_or_none()
                if doc:
                    meta = doc.extra_metadata or {}
                    meta.update({"is_summarizing": False})
                    doc.extra_metadata = meta
                    await db.commit()
                    _publish_sum_status(document_id, {"is_summarizing": False, "failed": True})
            except Exception as e2:
                logger.debug(f"Unable to clear is_summarizing flag after error: {e2}")
            return {"success": False, "document_id": document_id, "error": str(e)}


def _get_redis_client():
    try:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for summarization progress: {e}")
        return None


def _publish_sum_progress(document_id: str, progress: dict):
    try:
        client = _get_redis_client()
        if client:
            channel = f"summarization_progress:{document_id}"
            msg = json.dumps({
                "type": "progress",
                "document_id": document_id,
                "progress": progress,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish summarization progress: {e}")


def _publish_sum_complete(document_id: str, result: dict):
    try:
        client = _get_redis_client()
        if client:
            channel = f"summarization_progress:{document_id}"
            msg = json.dumps({
                "type": "complete",
                "document_id": document_id,
                "result": result,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish summarization complete: {e}")


def _publish_sum_error(document_id: str, error: str):
    try:
        client = _get_redis_client()
        if client:
            channel = f"summarization_progress:{document_id}"
            msg = json.dumps({
                "type": "error",
                "document_id": document_id,
                "error": error,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish summarization error: {e}")


def _publish_sum_status(document_id: str, status: dict):
    try:
        client = _get_redis_client()
        if client:
            channel = f"summarization_progress:{document_id}"
            msg = json.dumps({
                "type": "status",
                "document_id": document_id,
                "status": status,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish summarization status: {e}")
