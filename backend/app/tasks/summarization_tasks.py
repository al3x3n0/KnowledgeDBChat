"""
Background task to summarize a document.
"""

import asyncio
from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.services import llm_json
from app.services.document_service import DocumentService
from app.services.llm_service import UserLLMSettings, load_user_llm_settings
from app.tasks import job_support


async def _load_user_settings(db, user_id: Optional[str]) -> Optional[UserLLMSettings]:
    """Load user LLM settings from preferences."""
    return await load_user_llm_settings(db, user_id)


@celery_app.task(bind=True, name="app.tasks.summarization_tasks.summarize_document")
def summarize_document(
    self, document_id: str, force: bool = False, user_id: Optional[str] = None
) -> Dict[str, Any]:
    return asyncio.run(_async_summarize_document(self, document_id, force, user_id))


async def _async_summarize_document(
    task, document_id: str, force: bool, user_id: Optional[str] = None
) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            logger.info(
                f"Summarizing document {document_id}, force={force}, user_id={user_id}"
            )

            # Load user settings for LLM provider preference
            user_settings = await _load_user_settings(db, user_id)

            # Mark document as summarizing and publish status
            try:
                from sqlalchemy import select

                from app.models.document import Document as _Document

                result = await db.execute(
                    select(_Document).where(_Document.id == UUID(document_id))
                )
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
            summary = await svc.summarize_document(
                UUID(document_id), db, force=force, user_settings=user_settings
            )

            # If this is an arXiv paper, extract structured insights.
            try:
                from sqlalchemy import select

                from app.models.document import Document as _Document
                from app.models.document import DocumentSource as _Source

                result = await db.execute(
                    select(_Document).where(_Document.id == UUID(document_id))
                )
                doc = result.scalar_one_or_none()
                if doc:
                    src = await db.get(_Source, doc.source_id)
                    if src and getattr(src, "source_type", None) == "arxiv":
                        insights = await _extract_paper_insights(
                            svc, doc, user_settings=user_settings
                        )
                        if insights:
                            meta = doc.extra_metadata or {}
                            from datetime import datetime as _dt

                            meta["paper_insights"] = {
                                **insights,
                                "extracted_at": _dt.utcnow().isoformat(),
                            }
                            doc.extra_metadata = meta
                            await db.commit()
                            try:
                                from app.tasks.paper_kg_tasks import (
                                    upsert_paper_insights_to_kg,
                                )

                                upsert_paper_insights_to_kg.delay(
                                    document_id, force=True
                                )
                            except Exception:
                                pass
            except Exception as e:
                logger.debug(
                    f"Paper insights extraction skipped for {document_id}: {e}"
                )

            _publish_sum_complete(document_id, {"summary_length": len(summary or "")})
            # Clear summarizing flag and broadcast
            try:
                from sqlalchemy import select

                from app.models.document import Document as _Document

                result = await db.execute(
                    select(_Document).where(_Document.id == UUID(document_id))
                )
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

                result = await db.execute(
                    select(_Document).where(_Document.id == UUID(document_id))
                )
                doc = result.scalar_one_or_none()
                if doc:
                    meta = doc.extra_metadata or {}
                    meta.update({"is_summarizing": False})
                    doc.extra_metadata = meta
                    await db.commit()
                    _publish_sum_status(
                        document_id, {"is_summarizing": False, "failed": True}
                    )
            except Exception as e2:
                logger.debug(f"Unable to clear is_summarizing flag after error: {e2}")
            return {"success": False, "document_id": document_id, "error": str(e)}


async def _extract_paper_insights(
    svc: DocumentService, doc: Any, user_settings=None
) -> Dict[str, Any] | None:
    title = getattr(doc, "title", "") or ""
    content = getattr(doc, "content", "") or ""
    summary = getattr(doc, "summary", "") or ""

    prompt = (
        "Extract structured research-paper insights. Return ONLY valid JSON with keys:\n"
        "- key_claims: string[] (3-8 concise claims)\n"
        "- methods: string[] (main techniques)\n"
        "- limitations: string[] (3-8)\n"
        "- datasets: string[] (if mentioned)\n"
        "- metrics: string[] (if mentioned)\n"
        "- tasks: string[] (if mentioned)\n"
        "- takeaways: string[] (3-8 actionable)\n\n"
        f"Title: {title}\n\n"
        f"Summary (if available):\n{summary}\n\n"
        f"Paper content (may include abstract):\n{content[:12000]}\n"
    )
    try:
        raw = await svc.llm.generate_response(
            query=prompt,
            context=None,
            temperature=0.1,
            max_tokens=600,
            task_type="summarization",
            user_settings=user_settings,
        )
        # best-effort parse (allow surrounding text)
        return llm_json.extract_json_object(raw)
    except Exception as e:
        logger.debug(f"Failed to extract paper insights: {e}")
        return None


def _publish_sum_progress(document_id: str, progress: dict):
    job_support.publish_sync(
        f"summarization_progress:{document_id}",
        {"type": "progress", "document_id": document_id, "progress": progress},
    )


def _publish_sum_complete(document_id: str, result: dict):
    job_support.publish_sync(
        f"summarization_progress:{document_id}",
        {"type": "complete", "document_id": document_id, "result": result},
    )


def _publish_sum_error(document_id: str, error: str):
    job_support.publish_sync(
        f"summarization_progress:{document_id}",
        {"type": "error", "document_id": document_id, "error": error},
    )


def _publish_sum_status(document_id: str, status: dict):
    job_support.publish_sync(
        f"summarization_progress:{document_id}",
        {"type": "status", "document_id": document_id, "status": status},
    )
