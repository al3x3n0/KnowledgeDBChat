"""
Research-related background tasks (literature review generation, etc.).
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document, DocumentSource
from app.services.document_service import DocumentService
from app.services.llm_service import UserLLMSettings
from app.tasks.ingestion_tasks import process_uploaded_document


async def _load_user_settings(db, user_id: Optional[str]) -> Optional[UserLLMSettings]:
    """Load user LLM settings from preferences."""
    if not user_id:
        return None
    try:
        from app.models.memory import UserPreferences
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == UUID(user_id))
        )
        user_prefs = result.scalar_one_or_none()
        if user_prefs:
            return UserLLMSettings.from_preferences(user_prefs)
    except Exception as e:
        logger.debug(f"Could not load user preferences for research task: {e}")
    return None


@celery_app.task(bind=True, name="app.tasks.research_tasks.generate_literature_review")
def generate_literature_review(self, source_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    return asyncio.run(_async_generate_literature_review(self, source_id, user_id))


async def _async_generate_literature_review(task, source_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            # Load user settings for LLM provider preference
            user_settings = await _load_user_settings(db, user_id)

            src = await db.get(DocumentSource, UUID(source_id))
            if not src:
                raise ValueError("Source not found")
            if src.source_type != "arxiv":
                raise ValueError("Literature review only supported for arXiv sources")

            result = await db.execute(select(Document).where(Document.source_id == src.id).order_by(Document.created_at.desc()))
            docs = list(result.scalars().all())
            if not docs:
                return {"success": False, "source_id": source_id, "error": "No documents found for source"}

            cfg = src.config or {}
            topic = None
            if isinstance(cfg, dict):
                topic = cfg.get("topic")
            topic = topic or src.name.replace("#", "").strip()

            svc = DocumentService()

            papers: List[Dict[str, Any]] = []
            for d in docs[:25]:
                meta = d.extra_metadata or {}
                insights = meta.get("paper_insights") if isinstance(meta, dict) else None
                papers.append(
                    {
                        "title": d.title,
                        "url": d.url,
                        "author": d.author,
                        "summary": d.summary,
                        "abstract": (d.content or "")[:2000],
                        "insights": insights,
                        "arxiv_id": d.source_identifier.split("/")[-1] if d.source_identifier else None,
                    }
                )

            # Build a compact prompt with structured hints.
            prompt = (
                "Write a literature review in Markdown based on the provided paper list.\n"
                "Requirements:\n"
                "- Start with a 6-10 sentence overview.\n"
                "- Include a comparison table with columns: Paper, Key claims, Methods, Limitations, Link.\n"
                "- Include a bullet list of open questions / future work.\n"
                "- Cite papers by short title and arXiv id.\n"
                "Keep it concise and practical.\n\n"
                f"Topic: {topic}\n\n"
                f"Papers JSON:\n{json.dumps(papers, ensure_ascii=False)}\n"
            )

            report_md = await svc.llm.generate_response(
                query=prompt,
                context=None,
                temperature=0.2,
                max_tokens=1200,
                task_type="summarization",
                user_settings=user_settings,
            )

            title = f"Literature Review: {topic}"
            content = f"# {title}\n\n{report_md.strip()}\n"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            review_doc = Document(
                title=title,
                content=content,
                content_hash=content_hash,
                url=None,
                file_type="md",
                source_id=src.id,
                source_identifier=f"literature_review:{source_id}",
                author="research_assistant",
                tags=["literature_review", "arxiv"],
                extra_metadata={
                    "report_type": "literature_review",
                    "source_id": source_id,
                    "topic": topic,
                    "papers_count": len(docs),
                },
                is_processed=False,
            )
            db.add(review_doc)
            await db.commit()
            await db.refresh(review_doc)

            process_uploaded_document.delay(str(review_doc.id))

            return {"success": True, "source_id": source_id, "document_id": str(review_doc.id), "title": title}
        except Exception as e:
            logger.error(f"Literature review generation failed for {source_id}: {e}")
            return {"success": False, "source_id": source_id, "error": str(e)}
