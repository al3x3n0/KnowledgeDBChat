"""
Celery tasks for git branch comparison and summaries.
"""

import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID
from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import DocumentSource, GitBranchDiff
from app.services.git_service import GitService
from app.services.llm_service import UserLLMSettings
from app.utils.ingestion_state import (
    set_git_compare_task,
    get_git_compare_task,
    set_git_compare_cancel_flag,
    is_git_compare_cancelled,
    clear_git_compare_task,
)

git_service = GitService()


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
        logger.debug(f"Could not load user preferences for git compare task: {e}")
    return None


@celery_app.task(bind=True, name="app.tasks.git_compare_tasks.compare_git_branches")
def compare_git_branches(self, diff_id: str, user_id: Optional[str] = None) -> dict:
    return asyncio.run(_async_compare_git_branches(self, diff_id, user_id))


async def _async_compare_git_branches(task, diff_id: str, user_id: Optional[str] = None) -> dict:
    async with create_celery_session()() as db:
        # Load user settings for LLM provider preference
        user_settings = await _load_user_settings(db, user_id)

        diff = await db.get(GitBranchDiff, UUID(diff_id))
        if not diff:
            raise ValueError("Comparison job not found")
        diff.status = "running"
        diff.updated_at = datetime.utcnow()
        await db.commit()

        source = await db.get(DocumentSource, diff.source_id)
        if not source:
            diff.status = "failed"
            diff.error = "Document source not found"
            await db.commit()
            raise ValueError("Document source not found")

        options = diff.options or {}
        explain = bool(options.get("explain", True))
        try:
            if await is_git_compare_cancelled(diff_id):
                diff.status = "canceled"
                diff.error = "Canceled before start"
                diff.completed_at = datetime.utcnow()
                await db.commit()
                await clear_git_compare_task(diff_id)
                return {"canceled": True}

            compare_payload = await git_service.fetch_compare(
                source,
                diff.repository,
                diff.base_branch,
                diff.compare_branch,
            )
            summary = git_service.build_diff_summary(compare_payload)
            diff.diff_summary = summary

            if explain and not await is_git_compare_cancelled(diff_id):
                explanation = await git_service.generate_llm_summary(
                    diff.repository,
                    diff.base_branch,
                    diff.compare_branch,
                    summary,
                    user_settings=user_settings,
                )
                diff.llm_summary = explanation

            if await is_git_compare_cancelled(diff_id):
                diff.status = "canceled"
                diff.error = "Canceled during processing"
            else:
                diff.status = "completed"
                diff.error = None
            diff.completed_at = datetime.utcnow()
            diff.updated_at = datetime.utcnow()
            await db.commit()
            await clear_git_compare_task(diff_id)
            return {"success": diff.status == "completed"}
        except Exception as exc:
            logger.exception(f"Git compare failed for job {diff_id}: {exc}")
            diff.status = "failed"
            diff.error = str(exc)
            diff.completed_at = datetime.utcnow()
            await db.commit()
            await clear_git_compare_task(diff_id)
            raise
