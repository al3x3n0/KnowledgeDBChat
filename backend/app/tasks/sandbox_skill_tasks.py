"""Sandbox-skill work that does not belong on a request.

Two tasks, both slow for reasons that are not going away.

Drafting calls a model up to three times and runs the skill's control in the
sandbox between attempts -- the repair loop is what makes the draft worth
having, so the slow case is the common one. Like plugin drafting it keeps no
table: a draft is reviewed and then created or discarded, and Celery's result
backend already holds exactly that shape of state.

Building an image runs `docker build`, which downloads packages and can take
many minutes. Its state *is* a row, because an image outlives the request, the
worker and the person who proposed it.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session

DRAFT_SOFT_LIMIT_SECONDS = 600
DRAFT_HARD_LIMIT_SECONDS = 660


@celery_app.task(
    bind=True,
    name="app.tasks.sandbox_skill_tasks.draft_sandbox_skill",
    soft_time_limit=DRAFT_SOFT_LIMIT_SECONDS,
    time_limit=DRAFT_HARD_LIMIT_SECONDS,
)
def draft_sandbox_skill(
    self,
    description: str,
    user_id: str,
    current: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Draft a skill for one user, reporting each attempt as it goes."""

    def report(stage: str, attempt: int, notes: List[str]) -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                # Carried on every update so the polling endpoint can refuse a
                # draft that is not the caller's.
                "user_id": str(user_id),
                "stage": stage,
                "attempt": attempt,
                "notes": list(notes),
            },
        )

    async def _run() -> Dict[str, Any]:
        from app.services.sandbox_skill_author_service import draft_skill

        # A fresh engine per invocation: workers fork, and an engine bound to
        # the parent's event loop does not survive it.
        session_factory = create_celery_session()
        async with session_factory() as db:
            result = await draft_skill(
                description,
                db=db,
                user_id=user_id,
                current=current,
                on_progress=report,
            )
            return {"user_id": str(user_id), **result}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error(f"Sandbox skill draft task failed: {exc}")
        # Returned rather than raised: the caller wants the reason, and a
        # Celery FAILURE carries a traceback nobody in an editor can use.
        return {
            "user_id": str(user_id),
            "manifest": None,
            "notes": [f"Drafting failed: {str(exc)[:300]}"],
            "attempts": 0,
            "dry_run": None,
        }


@celery_app.task(
    name="app.tasks.sandbox_skill_tasks.build_sandbox_skill_image",
    # The build enforces its own timeout and records the outcome; these are
    # only the backstop for a worker that stops answering.
    soft_time_limit=3600,
    time_limit=3660,
)
def build_sandbox_skill_image(image_id: str) -> Dict[str, Any]:
    """Build one approved image and record how it went."""

    async def _run() -> Dict[str, Any]:
        from app.services import sandbox_skill_image_service

        session_factory = create_celery_session()
        async with session_factory() as db:
            image = await sandbox_skill_image_service.get_image(db, image_id)
            if image is None:
                return {"image_id": image_id, "status": "missing"}
            image = await sandbox_skill_image_service.build_image(db, image)
            return {"image_id": image_id, "status": image.status, "tag": image.tag}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error(f"Sandbox skill image build task failed: {exc}")
        return {"image_id": image_id, "status": "failed", "error": str(exc)[:300]}
