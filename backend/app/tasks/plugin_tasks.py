"""Drafting a plugin manifest off the request thread.

Drafting is slow for a reason that is not going away: the draft is validated
against the real validator and its tools are actually run, and when either
refuses, the model is asked again. Measured, a first-time-right draft took
about twenty seconds and one needing two repairs took two minutes -- and the
repair loop is precisely what makes the output worth having, so the slow case
is the *common* one whenever a first draft is wrong.

Held open on an HTTP request that is a connection tied up for two minutes, a
proxy timeout waiting to happen, and a spinner that says nothing while the
interesting part -- "this draft was refused because the id had a hyphen, trying
again" -- is known and thrown away.

No new table. A draft is ephemeral: you review it and either create a plugin or
discard it, and rows for every discarded attempt would accumulate for nothing.
Celery's own result backend already holds exactly this shape of state, and
`update_state` exists to carry the attempt-by-attempt detail.
"""

import asyncio
from typing import Any, Dict, List

from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.user import User

#: Drafting makes up to MAX_ATTEMPTS model calls and runs the tools it writes.
#: The ceiling is generous against the two minutes observed, and finite so a
#: wedged provider cannot hold a worker for ever.
SOFT_LIMIT_SECONDS = 300
HARD_LIMIT_SECONDS = 360


@celery_app.task(
    bind=True,
    name="app.tasks.plugin_tasks.draft_plugin_manifest",
    soft_time_limit=SOFT_LIMIT_SECONDS,
    time_limit=HARD_LIMIT_SECONDS,
)
def draft_plugin_manifest(self, description: str, user_id: str) -> Dict[str, Any]:
    """Draft a manifest for one user, reporting each attempt as it goes."""

    def report(stage: str, attempt: int, notes: List[str]) -> None:
        self.update_state(
            state="PROGRESS",
            meta={
                # Carried on every update so the polling endpoint can refuse a
                # draft that is not the caller's. A task id is unguessable, but
                # unguessable is not the same as checked.
                "user_id": str(user_id),
                "stage": stage,
                "attempt": attempt,
                "notes": list(notes),
            },
        )

    async def _run() -> Dict[str, Any]:
        from app.services.plugin_author_service import draft_manifest

        # A fresh engine per invocation: workers fork, and an engine bound to
        # the parent's event loop does not survive it. This returns a session
        # *factory*, not a session.
        session_factory = create_celery_session()
        async with session_factory() as db:
            user = (
                await db.execute(select(User).where(User.id == user_id))
            ).scalar_one_or_none()
            if user is None:
                return {
                    "user_id": str(user_id),
                    "manifest": None,
                    "notes": ["The user this draft belongs to no longer exists."],
                    "attempts": 0,
                }
            result = await draft_manifest(
                description,
                user=user,
                db=db,
                user_id=user.id,
                on_progress=report,
            )
            return {"user_id": str(user_id), **result}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error(f"Plugin draft task failed: {exc}")
        # Returned rather than raised. A failure here is a fact about the
        # draft -- the caller wants to see why, and a Celery FAILURE state
        # carries a traceback nobody reading a manifest editor can use.
        return {
            "user_id": str(user_id),
            "manifest": None,
            "notes": [f"Drafting failed: {str(exc)[:300]}"],
            "attempts": 0,
        }
