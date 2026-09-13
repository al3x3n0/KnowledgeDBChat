"""Finalize autonomous R&D evaluation launches once their trials settle."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from celery.exceptions import MaxRetriesExceededError
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.autonomous_rnd_eval_launch import (
    EVAL_LAUNCH_STATUS_FAILED,
    EVAL_LAUNCH_STATUS_RUNNING,
    AutonomousRndEvalLaunch,
)
from app.services.autonomous_rnd_eval_launch_service import (
    autonomous_rnd_eval_launch_service,
)

# Trials are ordinary agent jobs with their own runtime caps, so polling is
# cheap relative to the work being waited on.
POLL_INTERVAL_SECONDS = 60
MAX_POLL_ATTEMPTS = 240  # ~4 hours before a launch is abandoned as stalled


@celery_app.task(
    bind=True,
    name="app.tasks.autonomous_rnd_eval_tasks.finalize_autonomous_rnd_eval_launch",
    max_retries=MAX_POLL_ATTEMPTS,
    default_retry_delay=POLL_INTERVAL_SECONDS,
)
def finalize_autonomous_rnd_eval_launch(self, launch_id: str) -> Dict[str, Any]:
    """Grade a launch when every trial has settled, retrying while they run."""
    result = asyncio.run(_async_finalize(launch_id))
    if not result.get("retry"):
        return result
    try:
        raise self.retry(countdown=POLL_INTERVAL_SECONDS)
    except MaxRetriesExceededError:
        # Out of polling attempts: settle the launch instead of leaving it
        # running forever, so the stall is visible rather than silent.
        return asyncio.run(
            _async_abandon(
                launch_id,
                "Trials did not reach a terminal state within "
                f"{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SECONDS // 60} minutes",
            )
        )


async def _async_finalize(launch_id: str) -> Dict[str, Any]:
    try:
        launch_uuid = UUID(str(launch_id))
    except (TypeError, ValueError):
        return {"finalized": False, "reason": "invalid_launch_id"}

    async with create_celery_session()() as db:
        launch = await db.get(AutonomousRndEvalLaunch, launch_uuid)
        if launch is None:
            return {"finalized": False, "reason": "launch_missing"}
        if launch.status != EVAL_LAUNCH_STATUS_RUNNING:
            return {
                "finalized": launch.run_id is not None,
                "reason": "already_settled",
                "status": launch.status,
            }

        run = await autonomous_rnd_eval_launch_service.finalize(db, launch=launch)
        if run is None and launch.status == EVAL_LAUNCH_STATUS_RUNNING:
            await db.rollback()
            return {"finalized": False, "retry": True, "reason": "trials_running"}

        await db.commit()
        if run is None:
            logger.warning(
                "Autonomous R&D eval launch {} could not be graded: {}",
                launch_id,
                launch.error,
            )
            return {
                "finalized": False,
                "reason": "grading_failed",
                "status": launch.status,
            }
        logger.info(
            "Autonomous R&D eval launch {} graded as run {} (pass_pow_k={})",
            launch_id,
            run.id,
            run.pass_pow_k,
        )
        return {
            "finalized": True,
            "run_id": str(run.id),
            "pass_pow_k": run.pass_pow_k,
            "pass_at_k": run.pass_at_k,
        }


@celery_app.task(
    name="app.tasks.autonomous_rnd_eval_tasks.abandon_stalled_eval_launch",
)
def abandon_stalled_eval_launch(launch_id: str, reason: str) -> Dict[str, Any]:
    """Mark a launch failed when its trials never settle."""
    return asyncio.run(_async_abandon(launch_id, reason))


async def _async_abandon(launch_id: str, reason: str) -> Dict[str, Any]:
    try:
        launch_uuid = UUID(str(launch_id))
    except (TypeError, ValueError):
        return {"abandoned": False, "reason": "invalid_launch_id"}

    async with create_celery_session()() as db:
        launch = await db.get(AutonomousRndEvalLaunch, launch_uuid)
        if launch is None or launch.status != EVAL_LAUNCH_STATUS_RUNNING:
            return {"abandoned": False, "reason": "not_running"}
        launch.status = EVAL_LAUNCH_STATUS_FAILED
        launch.error = str(reason)[:1000]
        launch.completed_at = datetime.now(timezone.utc)
        await db.commit()
        return {"abandoned": True, "launch_id": str(launch.id)}
