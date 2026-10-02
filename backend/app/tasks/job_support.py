"""What every job-shaped Celery task does around its actual work.

Async tasks report through :func:`publish_progress`; synchronous ones through
:func:`publish_sync`, which takes the message whole because those modules
publish several kinds (progress, status, segment, complete, error).

Export, presentation and repository-report tasks each carried their own copy
of two things: publishing a progress message to Redis, and marking the job
failed when the task raised. The copies were identical apart from the channel
name and the model, which is the kind of duplication that stays identical
only until one of them is fixed.

One of them needed fixing: each publisher closed its Redis connection on the
line after ``publish`` rather than in a ``finally``, so a publish that raised
left the connection open -- in a worker that publishes on every step.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, List, Mapping, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.core.database import create_celery_session

#: A job in one of these states is over; a late failure must not rewrite it.
FINISHED = ("completed", "failed", "cancelled")


def run_async(coroutine):
    """Run a coroutine to completion from synchronous task code.

    Four task modules carried this; three were identical and the fourth
    differed only in how it failed when a loop was already running, which a
    prefork worker never has.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # Fallback for unexpected contexts
        return asyncio.run(coroutine)
    return loop.run_until_complete(coroutine)


def attempt_reporter(task: Any, user_id: Any) -> Callable[[str, int, List[str]], None]:
    """The progress callback a drafting task hands to its repair loop.

    ``user_id`` is carried on every update so the polling endpoint can refuse
    a draft that is not the caller's. A task id is unguessable, but
    unguessable is not the same as checked.
    """

    def report(stage: str, attempt: int, notes: List[str]) -> None:
        task.update_state(
            state="PROGRESS",
            meta={
                "user_id": str(user_id),
                "stage": stage,
                "attempt": attempt,
                "notes": list(notes),
            },
        )

    return report


async def publish_message(channel: str, message: Mapping[str, Any]) -> None:
    """Publish one message for WebSocket subscribers.

    Never raises: progress is a courtesy to whoever is watching, and a Redis
    that is down must not fail the job it is reporting on.
    """
    import redis.asyncio as redis

    from app.core.config import settings

    client = None
    try:
        client = redis.from_url(settings.REDIS_URL)
        await client.publish(channel, json.dumps(message))
    except Exception as exc:
        logger.warning(f"Failed to publish progress on {channel}: {exc}")
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


async def publish_progress(
    channel: str,
    progress: int,
    stage: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """The plain progress message; a task with more to say builds its own and
    calls :func:`publish_message`."""
    message = {
        "type": "progress",
        "progress": progress,
        "stage": stage,
        "status": status,
    }
    if error:
        message["error"] = error
    await publish_message(channel, message)


def publish_sync(channel: str, message: Mapping[str, Any]) -> None:
    """Publish one message from synchronous task code. Never raises.

    Seven task modules had some twenty-five copies of this, one per message
    type, differing in the channel and the payload. Every one opened a client
    for the message and none closed it, so each progress tick left a
    connection pool for the garbage collector.
    """
    import redis

    from app.core.config import settings

    client = None
    try:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.publish(channel, json.dumps(message))
    except Exception as exc:
        logger.debug(f"Failed to publish on {channel}: {exc}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def flag_is_set(key: str) -> bool:
    """Whether a Redis key holds a value -- how a task polls for "cancel".

    False when Redis cannot be reached: a task that cannot ask keeps working.
    The callers of this each opened a client per poll (once per document, once
    per training step) and closed none of them.
    """
    import redis

    from app.core.config import settings

    client = None
    try:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return bool(client.get(key))
    except Exception as exc:
        logger.debug(f"Failed to read {key}: {exc}")
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def delete_keys(*keys: str) -> None:
    """Remove Redis keys a task left for itself. Never raises."""
    import redis

    from app.core.config import settings

    client = None
    try:
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.delete(*keys)
    except Exception as exc:
        logger.debug(f"Failed to delete {keys}: {exc}")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


async def mark_job_failed(model: Any, job_id: Any, error_text: str) -> bool:
    """Record that a task died, unless the job already reached an ending.

    Uses a fresh session: this runs from a task's ``except`` block, where the
    session the work was using may be the thing that broke. Returns whether
    the row was changed.
    """
    session_factory = create_celery_session()
    async with session_factory() as db:
        job = (
            await db.execute(select(model).where(model.id == UUID(str(job_id))))
        ).scalar_one_or_none()
        if job is None or job.status in FINISHED:
            return False
        job.status = "failed"
        job.error = f"Task error: {error_text}"
        job.completed_at = datetime.utcnow()
        await db.commit()
        return True
