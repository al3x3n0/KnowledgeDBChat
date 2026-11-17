"""What is waiting in the Celery broker, read without consuming it.

The stalled-job sweep fails a pending job that no worker has claimed after
three requeues, taking it for a message the broker lost. A job waiting its
turn behind busy workers looks exactly the same from the database -- pending,
no task id, old -- and a paired eval of 30 trials on a worker with two slots
lost 19 of them that way: every job not started within ~90 minutes was
failed while its message sat in the queue. Only the broker can tell the two
apart, so the sweep asks it.

Read from Redis as kombu's Redis transport stores it: each queue is a list
(``<queue>`` and, with priorities, ``<queue>\\x06\\x16<n>``) of JSON messages
whose ``headers.task`` names the task and whose ``body`` is the base64 JSON
``[args, kwargs, embed]``; a message a worker has reserved but not finished
acknowledging is in the ``unacked`` hash as ``[message, exchange, key]``.
Any other broker answers ``None``: unknown, never "empty".
"""

import base64
import json
import time
from typing import Any, Iterable, Optional, Set

from loguru import logger

from app.core.config import settings

#: kombu's separator between a queue's name and its priority.
PRIORITY_SEPARATOR = "\x06\x16"
UNACKED_KEY = "unacked"


def _first_arg(message: Any, task_name: str) -> Optional[str]:
    if not isinstance(message, dict):
        return None
    headers = message.get("headers") or {}
    if headers.get("task") != task_name:
        return None
    try:
        body = message.get("body")
        if (message.get("properties") or {}).get("body_encoding") == "base64":
            body = base64.b64decode(body)
        args = json.loads(body)[0]
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return str(args[0]) if args else None


def _decode(raw: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def queued_first_args(client: Any, queue: str, task_name: str) -> Set[str]:
    """First argument of every ``task_name`` message queued or reserved."""
    found: Set[str] = set()
    keys: Iterable[Any] = [queue, *client.scan_iter(f"{queue}{PRIORITY_SEPARATOR}*")]
    for key in keys:
        for raw in client.lrange(key, 0, -1):
            arg = _first_arg(_decode(raw), task_name)
            if arg:
                found.add(arg)
    for raw in client.hvals(UNACKED_KEY):
        entry = _decode(raw)
        message = entry[0] if isinstance(entry, list) and entry else None
        arg = _first_arg(message, task_name)
        if arg:
            found.add(arg)
    return found


def message_presence(
    client: Any,
    queue: str,
    task_name: str,
    job_ids: Set[str],
    *,
    max_messages: int = 10_000,
    max_seconds: float = 2.0,
) -> dict[str, Optional[bool]]:
    """Bounded lookup: preserve positive sightings, return None if incomplete."""
    result = {str(job_id): None for job_id in job_ids}
    if not result or max_messages <= 0 or max_seconds <= 0:
        return result
    deadline = time.monotonic() + max_seconds
    examined = 0

    class Incomplete(Exception):
        pass

    def check_budget():
        if examined >= max_messages or time.monotonic() >= deadline:
            raise Incomplete()

    def inspect(raw, *, reserved=False):
        nonlocal examined
        check_budget()
        examined += 1
        value = _decode(raw)
        message = value
        if reserved:
            message = value[0] if isinstance(value, list) and value else None
        job_id = _first_arg(message, task_name)
        if job_id in result:
            result[job_id] = True
        return all(value is True for value in result.values())

    def queue_keys():
        yield queue
        cursor, seen = 0, set()
        while True:
            check_budget()
            cursor, keys = client.scan(
                cursor=cursor, match=f"{queue}{PRIORITY_SEPARATOR}*", count=256
            )
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    yield key
            if not cursor:
                break

    try:
        for key in queue_keys():
            start = 0
            while True:
                check_budget()
                size = min(256, max_messages - examined)
                page = client.lrange(key, start, start + size - 1)
                for raw in page:
                    if inspect(raw):
                        return result
                if len(page) < size:
                    break
                start += len(page)
        cursor = 0
        while True:
            check_budget()
            cursor, page = client.hscan(UNACKED_KEY, cursor=cursor, count=256)
            for raw in page.values():
                if inspect(raw, reserved=True):
                    return result
            if not cursor:
                break
    except Incomplete:
        return result
    except Exception as exc:
        logger.warning(f"Could not finish checking broker messages: {exc}")
        return result
    return {job_id: bool(present) for job_id, present in result.items()}


def agent_job_message_presence(job_ids: Set[str]) -> dict[str, Optional[bool]]:
    """Check only this poll batch, without consuming messages or reading whole lists."""
    unknown = {str(job_id): None for job_id in job_ids}
    url = str(settings.CELERY_BROKER_URL or "")
    if not unknown or not url.startswith(("redis://", "rediss://")):
        return unknown
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        try:
            return message_presence(
                client,
                settings.AGENT_JOB_CELERY_QUEUE,
                "app.tasks.agent_job_tasks.execute_agent_job_task",
                set(unknown),
            )
        finally:
            client.close()
    except Exception as exc:
        logger.warning(f"Could not check the broker's agent messages: {exc}")
        return unknown


def waiting_agent_job_ids() -> Optional[Set[str]]:
    """Ids of agent jobs with a message in the broker; None when unknown."""
    url = str(settings.CELERY_BROKER_URL or "")
    if not url.startswith(("redis://", "rediss://")):
        return None
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=5, socket_timeout=10)
        try:
            return queued_first_args(
                client,
                settings.AGENT_JOB_CELERY_QUEUE,
                "app.tasks.agent_job_tasks.execute_agent_job_task",
            )
        finally:
            client.close()
    except Exception as exc:
        logger.warning(f"Could not read the broker's agent queue: {exc}")
        return None


__all__ = [
    "queued_first_args",
    "waiting_agent_job_ids",
    "message_presence",
    "agent_job_message_presence",
]
