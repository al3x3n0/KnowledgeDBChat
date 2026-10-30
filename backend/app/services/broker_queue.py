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


__all__ = ["queued_first_args", "waiting_agent_job_ids"]
