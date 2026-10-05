"""Forwarding one job's progress from Redis to a WebSocket.

Five handlers did this by hand -- presentations, repository reports, training
jobs, template jobs and workflow executions -- and each got a different part
of it wrong, in ways that only show under conditions a quick manual test does
not create:

* Two never released their Redis connection when the client went away or the
  job had already finished: the cleanup sat after the loop, not in a
  ``finally``. One leaked connection per closed tab.
* Three waited on ``pubsub.listen()`` and nothing else, so they could not
  notice a client leaving. A socket watching a job that never finishes held
  its handler and its connection until the server restarted.
* One used the *blocking* Redis client inside the event loop, polling with a
  one-second timeout: while anyone watched a template job, every other
  request the server was handling stalled for up to a second at a time.

This is the one loop. A handler decides who may watch and what the first
message says; everything after that is here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

#: How long one wait on Redis lasts before the client is checked on.
POLL_SECONDS = 1.0

#: How long the client is listened to between Redis polls. Long enough to
#: notice a close or a ping, short enough not to delay the next update.
CLIENT_CHECK_SECONDS = 0.05

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def status_is_terminal(payload: Dict[str, Any]) -> bool:
    """The usual rule: a message whose status says the job is over."""
    return payload.get("status") in TERMINAL_STATUSES


def _redis_client(url: str) -> Any:
    import redis.asyncio as redis

    return redis.from_url(url, decode_responses=True)


async def _client_is_gone(websocket: WebSocket) -> bool:
    """Listen to the client briefly; answer a ping; say whether it has left."""
    try:
        message = await asyncio.wait_for(
            websocket.receive_text(), timeout=CLIENT_CHECK_SECONDS
        )
    except asyncio.TimeoutError:
        return False
    except WebSocketDisconnect:
        return True
    except RuntimeError:
        # Starlette raises this for a receive on a socket that has closed.
        return True
    if message == "ping":
        await websocket.send_text("pong")
    return False


async def forward_progress(
    websocket: WebSocket,
    channel: str,
    *,
    is_terminal: Callable[[Dict[str, Any]], bool] = status_is_terminal,
    initial: Optional[Dict[str, Any]] = None,
    already_finished: bool = False,
    redis_url: Optional[str] = None,
    redis_factory: Callable[[str], Any] = _redis_client,
) -> None:
    """Send ``initial``, then every message on ``channel`` until it is over.

    The socket must already be accepted and the caller authorised. Returns
    when a terminal message has been forwarded, the client has gone, or the
    job was ``already_finished``; the Redis connection is released and the
    socket closed in every one of those cases, and in the case of an error.

    Subscribes *before* sending the initial message, so an update published
    between the caller reading the job and this function starting is not
    lost.
    """
    if redis_url is None:
        from app.core.config import settings

        redis_url = settings.REDIS_URL

    client = None
    pubsub = None
    try:
        client = redis_factory(redis_url)
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)

        if initial is not None:
            await websocket.send_json(initial)
        if already_finished:
            return

        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=POLL_SECONDS
            )
            if message and message.get("type") == "message":
                data = message.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8", "replace")
                await websocket.send_text(str(data))
                try:
                    payload = json.loads(data)
                except (TypeError, ValueError):
                    payload = None
                if isinstance(payload, dict) and is_terminal(payload):
                    return
            if await _client_is_gone(websocket):
                return
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.error(f"Progress stream {channel} failed: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)[:300]})
        except Exception:
            pass
    finally:
        # Each step on its own: a failure to unsubscribe must not skip closing
        # the connection, which is the part that leaks.
        if pubsub is not None:
            try:
                await pubsub.unsubscribe(channel)
            except Exception:
                pass
            try:
                await pubsub.close()
            except Exception:
                pass
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
