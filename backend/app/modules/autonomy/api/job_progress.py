"""WebSocket boundary for autonomous-job progress streaming."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

import redis.asyncio as redis
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from loguru import logger
from sqlalchemy import and_, select

from app.core.config import settings
from app.models.agent_job import AgentJob

Authenticator = Callable[[str], Awaitable[Any]]
SessionFactory = Callable[[], Any]
RedisFactory = Callable[[str], Any]
WaitFor = Callable[..., Awaitable[Any]]

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class JobProgressApi:
    router: APIRouter
    agent_job_progress_websocket: Callable[..., Any]


async def _authenticate_token(token: str) -> Any:
    from app.api.endpoints.auth import get_user_from_token

    return await get_user_from_token(token)


def _session_factory() -> Any:
    from app.core.database import async_session_factory

    return async_session_factory()


def _redis_factory(url: str) -> Any:
    return redis.from_url(url)


def build_job_progress_api(
    *,
    authenticate_token: Authenticator = _authenticate_token,
    session_factory: SessionFactory = _session_factory,
    redis_factory: RedisFactory = _redis_factory,
    redis_url: str = settings.REDIS_URL,
    wait_for: WaitFor = asyncio.wait_for,
) -> JobProgressApi:
    """Build the progress socket with infrastructure edges injected."""
    router = APIRouter()

    @router.websocket("/{job_id}/progress")
    async def agent_job_progress_websocket(
        websocket: WebSocket,
        job_id: str,
        token: str = Query(...),
    ) -> None:
        try:
            user = await authenticate_token(token)
            if not user:
                await websocket.close(code=4001, reason="Invalid token")
                return
        except Exception:
            await websocket.close(code=4001, reason="Authentication failed")
            return

        async with session_factory() as db:
            result = await db.execute(
                select(AgentJob).where(
                    and_(AgentJob.id == UUID(job_id), AgentJob.user_id == user.id)
                )
            )
            job = result.scalar_one_or_none()
            if job is None:
                await websocket.accept()
                await websocket.send_json({"type": "error", "error": "Job not found"})
                await websocket.close(code=4004, reason="Job not found")
                return

        await websocket.accept()
        await websocket.send_json(
            {
                "type": "connected",
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
            }
        )

        redis_client = None
        pubsub = None
        channel = f"agent_job:{job_id}:progress"
        try:
            redis_client = redis_factory(redis_url)
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(channel)
            logger.info(f"WebSocket subscribed to {channel}")

            while True:
                try:
                    try:
                        message = await wait_for(
                            websocket.receive_text(),
                            timeout=0.1,
                        )
                        if message == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        pass

                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if message and message["type"] == "message":
                        payload = json.loads(message["data"])
                        await websocket.send_json(payload)
                        if payload.get("status") in TERMINAL_STATUSES:
                            logger.info(f"Job {job_id} finished, closing WebSocket")
                            break
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for job {job_id}")
                    break
                except Exception as error:
                    logger.error(f"Error in WebSocket loop: {error}")
                    break
        except Exception as error:
            logger.error(f"WebSocket error for job {job_id}: {error}")
            try:
                await websocket.send_json({"type": "error", "error": str(error)})
            except Exception:
                pass
        finally:
            if pubsub:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                except Exception:
                    pass
            if redis_client:
                try:
                    await redis_client.close()
                except Exception:
                    pass
            try:
                await websocket.close()
            except Exception:
                pass

    return JobProgressApi(
        router=router,
        agent_job_progress_websocket=agent_job_progress_websocket,
    )
