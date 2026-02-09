"""
Background tasks for ad-hoc URL ingestion.
"""

import asyncio
import json
from typing import Any, Dict, Optional
from uuid import UUID

import redis
from celery import current_task
from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.config import settings
from app.core.database import create_celery_session
from app.models.user import User
from app.services.url_ingestion_service import UrlIngestionService


def _run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return asyncio.run(coroutine)
    return loop.run_until_complete(coroutine)


def _get_redis_client() -> Optional[redis.Redis]:
    try:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for URL ingestion progress: {e}")
        return None


def _publish(source_id: str, message_type: str, payload_key: str, payload: Any):
    try:
        client = _get_redis_client()
        if not client:
            return
        channel = f"ingestion_progress:{source_id}"
        msg = json.dumps(
            {
                "type": message_type,
                "document_id": source_id,
                payload_key: payload,
            }
        )
        client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish URL ingestion progress: {e}")


@celery_app.task(bind=True, name="app.tasks.url_ingestion_tasks.ingest_url")
def ingest_url(self, request: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Ingest a URL in the background and publish progress on Redis pubsub.

    Progress channel key is `url_ingest:{task_id}`.
    """
    return _run_async(_async_ingest_url(self, request, user_id))


async def _async_ingest_url(task, request: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    task_id = current_task.request.id if current_task else None
    source_key = f"url_ingest:{task_id}" if task_id else "url_ingest:unknown"

    def publish(event_type: str, payload: dict) -> None:
        if event_type == "progress":
            _publish(source_key, "progress", "progress", payload)
        elif event_type == "status":
            _publish(source_key, "status", "status", payload)
        elif event_type == "complete":
            _publish(source_key, "complete", "result", payload)

        # Best-effort celery state updates for polling clients
        try:
            if isinstance(payload, dict) and "progress" in payload:
                task.update_state(state="PROGRESS", meta={"progress": payload.get("progress"), "stage": payload.get("stage"), "status": payload.get("status")})
        except Exception:
            pass

    def cancel_check() -> bool:
        try:
            client = _get_redis_client()
            if not client or not task_id:
                return False
            return bool(client.get(f"url_ingest:cancel:{task_id}"))
        except Exception:
            return False

    async with create_celery_session()() as db:
        try:
            # Validate user
            ures = await db.execute(select(User).where(User.id == UUID(user_id)))
            user = ures.scalar_one_or_none()
            if not user:
                err = "User not found"
                _publish(source_key, "error", "error", err)
                return {"error": err}

            service = UrlIngestionService()
            result = await service.ingest_url(
                db=db,
                user=user,
                url=str(request.get("url") or ""),
                title=request.get("title"),
                tags=request.get("tags"),
                follow_links=bool(request.get("follow_links", False)),
                max_pages=int(request.get("max_pages", 1)),
                max_depth=int(request.get("max_depth", 0)),
                same_domain_only=bool(request.get("same_domain_only", True)),
                one_document_per_page=bool(request.get("one_document_per_page", False)),
                allow_private_networks=bool(request.get("allow_private_networks", False)),
                max_content_chars=int(request.get("max_content_chars", 50_000)),
                publish=publish,
                cancel_check=cancel_check,
            )
            if result.get("error"):
                _publish(source_key, "error", "error", result["error"])
            return result
        except Exception as e:
            logger.error(f"URL ingestion task failed: {e}", exc_info=True)
            _publish(source_key, "error", "error", str(e))
            return {"error": str(e)}
