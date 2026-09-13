"""Celery delivery loop for transactional autonomous external-call outbox rows."""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from uuid import uuid4

from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.services.agent_external_call_outbox_service import (
    agent_external_call_outbox_service,
)
from app.services.agent_external_response_correlation_service import (
    agent_external_response_correlation_service,
)


@celery_app.task(
    bind=True,
    name="app.tasks.agent_external_call_outbox_tasks.deliver_external_call_outbox",
)
def deliver_external_call_outbox(
    self,
    batch_size: int = 50,
) -> Dict[str, Any]:
    owner_id = str(getattr(self.request, "id", None) or f"outbox:{uuid4()}")
    return asyncio.run(
        _async_deliver_external_call_outbox(
            owner_id=owner_id,
            batch_size=batch_size,
        )
    )


async def _async_deliver_external_call_outbox(
    *,
    owner_id: str,
    batch_size: int = 50,
) -> Dict[str, Any]:
    try:
        bounded_batch = max(1, min(int(batch_size or 50), 200))
    except (TypeError, ValueError):
        bounded_batch = 50
    summary: Dict[str, Any] = {
        "claimed": 0,
        "succeeded": 0,
        "retry": 0,
        "dead_letter": 0,
        "ack_conflict": 0,
        "resume_claimed": 0,
        "resume_enqueued": 0,
        "resume_retry": 0,
        "resume_skipped": 0,
    }
    session_factory = create_celery_session()
    async with session_factory() as db:
        for _ in range(bounded_batch):
            row = await agent_external_call_outbox_service.claim_next(
                db=db,
                owner_id=owner_id,
            )
            if row is None:
                break
            summary["claimed"] += 1
            result = await agent_external_call_outbox_service.deliver_claimed(
                db=db,
                row=row,
            )
            status = str(result.get("status") or "retry")
            if status in summary:
                summary[status] += 1
            else:
                summary["retry"] += 1
            if status in {"retry", "dead_letter", "ack_conflict"}:
                logger.warning(
                    "External-call outbox row {} finished with status {}: {}",
                    row.id,
                    status,
                    result.get("error"),
                )
        for _ in range(bounded_batch):
            row = await agent_external_response_correlation_service.claim_next(
                db=db,
                owner_id=f"{owner_id}:response",
            )
            if row is None:
                break
            summary["resume_claimed"] += 1
            result = await agent_external_response_correlation_service.correlate_and_dispatch(
                db=db,
                row=row,
            )
            status = str(result.get("status") or "")
            if status == "resume_enqueued":
                summary["resume_enqueued"] += 1
            elif status == "resume_retry":
                summary["resume_retry"] += 1
            else:
                summary["resume_skipped"] += 1
    return summary
