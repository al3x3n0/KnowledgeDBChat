"""Periodic synchronization for bounded CompOps evidence subscriptions."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.compops_evidence_subscription import (
    CompOpsEvidenceSubscription,
    CompOpsWebhookEvent,
)
from app.services.compops_evidence_sync_service import (
    CompOpsEvidenceSyncError,
    compops_evidence_sync_service,
)


@celery_app.task(name="app.tasks.compops_sync_tasks.sync_due_compops_evidence")
def sync_due_compops_evidence() -> Dict[str, Any]:
    return asyncio.run(_async_sync_due_compops_evidence())


@celery_app.task(name="app.tasks.compops_sync_tasks.sync_compops_webhook_event")
def sync_compops_webhook_event(
    subscription_id: str,
    webhook_event_id: str,
) -> Dict[str, Any]:
    return asyncio.run(
        _async_sync_compops_webhook_event(
            subscription_id=subscription_id,
            webhook_event_id=webhook_event_id,
        )
    )


async def _async_sync_compops_webhook_event(
    *,
    subscription_id: str,
    webhook_event_id: str,
) -> Dict[str, Any]:
    try:
        subscription_uuid = UUID(subscription_id)
        event_uuid = UUID(webhook_event_id)
    except (TypeError, ValueError):
        return {"processed": False, "reason": "invalid_identifier"}
    async with create_celery_session()() as db:
        subscription = await db.get(
            CompOpsEvidenceSubscription,
            subscription_uuid,
        )
        event = await db.get(CompOpsWebhookEvent, event_uuid)
        if (
            subscription is None
            or event is None
            or str(event.subscription_id) != str(subscription.id)
        ):
            return {"processed": False, "reason": "subscription_or_event_missing"}
        if event.status == "processed":
            return {
                "processed": True,
                "duplicate": True,
                "evidence_changed": bool(event.evidence_changed),
            }
        try:
            changed = await compops_evidence_sync_service.sync(
                subscription=subscription,
                db=db,
                trigger="webhook",
                trigger_event_id=event.event_id,
            )
            await db.refresh(event)
            event.status = "processed" if subscription.status == "active" else "failed"
            event.evidence_changed = changed
            event.error = (
                None if subscription.status == "active" else subscription.last_error
            )
            event.processed_at = datetime.now(timezone.utc)
            await db.commit()
            return {
                "processed": event.status == "processed",
                "duplicate": False,
                "evidence_changed": changed,
            }
        except CompOpsEvidenceSyncError as exc:
            event.status = "failed"
            event.error = str(exc)[:4000]
            event.processed_at = datetime.now(timezone.utc)
            await db.commit()
            return {
                "processed": False,
                "duplicate": False,
                "error": str(exc),
            }


async def _async_sync_due_compops_evidence() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    summary: Dict[str, Any] = {
        "checked": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
        "timestamp": now.isoformat(),
    }
    async with create_celery_session()() as db:
        for _ in range(100):
            row = (
                await db.execute(
                    select(CompOpsEvidenceSubscription)
                    .where(
                        CompOpsEvidenceSubscription.is_enabled.is_(True),
                        CompOpsEvidenceSubscription.next_sync_at.is_not(None),
                        CompOpsEvidenceSubscription.next_sync_at <= now,
                    )
                    .order_by(CompOpsEvidenceSubscription.next_sync_at.asc())
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is None:
                break
            # Commit a short lease before making the network call. Another Beat
            # worker will not select this row if it starts while the call is active.
            row.status = "syncing"
            row.next_sync_at = now + timedelta(minutes=5)
            await db.commit()
            summary["checked"] += 1
            try:
                changed = await compops_evidence_sync_service.sync(
                    subscription=row,
                    db=db,
                )
                if row.status != "active":
                    summary["failed"] += 1
                elif changed:
                    summary["updated"] += 1
                else:
                    summary["unchanged"] += 1
            except CompOpsEvidenceSyncError as exc:
                summary["failed"] += 1
                logger.warning(
                    "CompOps evidence subscription {} was skipped: {}",
                    row.id,
                    exc,
                )
    return summary
