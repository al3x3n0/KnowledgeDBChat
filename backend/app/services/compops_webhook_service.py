"""Authentication and replay protection for CompOps refresh webhooks."""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import datetime, timezone
from typing import Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.compops_evidence_subscription import (
    CompOpsEvidenceSubscription,
    CompOpsWebhookEvent,
)
from app.models.secret import UserSecret
from app.services.secret_service import SecretService


class CompOpsWebhookAuthError(RuntimeError):
    """Raised for malformed, stale, or unauthenticated webhook signals."""


class CompOpsWebhookConflictError(RuntimeError):
    """Raised when an event ID is replayed with a different body."""


class CompOpsWebhookService:
    MAX_BODY_BYTES = 64 * 1024
    MAX_CLOCK_SKEW_SECONDS = 5 * 60
    SIGNATURE_PATTERN = re.compile(r"^v1=([0-9a-f]{64})$")
    EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")

    async def verify_and_record(
        self,
        *,
        subscription_id: UUID,
        raw_body: bytes,
        timestamp_value: str,
        event_id: str,
        signature: str,
        event_type: str,
        db: AsyncSession,
    ) -> Tuple[CompOpsWebhookEvent, bool]:
        if len(raw_body) > self.MAX_BODY_BYTES:
            raise CompOpsWebhookAuthError("Webhook body exceeded the size limit")
        normalized_event_id = str(event_id or "").strip()
        if not self.EVENT_ID_PATTERN.fullmatch(normalized_event_id):
            raise CompOpsWebhookAuthError("Webhook authentication failed")
        signature_match = self.SIGNATURE_PATTERN.fullmatch(
            str(signature or "").strip().lower()
        )
        try:
            timestamp = int(str(timestamp_value or "").strip())
        except (TypeError, ValueError) as exc:
            raise CompOpsWebhookAuthError("Webhook authentication failed") from exc
        now = int(datetime.now(timezone.utc).timestamp())
        if abs(now - timestamp) > self.MAX_CLOCK_SKEW_SECONDS:
            raise CompOpsWebhookAuthError("Webhook timestamp is stale")
        if signature_match is None:
            raise CompOpsWebhookAuthError("Webhook authentication failed")

        subscription = await db.get(
            CompOpsEvidenceSubscription,
            subscription_id,
        )
        if (
            subscription is None
            or not subscription.is_enabled
            or not subscription.webhook_enabled
            or subscription.webhook_secret_id is None
        ):
            raise CompOpsWebhookAuthError("Webhook authentication failed")
        secret = await db.get(UserSecret, subscription.webhook_secret_id)
        plaintext = (
            SecretService().decrypt(secret.encrypted_value)
            if secret is not None
            else None
        )
        if not plaintext:
            raise CompOpsWebhookAuthError("Webhook authentication failed")
        signed = (
            str(timestamp).encode("ascii")
            + b"."
            + normalized_event_id.encode("utf-8")
            + b"."
            + raw_body
        )
        expected = hmac.new(
            plaintext.encode("utf-8"),
            signed,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, signature_match.group(1)):
            raise CompOpsWebhookAuthError("Webhook authentication failed")

        payload_sha256 = hashlib.sha256(raw_body).hexdigest()
        existing = (
            await db.execute(
                select(CompOpsWebhookEvent).where(
                    CompOpsWebhookEvent.subscription_id == subscription.id,
                    CompOpsWebhookEvent.event_id == normalized_event_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.payload_sha256 != payload_sha256:
                raise CompOpsWebhookConflictError(
                    "Webhook event ID was reused with different content"
                )
            return existing, True

        event = CompOpsWebhookEvent(
            subscription_id=subscription.id,
            event_id=normalized_event_id,
            event_type=str(event_type or "").strip()[:120] or None,
            payload_sha256=payload_sha256,
            status="queued",
        )
        db.add(event)
        try:
            await db.flush()
        except IntegrityError:
            await db.rollback()
            existing = (
                await db.execute(
                    select(CompOpsWebhookEvent).where(
                        CompOpsWebhookEvent.subscription_id == subscription_id,
                        CompOpsWebhookEvent.event_id == normalized_event_id,
                    )
                )
            ).scalar_one()
            if existing.payload_sha256 != payload_sha256:
                raise CompOpsWebhookConflictError(
                    "Webhook event ID was reused with different content"
                )
            return existing, True
        now_value = datetime.now(timezone.utc)
        subscription.last_webhook_at = now_value
        subscription.last_webhook_event_id = normalized_event_id
        subscription.status = "webhook_queued"
        subscription.next_sync_at = now_value
        await db.commit()
        await db.refresh(event)
        return event, False


compops_webhook_service = CompOpsWebhookService()
