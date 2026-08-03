"""Transactional enqueue and claimed delivery for autonomous external calls."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.models.agent_external_call_outbox import AgentExternalCallOutbox
from app.models.user import User
from app.models.workflow import UserTool
from app.services.external_agent_gateway_service import (
    ExternalAgentGatewayError,
    external_agent_gateway_service,
)


class AgentExternalCallOutboxError(RuntimeError):
    """Raised when an outbox request is invalid or cannot be persisted safely."""


class AgentExternalCallOutboxService:
    """Create idempotent outbox rows and deliver them under short claims."""

    CLAIM_TTL_SECONDS = 120
    MAX_PAYLOAD_BYTES = 512 * 1024
    MAX_ATTEMPTS = 8

    async def enqueue(
        self,
        *,
        db: Any,
        user_id: UUID,
        tool_id: UUID,
        capability: str,
        payload: Dict[str, Any],
        idempotency_key: str,
        job_id: Optional[UUID] = None,
        request_id: Optional[str] = None,
        max_attempts: int = 5,
        correlation: Optional[Dict[str, Any]] = None,
    ) -> tuple[AgentExternalCallOutbox, bool]:
        """Add a row without committing so the caller controls the transaction."""
        normalized_key = str(idempotency_key or "").strip()
        if not normalized_key or len(normalized_key) > 128:
            raise AgentExternalCallOutboxError(
                "idempotency_key must contain between 1 and 128 characters"
            )
        normalized_capability = str(capability or "").strip().lower()
        if not normalized_capability or len(normalized_capability) > 160:
            raise AgentExternalCallOutboxError("capability is required")
        if not isinstance(payload, dict):
            raise AgentExternalCallOutboxError("payload must be an object")
        try:
            encoded = json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise AgentExternalCallOutboxError(
                "payload must be JSON serializable"
            ) from exc
        if len(encoded) > self.MAX_PAYLOAD_BYTES:
            raise AgentExternalCallOutboxError("payload exceeded outbox size limit")
        try:
            bounded_attempts = max(1, min(int(max_attempts or 5), self.MAX_ATTEMPTS))
        except (TypeError, ValueError):
            bounded_attempts = 5

        existing = (
            await db.execute(
                select(AgentExternalCallOutbox).where(
                    AgentExternalCallOutbox.idempotency_key == normalized_key
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing, False

        normalized_request_id = (
            str(request_id or "").strip()
            or f"agent-outbox-{str(job_id or user_id)}-{normalized_key}"
        )
        row_id = uuid4()
        values = {
            "id": row_id,
            "job_id": job_id,
            "user_id": user_id,
            "tool_id": tool_id,
            "capability": normalized_capability,
            "payload": dict(payload),
            "request_id": normalized_request_id[:200],
            "idempotency_key": normalized_key,
            "correlation": dict(correlation or {}) or None,
            "status": "pending",
            "attempts": 0,
            "max_attempts": bounded_attempts,
            "next_attempt_at": datetime.now(timezone.utc),
        }
        bind = db.get_bind()
        dialect_name = str(getattr(getattr(bind, "dialect", None), "name", ""))
        if dialect_name == "postgresql":
            statement = (
                postgresql_insert(AgentExternalCallOutbox)
                .values(**values)
                .on_conflict_do_nothing()
            )
            insert_result = await db.execute(statement)
            created = int(insert_result.rowcount or 0) == 1
        elif dialect_name == "sqlite":
            statement = (
                sqlite_insert(AgentExternalCallOutbox)
                .values(**values)
                .on_conflict_do_nothing()
            )
            insert_result = await db.execute(statement)
            created = int(insert_result.rowcount or 0) == 1
        else:
            row = AgentExternalCallOutbox(**values)
            db.add(row)
            await db.flush()
            return row, True
        row = (
            await db.execute(
                select(AgentExternalCallOutbox).where(
                    or_(
                        AgentExternalCallOutbox.id == row_id,
                        AgentExternalCallOutbox.idempotency_key == normalized_key,
                        AgentExternalCallOutbox.request_id
                        == normalized_request_id[:200],
                    )
                )
            )
        ).scalar_one()
        return row, created

    async def claim_next(
        self,
        *,
        db: Any,
        owner_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[AgentExternalCallOutbox]:
        """Claim one due row, including abandoned expired claims."""
        claimed_at = now or datetime.now(timezone.utc)
        row = (
            await db.execute(
                select(AgentExternalCallOutbox)
                .where(
                    or_(
                        and_(
                            AgentExternalCallOutbox.status.in_(["pending", "retry"]),
                            AgentExternalCallOutbox.next_attempt_at <= claimed_at,
                        ),
                        and_(
                            AgentExternalCallOutbox.status == "in_flight",
                            AgentExternalCallOutbox.claim_expires_at <= claimed_at,
                        ),
                    )
                )
                .order_by(
                    AgentExternalCallOutbox.next_attempt_at.asc(),
                    AgentExternalCallOutbox.created_at.asc(),
                )
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = "in_flight"
        row.claim_owner = str(owner_id)[:200]
        row.claim_token = str(uuid4())
        row.claim_expires_at = claimed_at + timedelta(seconds=self.CLAIM_TTL_SECONDS)
        row.attempts = int(row.attempts or 0) + 1
        await db.commit()
        await db.refresh(row)
        return row

    async def deliver_claimed(
        self,
        *,
        db: Any,
        row: AgentExternalCallOutbox,
    ) -> Dict[str, Any]:
        """Deliver a claimed request and fence its acknowledgement by claim token."""
        claim_token = str(row.claim_token or "")
        if row.status != "in_flight" or not claim_token:
            raise AgentExternalCallOutboxError("outbox row is not actively claimed")
        user = await db.get(User, row.user_id)
        tool = await db.get(UserTool, row.tool_id) if row.tool_id else None
        if user is None or tool is None or not bool(tool.is_enabled):
            return await self._record_failure(
                db=db,
                row=row,
                claim_token=claim_token,
                error="Outbox owner or enabled external-agent connection is missing",
                permanent=True,
            )
        if tool.tool_type != "external_agent":
            return await self._record_failure(
                db=db,
                row=row,
                claim_token=claim_token,
                error="Outbox tool is not an external-agent connection",
                permanent=True,
            )
        try:
            result = await external_agent_gateway_service.invoke(
                tool=tool,
                user=user,
                db=db,
                capability=row.capability,
                payload=dict(row.payload or {}),
                request_id=row.request_id,
            )
        except ExternalAgentGatewayError as exc:
            return await self._record_failure(
                db=db,
                row=row,
                claim_token=claim_token,
                error=str(exc),
                permanent=False,
            )
        except Exception as exc:  # noqa: BLE001 - retained for durable retry
            logger.exception("Unexpected external-call outbox delivery failure")
            return await self._record_failure(
                db=db,
                row=row,
                claim_token=claim_token,
                error=f"{type(exc).__name__}: {exc}",
                permanent=False,
            )

        delivered_at = datetime.now(timezone.utc)
        update_result = await db.execute(
            update(AgentExternalCallOutbox)
            .where(
                AgentExternalCallOutbox.id == row.id,
                AgentExternalCallOutbox.status == "in_flight",
                AgentExternalCallOutbox.claim_token == claim_token,
            )
            .values(
                status="succeeded",
                response=result,
                error=None,
                delivered_at=delivered_at,
                claim_owner=None,
                claim_token=None,
                claim_expires_at=None,
            )
        )
        await db.commit()
        acknowledged = int(update_result.rowcount or 0) == 1
        return {
            "status": "succeeded" if acknowledged else "ack_conflict",
            "outbox_id": str(row.id),
            "acknowledged": acknowledged,
            "response": result if acknowledged else None,
        }

    async def _record_failure(
        self,
        *,
        db: Any,
        row: AgentExternalCallOutbox,
        claim_token: str,
        error: str,
        permanent: bool,
    ) -> Dict[str, Any]:
        attempts = int(row.attempts or 0)
        dead_letter = permanent or attempts >= int(row.max_attempts or 5)
        delay_seconds = min(30 * (2 ** max(0, attempts - 1)), 60 * 60)
        next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
        update_result = await db.execute(
            update(AgentExternalCallOutbox)
            .where(
                AgentExternalCallOutbox.id == row.id,
                AgentExternalCallOutbox.status == "in_flight",
                AgentExternalCallOutbox.claim_token == claim_token,
            )
            .values(
                status="dead_letter" if dead_letter else "retry",
                error=str(error)[:4000],
                next_attempt_at=next_attempt_at,
                claim_owner=None,
                claim_token=None,
                claim_expires_at=None,
            )
        )
        await db.commit()
        acknowledged = int(update_result.rowcount or 0) == 1
        return {
            "status": (
                ("dead_letter" if dead_letter else "retry")
                if acknowledged
                else "ack_conflict"
            ),
            "outbox_id": str(row.id),
            "acknowledged": acknowledged,
            "attempts": attempts,
            "error": str(error)[:4000],
        }


agent_external_call_outbox_service = AgentExternalCallOutboxService()
