"""Atomic database leases and fencing tokens for autonomous job execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select, update

from app.models.agent_job import AgentJob, AgentJobStatus


class ExecutionLeaseLostError(RuntimeError):
    """Raised when a worker no longer owns the execution lease it was given."""

    fatal = True


@dataclass(frozen=True)
class ExecutionLease:
    job_id: UUID
    owner_id: str
    token: str
    fence: int
    expires_at: datetime


class AgentExecutionLeaseService:
    """Acquire, renew, validate, and release fenced job execution leases."""

    DEFAULT_TTL_SECONDS = 120
    MIN_TTL_SECONDS = 30
    MAX_TTL_SECONDS = 1800

    def normalize_ttl(self, ttl_seconds: Any = None) -> int:
        try:
            value = int(ttl_seconds or self.DEFAULT_TTL_SECONDS)
        except (TypeError, ValueError):
            value = self.DEFAULT_TTL_SECONDS
        return max(self.MIN_TTL_SECONDS, min(value, self.MAX_TTL_SECONDS))

    async def acquire(
        self,
        *,
        db: Any,
        job_id: UUID,
        owner_id: str,
        ttl_seconds: Any = None,
        now: Optional[datetime] = None,
    ) -> Optional[ExecutionLease]:
        """Atomically claim an unleased or expired runnable job."""
        acquired_at = now or datetime.utcnow()
        expires_at = acquired_at + timedelta(seconds=self.normalize_ttl(ttl_seconds))
        token = str(uuid4())
        statement = (
            update(AgentJob)
            .where(
                AgentJob.id == job_id,
                AgentJob.status.in_(
                    [
                        AgentJobStatus.PENDING.value,
                        AgentJobStatus.RUNNING.value,
                    ]
                ),
                or_(
                    AgentJob.execution_lease_expires_at.is_(None),
                    AgentJob.execution_lease_expires_at <= acquired_at,
                ),
            )
            .values(
                execution_lease_owner=str(owner_id),
                execution_lease_token=token,
                execution_lease_expires_at=expires_at,
                execution_lease_heartbeat_at=acquired_at,
                execution_fence=func.coalesce(AgentJob.execution_fence, 0) + 1,
            )
            .returning(AgentJob.execution_fence)
        )
        result = await db.execute(statement)
        fence = result.scalar_one_or_none()
        await db.commit()
        if fence is None:
            return None
        return ExecutionLease(
            job_id=job_id,
            owner_id=str(owner_id),
            token=token,
            fence=int(fence),
            expires_at=expires_at,
        )

    async def renew(
        self,
        *,
        db: Any,
        lease: ExecutionLease,
        ttl_seconds: Any = None,
        now: Optional[datetime] = None,
    ) -> Optional[ExecutionLease]:
        """Extend a live lease without allowing an expired owner to resurrect it."""
        renewed_at = now or datetime.utcnow()
        expires_at = renewed_at + timedelta(seconds=self.normalize_ttl(ttl_seconds))
        result = await db.execute(
            update(AgentJob)
            .where(
                AgentJob.id == lease.job_id,
                AgentJob.execution_lease_owner == lease.owner_id,
                AgentJob.execution_lease_token == lease.token,
                AgentJob.execution_fence == lease.fence,
                AgentJob.execution_lease_expires_at > renewed_at,
            )
            .values(
                execution_lease_expires_at=expires_at,
                execution_lease_heartbeat_at=renewed_at,
            )
        )
        await db.commit()
        if int(result.rowcount or 0) != 1:
            return None
        return ExecutionLease(
            job_id=lease.job_id,
            owner_id=lease.owner_id,
            token=lease.token,
            fence=lease.fence,
            expires_at=expires_at,
        )

    async def assert_owned(
        self,
        *,
        db: Any,
        lease: ExecutionLease,
        now: Optional[datetime] = None,
    ) -> None:
        """Fail if the token or fencing epoch is no longer current."""
        checked_at = now or datetime.utcnow()
        result = await db.execute(
            select(AgentJob.id).where(
                AgentJob.id == lease.job_id,
                AgentJob.execution_lease_owner == lease.owner_id,
                AgentJob.execution_lease_token == lease.token,
                AgentJob.execution_fence == lease.fence,
                AgentJob.execution_lease_expires_at > checked_at,
            )
        )
        if result.scalar_one_or_none() is None:
            raise ExecutionLeaseLostError(
                f"Execution lease lost for job {lease.job_id} at fence {lease.fence}"
            )

    async def release(self, *, db: Any, lease: ExecutionLease) -> bool:
        """Release only the exact lease token; stale workers cannot clear successors."""
        result = await db.execute(
            update(AgentJob)
            .where(
                AgentJob.id == lease.job_id,
                AgentJob.execution_lease_owner == lease.owner_id,
                AgentJob.execution_lease_token == lease.token,
                AgentJob.execution_fence == lease.fence,
            )
            .values(
                execution_lease_owner=None,
                execution_lease_token=None,
                execution_lease_expires_at=None,
                execution_lease_heartbeat_at=None,
            )
        )
        await db.commit()
        return int(result.rowcount or 0) == 1


agent_execution_lease_service = AgentExecutionLeaseService()
