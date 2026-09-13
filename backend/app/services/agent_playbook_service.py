"""Reconstruct reusable chain definitions from autonomous job runs."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobChainDefinition, AgentJobStatus
from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobSaveAsChainRequest,
)
from app.services.agent_chain_definition_service import (
    AgentChainDefinitionService,
    agent_chain_definition_service,
)
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state,
    queue_reason_label,
)


class AgentPlaybookError(RuntimeError):
    """Domain error translated at the API boundary."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class AgentPlaybookService:
    """Build and persist a linear playbook from an owned job chain."""

    _MAX_STEPS = 25

    def __init__(
        self,
        definition_service: AgentChainDefinitionService,
    ) -> None:
        self.definition_service = definition_service

    async def save(
        self,
        *,
        job_id: UUID,
        request: AgentJobSaveAsChainRequest,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJobChainDefinition:
        job = await self._get_owned_job(job_id=job_id, user_id=user_id, db=db)
        root_job_id = job.root_job_id or job.id
        root_job = await db.get(AgentJob, root_job_id) or job
        is_recovery, recovery_reason, recovery_summary = self.build_recovery_metadata(
            root_job
        )

        steps = self._steps_from_nested_payload(root_job)
        if len(steps) <= 1:
            persisted_steps = await self._steps_from_persisted_chain(
                root_job_id=root_job_id,
                user_id=user_id,
                db=db,
            )
            if persisted_steps:
                steps = persisted_steps
        if steps:
            steps[-1]["trigger_condition"] = "on_complete"
            steps[-1]["trigger_thresholds"] = None

        now = datetime.utcnow()
        name = await self._resolve_name(
            requested_name=request.name,
            root_job=root_job,
            is_recovery=is_recovery,
            now=now,
            db=db,
        )
        display_name = self._display_name(
            request=request,
            root_job=root_job,
            is_recovery=is_recovery,
        )
        description = self._description(
            request=request,
            root_job_id=root_job_id,
            is_recovery=is_recovery,
            recovery_reason=recovery_reason,
            recovery_summary=recovery_summary,
            now=now,
        )

        return await self.definition_service.create(
            request=AgentJobChainDefinitionCreate(
                name=name,
                display_name=display_name,
                description=description,
                chain_steps=steps,
                default_settings={
                    "inherit_results": True,
                    "inherit_config": True,
                    "max_iterations": int(root_job.max_iterations or 100),
                    "max_tool_calls": int(root_job.max_tool_calls or 500),
                    "max_llm_calls": int(root_job.max_llm_calls or 200),
                    "max_runtime_minutes": int(root_job.max_runtime_minutes or 60),
                },
            ),
            user_id=user_id,
            db=db,
        )

    @staticmethod
    def is_recovery_candidate(job: AgentJob) -> bool:
        scheduler_state = extract_scheduler_state(job) or {}
        queue_reason = str(scheduler_state.get("queue_reason") or "").strip().lower()
        if queue_reason in {
            "execution_failure",
            "stalled_run",
            "scheduled_recovery",
            "scheduler_backoff",
        }:
            return True
        status = str(job.status or "").strip().lower()
        if status in {
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }:
            return bool(
                str(job.error or "").strip()
                or str(job.phase_details or "").strip()
                or queue_reason
            )
        if status == AgentJobStatus.PAUSED.value:
            return bool(queue_reason or str(job.phase_details or "").strip())
        return False

    @classmethod
    def build_recovery_metadata(
        cls,
        job: AgentJob,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        scheduler_state = extract_scheduler_state(job) or {}
        queue_reason = str(scheduler_state.get("queue_reason") or "").strip().lower()
        if not cls.is_recovery_candidate(job):
            return False, None, None

        reason_label = queue_reason_label(queue_reason) if queue_reason else None
        fragments: list[str] = []
        status = str(job.status or "").strip().lower()
        if status:
            fragments.append(f"Current status: {status}.")
        error = str(job.error or "").strip()
        if error:
            fragments.append(f"Error: {error[:240]}.")
        phase_details = str(job.phase_details or "").strip()
        if phase_details and phase_details != error:
            fragments.append(f"Details: {phase_details[:240]}.")
        return True, reason_label, " ".join(fragments).strip() or None

    async def _get_owned_job(
        self,
        *,
        job_id: UUID,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJob:
        job = (
            await db.execute(
                select(AgentJob).where(
                    and_(AgentJob.id == job_id, AgentJob.user_id == user_id)
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise AgentPlaybookError("Agent job not found", status_code=404)
        return job

    def _steps_from_nested_payload(self, root_job: AgentJob) -> list[dict]:
        steps: list[dict] = []
        payload = {
            "name": root_job.name,
            "job_type": root_job.job_type,
            "goal": root_job.goal,
            "config": root_job.config,
        }
        chain_config = (
            root_job.chain_config if isinstance(root_job.chain_config, dict) else None
        )
        seen: set[str] = set()

        while len(steps) < self._MAX_STEPS:
            steps.append(self._step_from_payload(payload, chain_config))
            child_jobs = (
                chain_config.get("child_jobs")
                if isinstance(chain_config, dict)
                else None
            )
            if not isinstance(child_jobs, list) or not child_jobs:
                break
            child = child_jobs[0]
            if not isinstance(child, dict):
                break
            fingerprint = json.dumps(child, sort_keys=True, default=str)[:2000]
            if fingerprint in seen:
                break
            seen.add(fingerprint)
            payload = child
            chain_config = (
                child.get("chain_config")
                if isinstance(child.get("chain_config"), dict)
                else None
            )
        return steps

    async def _steps_from_persisted_chain(
        self,
        *,
        root_job_id: UUID,
        user_id: UUID,
        db: AsyncSession,
    ) -> list[dict]:
        jobs = list(
            (
                await db.execute(
                    select(AgentJob)
                    .where(
                        and_(
                            AgentJob.user_id == user_id,
                            or_(
                                AgentJob.id == root_job_id,
                                AgentJob.root_job_id == root_job_id,
                            ),
                        )
                    )
                    .order_by(AgentJob.chain_depth, AgentJob.created_at)
                )
            )
            .scalars()
            .all()
        )
        if not jobs:
            return []

        by_parent: dict[UUID, list[AgentJob]] = {}
        for job in jobs:
            if job.parent_job_id:
                by_parent.setdefault(job.parent_job_id, []).append(job)
        for children in by_parent.values():
            children.sort(
                key=lambda job: (
                    job.created_at.isoformat() if job.created_at else "",
                    str(job.id),
                )
            )

        linear = [jobs[0]]
        current = jobs[0]
        while len(linear) < self._MAX_STEPS:
            children = by_parent.get(current.id) or []
            if not children:
                break
            current = children[0]
            linear.append(current)

        return [
            self._step_from_payload(
                {
                    "name": job.name,
                    "job_type": job.job_type,
                    "goal": job.goal,
                    "config": job.config,
                },
                job.chain_config if isinstance(job.chain_config, dict) else None,
            )
            for job in linear
        ]

    @staticmethod
    def _step_from_payload(
        payload: dict,
        chain_config: Optional[dict],
    ) -> dict:
        thresholds = {}
        for field in ("progress_threshold", "findings_threshold"):
            value = chain_config.get(field) if isinstance(chain_config, dict) else None
            if isinstance(value, int):
                thresholds[field] = value
        name = str(payload.get("name") or "").strip()
        return {
            "step_name": name[:200] if name else "Step",
            "template_id": None,
            "job_type": str(payload.get("job_type") or "custom"),
            "goal_template": str(payload.get("goal") or ""),
            "config": (
                payload.get("config")
                if isinstance(payload.get("config"), dict)
                else None
            ),
            "trigger_condition": str(
                (chain_config or {}).get("trigger_condition") or "on_complete"
            ),
            "trigger_thresholds": thresholds or None,
        }

    async def _resolve_name(
        self,
        *,
        requested_name: Optional[str],
        root_job: AgentJob,
        is_recovery: bool,
        now: datetime,
        db: AsyncSession,
    ) -> str:
        requested = str(requested_name or "").strip()
        if requested:
            if await self._name_exists(requested, db=db):
                raise AgentPlaybookError(
                    "Chain definition name already exists",
                    status_code=400,
                )
            return requested

        base = self._slugify(root_job.name)
        prefix = "playbook_recovery" if is_recovery else "playbook"
        name = f"{prefix}_{base}_{now.strftime('%Y%m%d_%H%M%S')}"[:100]
        for _ in range(5):
            if not await self._name_exists(name, db=db):
                return name
            name = (name[:90] + "_" + uuid.uuid4().hex[:8])[:100]
        return name

    @staticmethod
    async def _name_exists(name: str, *, db: AsyncSession) -> bool:
        return (
            await db.execute(
                select(AgentJobChainDefinition.id).where(
                    AgentJobChainDefinition.name == name
                )
            )
        ).scalar_one_or_none() is not None

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(
            r"[^a-z0-9_]+",
            "_",
            str(value or "").strip().lower(),
        )
        return re.sub(r"_+", "_", slug).strip("_")[:40] or "job"

    @staticmethod
    def _display_name(
        *,
        request: AgentJobSaveAsChainRequest,
        root_job: AgentJob,
        is_recovery: bool,
    ) -> str:
        requested = str(request.display_name or "").strip()
        if requested:
            return requested
        suffix = "Recovery Playbook" if is_recovery else "Playbook"
        return f"{root_job.name} ({suffix})"[:200]

    @staticmethod
    def _description(
        *,
        request: AgentJobSaveAsChainRequest,
        root_job_id: UUID,
        is_recovery: bool,
        recovery_reason: Optional[str],
        recovery_summary: Optional[str],
        now: datetime,
    ) -> str:
        requested = str(request.description or "").strip()
        if requested:
            return requested
        description = f"Saved from job {root_job_id} on {now.isoformat()}."
        if is_recovery:
            description += " Saved as a recovery playbook."
            if recovery_reason:
                description += f" Recovery reason: {recovery_reason}."
            if recovery_summary:
                description += f" {recovery_summary}"
        return description


agent_playbook_service = AgentPlaybookService(agent_chain_definition_service)
