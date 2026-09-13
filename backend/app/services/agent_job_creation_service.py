"""Shared construction and persistence for autonomous agent jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_definition import AgentDefinition
from app.models.agent_job import AgentJob, AgentJobStatus, AgentJobTemplate
from app.schemas.agent_job import AgentJobCreate, AgentJobFromTemplate
from app.services.agent_job_queue_helpers import extract_launch_mode
from app.services.agent_job_templates import get_builtin_agent_job_template
from app.services.agent_scope_service import (
    normalize_scope_config,
    normalize_scope_keys_deep,
)


class AgentJobCreationError(RuntimeError):
    """Domain error translated at the API boundary."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class AgentJobSpec:
    name: str
    job_type: str
    goal: str
    description: Optional[str] = None
    goal_criteria: Optional[dict] = None
    config: Optional[dict] = None
    agent_definition_id: Optional[UUID] = None
    enable_memory: Optional[bool] = None
    default_enable_memory: bool = True
    max_iterations: int = 100
    max_tool_calls: int = 500
    max_llm_calls: int = 200
    max_runtime_minutes: int = 60
    schedule_type: Optional[str] = None
    schedule_cron: Optional[str] = None
    chain_config: Optional[dict] = None
    parent_job_id: Optional[UUID] = None


class AgentJobCreationService:
    """Create jobs consistently while leaving task dispatch to callers."""

    async def create_from_request(
        self,
        *,
        request: AgentJobCreate,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJob:
        return await self.create(
            spec=AgentJobSpec(
                name=request.name,
                description=request.description,
                job_type=request.job_type,
                goal=request.goal,
                goal_criteria=request.goal_criteria,
                config=request.config,
                agent_definition_id=request.agent_definition_id,
                max_iterations=request.max_iterations or 100,
                max_tool_calls=request.max_tool_calls or 500,
                max_llm_calls=request.max_llm_calls or 200,
                max_runtime_minutes=request.max_runtime_minutes or 60,
                schedule_type=request.schedule_type,
                schedule_cron=request.schedule_cron,
                chain_config=request.chain_config,
                parent_job_id=request.parent_job_id,
            ),
            user_id=user_id,
            db=db,
            validate_agent_definition=True,
            validate_parent=True,
        )

    async def create_from_template(
        self,
        *,
        request: AgentJobFromTemplate,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJob:
        builtin = get_builtin_agent_job_template(request.template_id)
        template = None
        if builtin is None:
            template = (
                await db.execute(
                    select(AgentJobTemplate).where(
                        and_(
                            AgentJobTemplate.id == request.template_id,
                            AgentJobTemplate.is_active.is_(True),
                        )
                    )
                )
            ).scalar_one_or_none()
        if builtin is None and template is None:
            raise AgentJobCreationError(
                "Job template not found or not active",
                status_code=404,
            )

        source = builtin or template
        base_config = getattr(source, "default_config", None) or {}
        config = normalize_scope_config(dict(base_config)) or {}
        if request.config:
            config.update(normalize_scope_config(request.config) or {})
        config = normalize_scope_config(config)
        chain_config = (
            request.chain_config
            if request.chain_config
            else getattr(source, "default_chain_config", None)
        )

        return await self.create(
            spec=AgentJobSpec(
                name=request.name,
                description=getattr(source, "description", None),
                job_type=str(getattr(source, "job_type", None) or "custom"),
                goal=str(request.goal or getattr(source, "default_goal", None) or ""),
                config=config,
                agent_definition_id=getattr(source, "agent_definition_id", None),
                max_iterations=self._resource_value(
                    getattr(source, "default_max_iterations", None), 100
                ),
                max_tool_calls=self._resource_value(
                    getattr(source, "default_max_tool_calls", None), 500
                ),
                max_llm_calls=self._resource_value(
                    getattr(source, "default_max_llm_calls", None), 200
                ),
                max_runtime_minutes=self._resource_value(
                    getattr(source, "default_max_runtime_minutes", None), 60
                ),
                chain_config=chain_config,
            ),
            user_id=user_id,
            db=db,
        )

    async def create(
        self,
        *,
        spec: AgentJobSpec,
        user_id: UUID,
        db: AsyncSession,
        validate_agent_definition: bool = False,
        validate_parent: bool = False,
    ) -> AgentJob:
        if validate_agent_definition and spec.agent_definition_id:
            await self._ensure_agent_definition(
                agent_definition_id=spec.agent_definition_id,
                db=db,
            )

        parent_job = None
        if validate_parent and spec.parent_job_id:
            parent_job = await self._get_owned_parent(
                parent_job_id=spec.parent_job_id,
                user_id=user_id,
                db=db,
            )

        config = normalize_scope_config(spec.config)
        enable_memory = (
            spec.enable_memory
            if spec.enable_memory is not None
            else self.extract_enable_memory(
                config,
                default=spec.default_enable_memory,
            )
        )
        job = AgentJob(
            name=spec.name,
            description=spec.description,
            job_type=spec.job_type,
            goal=spec.goal,
            goal_criteria=spec.goal_criteria,
            config=config,
            agent_definition_id=spec.agent_definition_id,
            user_id=user_id,
            status=AgentJobStatus.PENDING.value,
            enable_memory=enable_memory,
            max_iterations=spec.max_iterations,
            max_tool_calls=spec.max_tool_calls,
            max_llm_calls=spec.max_llm_calls,
            max_runtime_minutes=spec.max_runtime_minutes,
            schedule_type=spec.schedule_type,
            schedule_cron=spec.schedule_cron,
            chain_config=normalize_scope_keys_deep(spec.chain_config),
            parent_job_id=spec.parent_job_id,
            chain_depth=(parent_job.chain_depth + 1) if parent_job else 0,
            root_job_id=(
                parent_job.root_job_id or parent_job.id if parent_job else None
            ),
        )
        self._set_initial_schedule(job)
        db.add(job)
        await db.flush()
        self.append_launch_log_if_present(job)
        await db.commit()
        await db.refresh(job)
        return job

    async def mark_immediately_dispatched(
        self,
        *,
        job: AgentJob,
        db: AsyncSession,
    ) -> None:
        if job.schedule_type == "continuous":
            interval = self._continuous_interval(job.config)
            job.next_run_at = datetime.utcnow() + timedelta(minutes=interval)
            await db.commit()
        elif job.schedule_type == "once":
            job.next_run_at = None
            await db.commit()

    @classmethod
    def append_launch_log_if_present(cls, job: object) -> bool:
        config = getattr(job, "config", None)
        config = config if isinstance(config, dict) else {}
        launch_mode = extract_launch_mode(config)
        if not launch_mode:
            return False
        quick_start = (
            config.get("quick_start")
            if isinstance(config.get("quick_start"), dict)
            else {}
        )
        coding_recovery = (
            config.get("coding_recovery")
            if isinstance(config.get("coding_recovery"), dict)
            else {}
        )
        commands = (
            config.get("commands") if isinstance(config.get("commands"), list) else []
        )
        file_paths = (
            config.get("file_paths")
            if isinstance(config.get("file_paths"), list)
            else []
        )
        normalized = normalize_scope_config(config) or {}
        getattr(job, "add_log_entry")(
            {
                "phase": "launch",
                "action": "job_launch",
                "result": {
                    "launch_mode": launch_mode,
                    "quick_start_profile": cls._clean(quick_start.get("profile")),
                    "quick_start_version": cls._clean(quick_start.get("version")),
                    "source_name": cls._clean(quick_start.get("source_name")),
                    "source_type": cls._clean(quick_start.get("source_type")),
                    "source_id": cls._clean(normalized.get("source_id")),
                    "search_query": cls._clean(config.get("search_query")),
                    "commands_count": len(commands),
                    "file_paths_count": len(file_paths),
                    "relaunch_from_job_id": cls._clean(
                        config.get("relaunch_from_job_id")
                    ),
                    "coding_recovery_strategy": cls._clean(
                        coding_recovery.get("strategy")
                    ),
                },
            }
        )
        return True

    @classmethod
    def extract_enable_memory(
        cls,
        config: Optional[dict],
        *,
        default: bool = True,
    ) -> bool:
        if not isinstance(config, dict):
            return bool(default)
        if "enable_memory" in config:
            return cls._coerce_bool(config.get("enable_memory"), default=default)
        memory = config.get("memory")
        if isinstance(memory, dict) and "enabled" in memory:
            return cls._coerce_bool(memory.get("enabled"), default=default)
        return bool(default)

    @staticmethod
    def find_unsafe_commands(commands: Optional[list[str]]) -> list[str]:
        if not isinstance(commands, list):
            return []
        blocked_patterns = [
            r"\brm\s+-rf\b",
            r"\bsudo\b",
            r"\bmkfs\b",
            r"\bdd\s+if=",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bhalt\b",
            r"\bpoweroff\b",
            r"\bchown\b",
            r"\bchmod\s+777\b",
        ]
        blocked: list[str] = []
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in blocked_patterns]
        for raw in commands:
            command = str(raw or "").strip()
            if command and any(regex.search(command) for regex in compiled):
                blocked.append(command)
        return blocked[:6]

    @staticmethod
    async def _ensure_agent_definition(
        *,
        agent_definition_id: UUID,
        db: AsyncSession,
    ) -> None:
        exists = (
            await db.execute(
                select(AgentDefinition.id).where(
                    AgentDefinition.id == agent_definition_id
                )
            )
        ).scalar_one_or_none()
        if exists is None:
            raise AgentJobCreationError(
                "Agent definition not found",
                status_code=404,
            )

    @staticmethod
    async def _get_owned_parent(
        *,
        parent_job_id: UUID,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJob:
        parent = (
            await db.execute(
                select(AgentJob).where(
                    and_(
                        AgentJob.id == parent_job_id,
                        AgentJob.user_id == user_id,
                    )
                )
            )
        ).scalar_one_or_none()
        if parent is None:
            raise AgentJobCreationError(
                "Parent job not found",
                status_code=404,
            )
        return parent

    @staticmethod
    def _set_initial_schedule(job: AgentJob) -> None:
        if job.schedule_type and job.schedule_cron:
            try:
                from croniter import croniter

                job.next_run_at = croniter(
                    job.schedule_cron, datetime.utcnow()
                ).get_next(datetime)
            except Exception as error:
                logger.warning(f"Invalid cron expression: {error}")
        elif job.schedule_type == "continuous" and not job.next_run_at:
            job.next_run_at = datetime.utcnow()

    @staticmethod
    def _continuous_interval(config: Optional[dict]) -> int:
        try:
            interval = int((config or {}).get("interval_minutes") or 30)
        except (TypeError, ValueError):
            interval = 30
        return max(1, min(interval, 24 * 60))

    @staticmethod
    def _resource_value(value: Any, default: int) -> int:
        return default if value is None else int(value)

    @staticmethod
    def _clean(value: Any) -> Optional[str]:
        return str(value or "").strip() or None

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return default


agent_job_creation_service = AgentJobCreationService()
