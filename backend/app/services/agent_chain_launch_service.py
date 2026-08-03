"""Create the root job and continuation payload for a reusable job chain."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.schemas.agent_job import AgentJobFromChainCreate
from app.services.agent_chain_definition_service import (
    AgentChainDefinitionService,
    agent_chain_definition_service,
)
from app.services.agent_scope_service import (
    merge_chain_step_config,
    normalize_scope_keys_deep,
)


class AgentChainLaunchError(RuntimeError):
    """Invalid chain content that prevents creation of its root job."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class AgentChainLaunchService:
    """Build and persist executable jobs from visible chain definitions."""

    _RESOURCE_DEFAULTS = {
        "max_iterations": 100,
        "max_tool_calls": 500,
        "max_llm_calls": 200,
        "max_runtime_minutes": 60,
    }

    def __init__(
        self,
        definition_service: AgentChainDefinitionService,
    ) -> None:
        self.definition_service = definition_service

    async def launch(
        self,
        *,
        request: AgentJobFromChainCreate,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentJob:
        chain = await self.definition_service.get_visible(
            chain_id=request.chain_definition_id,
            user_id=user_id,
            db=db,
            require_active=True,
        )
        steps = getattr(chain, "chain_steps", None)
        if not isinstance(steps, list) or not steps:
            raise AgentChainLaunchError(
                "Chain definition has no steps",
                status_code=400,
            )

        first_step = steps[0]
        if not isinstance(first_step, dict):
            raise AgentChainLaunchError(
                "Chain definition contains an invalid first step",
                status_code=400,
            )

        defaults = normalize_scope_keys_deep(getattr(chain, "default_settings", None))
        defaults = defaults if isinstance(defaults, dict) else {}
        overrides = normalize_scope_keys_deep(request.config_overrides)
        overrides = overrides if isinstance(overrides, dict) else {}
        job_config = self.build_step_config(
            defaults=defaults,
            step=first_step,
            overrides=overrides,
        )
        step_name = str(first_step.get("step_name") or "Step 1").strip()
        goal = self.substitute_variables(
            str(first_step.get("goal_template") or ""),
            request.variables,
        )

        job = AgentJob(
            name=self.build_job_name(request.name_prefix, step_name),
            description=getattr(chain, "description", None),
            job_type=str(first_step.get("job_type") or "custom"),
            goal=goal,
            config=job_config,
            user_id=user_id,
            status=AgentJobStatus.PENDING.value,
            enable_memory=self._enable_memory(job_config),
            max_iterations=self._resource_limit(job_config, "max_iterations"),
            max_tool_calls=self._resource_limit(job_config, "max_tool_calls"),
            max_llm_calls=self._resource_limit(job_config, "max_llm_calls"),
            max_runtime_minutes=self._resource_limit(job_config, "max_runtime_minutes"),
            chain_config=self.build_chain_config_for_step(
                chain,
                0,
                request.variables,
                defaults,
                overrides=overrides,
                name_prefix=request.name_prefix,
            ),
            chain_depth=0,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job

    def build_chain_config_for_step(
        self,
        chain: object,
        step_index: int,
        variables: dict,
        default_settings: dict,
        *,
        overrides: Optional[dict] = None,
        name_prefix: Optional[str] = None,
    ) -> Optional[dict]:
        """Build the recursive child-job payload consumed by orchestration."""
        steps = getattr(chain, "chain_steps", None)
        if not isinstance(steps, list) or step_index >= len(steps):
            return None
        step = steps[step_index]
        if not isinstance(step, dict):
            return None

        step_config = self.build_step_config(
            defaults=default_settings,
            step=step,
            overrides=overrides or {},
        )
        config: dict[str, Any] = {
            "trigger_condition": step.get("trigger_condition", "on_complete"),
            "inherit_results": self._coerce_bool(
                step_config.get("inherit_results"), default=True
            ),
            "inherit_config": self._coerce_bool(
                step_config.get("inherit_config"), default=False
            ),
            "chain_definition_id": str(getattr(chain, "id")),
            "current_step_index": step_index,
            "total_steps": len(steps),
            "variables": dict(variables or {}),
        }
        if name_prefix:
            config["name_prefix"] = str(name_prefix)
        thresholds = step.get("trigger_thresholds")
        if isinstance(thresholds, dict):
            config.update(thresholds)

        if step_index + 1 < len(steps):
            next_step = steps[step_index + 1]
            if isinstance(next_step, dict):
                next_name = str(next_step.get("step_name") or f"Step {step_index + 2}")
                next_config = self.build_step_config(
                    defaults=default_settings,
                    step=next_step,
                    overrides=overrides or {},
                )
                child: dict[str, Any] = {
                    "name": self.build_job_name(name_prefix, next_name),
                    "job_type": str(next_step.get("job_type") or "custom"),
                    "goal": self.substitute_variables(
                        str(next_step.get("goal_template") or ""),
                        variables,
                    ),
                    "config": next_config,
                    "chain_config": self.build_chain_config_for_step(
                        chain,
                        step_index + 1,
                        variables,
                        default_settings,
                        overrides=overrides,
                        name_prefix=name_prefix,
                    ),
                }
                for field in self._RESOURCE_DEFAULTS:
                    child[field] = self._resource_limit(next_config, field)
                config["child_jobs"] = [child]

        return normalize_scope_keys_deep(config)

    @classmethod
    def build_step_config(
        cls,
        *,
        defaults: dict,
        step: dict,
        overrides: dict,
    ) -> dict:
        """Apply chain defaults, step settings, then launch overrides."""
        merged = merge_chain_step_config(
            defaults,
            step.get("config") if isinstance(step.get("config"), dict) else {},
        )
        return normalize_scope_keys_deep(cls._merge_overrides(merged, overrides))

    @staticmethod
    def substitute_variables(template: str, variables: dict) -> str:
        result = template
        for key, value in (variables or {}).items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    @staticmethod
    def build_job_name(
        name_prefix: Optional[str],
        step_name: str,
    ) -> str:
        prefix = str(name_prefix or "").strip()
        step = str(step_name or "").strip()
        if prefix and step:
            return f"{prefix}: {step}"[:200]
        return (prefix or step or "Chain Job")[:200]

    @classmethod
    def _resource_limit(cls, config: dict, field: str) -> int:
        default = cls._RESOURCE_DEFAULTS[field]
        try:
            return int(config.get(field, default))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _merge_overrides(cls, base: Any, overrides: Any) -> Any:
        if not isinstance(base, dict) or not isinstance(overrides, dict):
            return deepcopy(overrides)
        merged = deepcopy(base)
        for key, value in overrides.items():
            if isinstance(merged.get(key), dict) and isinstance(value, dict):
                merged[key] = cls._merge_overrides(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    @classmethod
    def _enable_memory(cls, config: dict) -> bool:
        if "enable_memory" in config:
            return cls._coerce_bool(config.get("enable_memory"), default=True)
        memory = config.get("memory")
        if isinstance(memory, dict) and "enabled" in memory:
            return cls._coerce_bool(memory.get("enabled"), default=True)
        return True

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


agent_chain_launch_service = AgentChainLaunchService(agent_chain_definition_service)
