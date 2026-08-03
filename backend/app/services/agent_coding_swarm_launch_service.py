"""Launch orchestration for coding-swarm quick-start presets."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, TypeAlias
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.document import Document, DocumentSource
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobQuickStartBugTriageSwarmRequest,
    AgentJobQuickStartBuildBreakSwarmRequest,
    AgentJobQuickStartFrontendRegressionSwarmRequest,
)
from app.services.agent_coding_harness_service import (
    agent_coding_harness_service,
)
from app.services.agent_coding_workspace_session_service import (
    agent_coding_workspace_session_service,
)
from app.services.agent_job_creation_service import agent_job_creation_service
from app.services.agent_scope_service import normalize_scope_config
from app.services.agent_swarm_collaboration_service import (
    build_collaboration_payload,
    is_profile_visible_to_user,
    normalize_profile_visibility,
    normalize_uuid_str_list,
    store_swarm_collaboration,
)

CodingSwarmLaunchRequest: TypeAlias = (
    AgentJobQuickStartBugTriageSwarmRequest
    | AgentJobQuickStartBuildBreakSwarmRequest
    | AgentJobQuickStartFrontendRegressionSwarmRequest
)


CODING_SWARM_PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "bug_triage_swarm": {
        "launch_mode": "quick_start_bug_triage_swarm",
        "coding_profile": "bug_triage",
        "quick_start_profile": "bug_triage_swarm",
        "display_name": "Bug Triage Swarm",
        "goal_prefix": "Run a coding bug triage swarm",
        "default_scope": "auto",
        "default_search_suffix": "bug symptom",
        "roles": ["reproducer", "root_cause", "patcher", "verifier"],
        "fan_in_name": "Bug Triage Swarm Fan-In",
        "confidence_threshold": 0.70,
        "tiebreaker_threshold": 0.50,
    },
    "build_break_swarm": {
        "launch_mode": "quick_start_build_break_swarm",
        "coding_profile": "build_break",
        "quick_start_profile": "build_break_swarm",
        "display_name": "Build Break Swarm",
        "goal_prefix": "Run a coding swarm for the reported build break",
        "default_scope": "backend",
        "default_search_suffix": "build break compile failure",
        "roles": ["reproducer", "root_cause", "patcher", "verifier"],
        "fan_in_name": "Build Break Swarm Fan-In",
        "confidence_threshold": 0.72,
        "tiebreaker_threshold": 0.52,
    },
    "frontend_regression_swarm": {
        "launch_mode": "quick_start_frontend_regression_swarm",
        "coding_profile": "frontend_regression",
        "quick_start_profile": "frontend_regression_swarm",
        "display_name": "Frontend Regression Swarm",
        "goal_prefix": "Run a coding swarm for the reported frontend regression",
        "default_scope": "frontend",
        "default_search_suffix": "frontend regression ui failure",
        "roles": ["reproducer", "root_cause", "patcher", "verifier"],
        "fan_in_name": "Frontend Regression Swarm Fan-In",
        "confidence_threshold": 0.70,
        "tiebreaker_threshold": 0.50,
    },
}


class AgentCodingSwarmLaunchError(RuntimeError):
    """Domain error translated at the API boundary."""

    def __init__(self, detail: str | dict[str, Any], *, status_code: int) -> None:
        super().__init__(str(detail))
        self.detail = detail
        self.status_code = status_code


class AgentCodingSwarmLaunchService:
    """Validate and persist coding-swarm quick-start jobs."""

    @staticmethod
    def get_preset(preset_key: str) -> dict[str, Any]:
        preset = CODING_SWARM_PRESET_DEFINITIONS.get(
            str(preset_key or "").strip().lower()
        )
        if not isinstance(preset, dict):
            raise AgentCodingSwarmLaunchError(
                "Unknown coding swarm preset",
                status_code=400,
            )
        return preset

    @staticmethod
    def is_source_owned_by_user(source: DocumentSource, user: User) -> bool:
        config = source.config or {}
        if not isinstance(config, dict):
            return False
        requested_by = config.get("requested_by") or config.get("requestedBy")
        requested_by_user_id = config.get("requested_by_user_id") or config.get(
            "requestedByUserId"
        )
        return requested_by in {
            user.username,
            str(user.id),
        } or requested_by_user_id == str(user.id)

    def build_goal(
        self,
        request: CodingSwarmLaunchRequest,
        *,
        preset_key: str,
    ) -> str:
        preset = self.get_preset(preset_key)
        symptom = str(request.failure_symptom or "").strip()
        goal = str(request.goal or "").strip()
        scope = (
            str(request.scope or preset.get("default_scope") or "auto").strip().lower()
        )
        goal_prefix = str(preset.get("goal_prefix") or "Run a coding swarm").strip()

        if symptom and goal:
            return (
                f"{goal_prefix} for the reported {scope} issue.\n"
                f"Symptom: {symptom}\n"
                f"Desired outcome: {goal}"
            )
        if symptom:
            return f"{goal_prefix} for the reported {scope} issue. Symptom: {symptom}"
        return goal

    def merge_request_with_profile(
        self,
        request: CodingSwarmLaunchRequest,
        *,
        profile: Optional[CodingSwarmProfile],
        preset_key: str,
    ) -> CodingSwarmLaunchRequest:
        if profile is None:
            return request

        preset = self.get_preset(preset_key)
        profile_preset_key = (
            str(getattr(profile, "preset_key", "") or "").strip().lower()
        )
        if profile_preset_key and profile_preset_key != preset_key:
            raise AgentCodingSwarmLaunchError(
                (
                    "Selected coding swarm profile is for preset "
                    f"'{profile_preset_key}', not '{preset_key}'"
                ),
                status_code=422,
            )
        if str(
            getattr(profile, "status", "active") or "active"
        ).strip().lower() not in {"active", "enabled"}:
            raise AgentCodingSwarmLaunchError(
                "Coding swarm profile is not active",
                status_code=422,
            )

        payload = request.model_dump(exclude_none=False)
        if str(payload.get("source_id") or "") != str(profile.source_id):
            raise AgentCodingSwarmLaunchError(
                "Coding swarm profile source does not match request source",
                status_code=422,
            )

        if not str(payload.get("scope") or "").strip():
            payload["scope"] = (
                str(profile.scope_default or preset.get("default_scope") or "auto")
                .strip()
                .lower()
            )
        if (
            not str(payload.get("search_query") or "").strip()
            and str(profile.saved_search_query or "").strip()
        ):
            payload["search_query"] = str(profile.saved_search_query).strip()
        if not isinstance(payload.get("commands"), list) or not payload.get("commands"):
            payload["commands"] = list(profile.default_commands or []) or None
        if not isinstance(payload.get("file_paths"), list) or not payload.get(
            "file_paths"
        ):
            payload["file_paths"] = list(profile.default_file_paths or []) or None
        if not int(payload.get("max_agents") or 0):
            payload["max_agents"] = int(profile.max_agents or 4)
        payload["profile_id"] = profile.id
        return request.__class__(**payload)

    async def resolve_profile(
        self,
        db: AsyncSession,
        *,
        current_user: User,
        source_id: UUID,
        profile_id: Optional[UUID],
        preset_key: str,
    ) -> Optional[CodingSwarmProfile]:
        if profile_id:
            profile = await db.get(CodingSwarmProfile, profile_id)
            if not profile or not is_profile_visible_to_user(profile, current_user):
                raise AgentCodingSwarmLaunchError(
                    "Coding swarm profile not found",
                    status_code=404,
                )
            if str(profile.source_id) != str(source_id):
                raise AgentCodingSwarmLaunchError(
                    "Coding swarm profile source does not match request source",
                    status_code=422,
                )
            return profile

        return (
            (
                await db.execute(
                    select(CodingSwarmProfile)
                    .where(
                        CodingSwarmProfile.user_id == current_user.id,
                        CodingSwarmProfile.source_id == source_id,
                        CodingSwarmProfile.preset_key == preset_key,
                        CodingSwarmProfile.is_default.is_(True),
                    )
                    .order_by(desc(CodingSwarmProfile.updated_at))
                    .limit(1)
                )
            )
            .scalars()
            .first()
        )

    def build_config(
        self,
        request: CodingSwarmLaunchRequest,
        *,
        source_name: str,
        source_type: str,
        preset_key: str,
    ) -> dict:
        preset = self.get_preset(preset_key)
        scope = (
            str(request.scope or preset.get("default_scope") or "auto").strip().lower()
        )
        symptom = str(request.failure_symptom or "").strip()
        search_query = str(request.search_query or "").strip()
        if not search_query:
            scope_hint = "" if scope == "auto" else scope.replace("_", " ")
            search_suffix = str(preset.get("default_search_suffix") or "").strip()
            search_query = " ".join(
                part for part in [scope_hint, symptom, search_suffix] if part
            ).strip()[:500]

        max_agents = max(1, min(int(request.max_agents or 4), 4))
        swarm_roles = list(
            preset.get("roles") or ["reproducer", "root_cause", "patcher", "verifier"]
        )[:max_agents]
        config: dict[str, Any] = {
            "source_id": str(request.source_id),
            "launch_mode": str(preset.get("launch_mode") or "").strip(),
            "failure_symptom": symptom,
            "scope": scope or str(preset.get("default_scope") or "auto"),
            "quick_start": {
                "profile": str(preset.get("quick_start_profile") or preset_key).strip(),
                "version": "v1",
                "source_name": str(source_name or "").strip(),
                "source_type": str(source_type or "").strip().lower(),
                "scope": scope or str(preset.get("default_scope") or "auto"),
                "autonomy_mode": "max_autonomy",
                "entry_point": "dedicated_quick_start",
                "max_agents": max_agents,
                "roles": swarm_roles,
                "preset_key": preset_key,
                "profile_id": (
                    str(request.profile_id)
                    if getattr(request, "profile_id", None)
                    else None
                ),
            },
            "plan_then_act_enabled": True,
            "plan_max_steps": 6,
            "subgoal_decomposition_enabled": False,
            "swarm_child_jobs_enabled": True,
            "swarm_max_agents": max_agents,
            "swarm_roles": swarm_roles,
            "swarm_inherit_results": True,
            "swarm_inherit_config": True,
            "swarm_fan_in_enabled": True,
            "swarm_fan_in_name": str(
                preset.get("fan_in_name") or "Coding Swarm Fan-In"
            ).strip(),
            "swarm_fan_in_trigger_condition": "on_any_end",
            "coding_swarm_enabled": True,
            "coding_swarm_profile": str(
                preset.get("coding_profile") or preset_key
            ).strip(),
            "coding_swarm_preset_key": preset_key,
            "coding_swarm_auto_promote_best_slice": True,
            "coding_swarm_auto_launch_repair_chain": True,
            "coding_swarm_confidence_threshold": float(
                preset.get("confidence_threshold") or 0.70
            ),
            "coding_swarm_tiebreaker_threshold": float(
                preset.get("tiebreaker_threshold") or 0.50
            ),
            "coding_swarm_repair_chain_name": "repo_bug_triage_repair",
            "create_workspace_from_source": True,
            "emit_execution_plan": True,
            "auto_commands_from_project_profile": True,
            "max_verification_commands": 3,
            "apply_patch_to_kb": False,
            "apply_patch_to_kb_confirm": False,
            "enable_memory": False,
        }
        if search_query:
            config["search_query"] = search_query
        if request.error_output is not None:
            config["error_output"] = str(request.error_output)
        if isinstance(request.file_paths, list):
            config["file_paths"] = [
                str(path).strip() for path in request.file_paths if str(path).strip()
            ]
        if isinstance(request.commands, list):
            config["commands"] = [
                str(command).strip()
                for command in request.commands
                if str(command).strip()
            ]
        if isinstance(request.config_overrides, dict):
            config.update(normalize_scope_config(request.config_overrides) or {})
        return agent_coding_harness_service.apply_launch_defaults(
            normalize_scope_config(config) or {},
            preset_key=preset_key,
        )

    async def launch(
        self,
        *,
        request: CodingSwarmLaunchRequest,
        db: AsyncSession,
        current_user: User,
        preset_key: str,
    ) -> AgentJob:
        preset = self.get_preset(preset_key)
        source = await db.get(DocumentSource, request.source_id)
        if source is None:
            raise AgentCodingSwarmLaunchError(
                "Document source not found",
                status_code=404,
            )

        source_type = str(source.source_type or "").strip().lower()
        if source_type not in {"github", "gitlab"}:
            raise AgentCodingSwarmLaunchError(
                "Quick start requires a github/gitlab document source",
                status_code=422,
            )

        profile = await self.resolve_profile(
            db,
            current_user=current_user,
            source_id=request.source_id,
            profile_id=getattr(request, "profile_id", None),
            preset_key=preset_key,
        )
        profile_grants_access = bool(
            profile is not None and is_profile_visible_to_user(profile, current_user)
        )
        if (
            not current_user.is_admin()
            and not self.is_source_owned_by_user(source, current_user)
            and not profile_grants_access
        ):
            raise AgentCodingSwarmLaunchError(
                "Not authorized for this source",
                status_code=403,
            )

        document_count = int(
            (
                await db.execute(
                    select(func.count()).where(Document.source_id == source.id)
                )
            ).scalar()
            or 0
        )
        if document_count <= 0:
            raise AgentCodingSwarmLaunchError(
                "Source has no documents; ingest/sync the repository first",
                status_code=422,
            )

        merged_request = self.merge_request_with_profile(
            request,
            profile=profile,
            preset_key=preset_key,
        )
        unsafe_commands = agent_job_creation_service.find_unsafe_commands(
            merged_request.commands
        )
        if unsafe_commands:
            raise AgentCodingSwarmLaunchError(
                {
                    "message": (
                        "Quick start rejected potentially destructive command(s)"
                    ),
                    "blocked_commands": unsafe_commands,
                },
                status_code=422,
            )

        config = self.build_config(
            merged_request,
            source_name=str(source.name or ""),
            source_type=source_type,
            preset_key=preset_key,
        )
        display_name = str(preset.get("display_name") or "Coding Swarm").strip()
        job_name = str(merged_request.name or "").strip() or (
            f"{display_name} - {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        job = AgentJob(
            name=job_name,
            description=(
                f"{display_name} with automatic fan-in and " "repair-chain handoff."
            ),
            job_type="analysis",
            goal=self.build_goal(merged_request, preset_key=preset_key),
            config=config,
            user_id=current_user.id,
            status=AgentJobStatus.PENDING.value,
            enable_memory=agent_job_creation_service.extract_enable_memory(
                config,
                default=False,
            ),
            max_iterations=90,
            max_tool_calls=360,
            max_llm_calls=140,
            max_runtime_minutes=120,
        )
        store_swarm_collaboration(
            job,
            build_collaboration_payload(
                owner_user_id=current_user.id,
                visibility=normalize_profile_visibility(
                    getattr(profile, "visibility", "private") if profile else "private"
                ),
                shared_with_user_ids=(
                    normalize_uuid_str_list(
                        getattr(profile, "shared_with_user_ids", None),
                        200,
                    )
                    if profile
                    else []
                ),
            ),
        )

        db.add(job)
        await db.flush()
        agent_coding_workspace_session_service.bind_job(job)
        if profile is not None:
            profile.latest_job_id = job.id
            db.add(profile)
        agent_job_creation_service.append_launch_log_if_present(job)
        await db.commit()
        await db.refresh(job)
        return job


agent_coding_swarm_launch_service = AgentCodingSwarmLaunchService()
