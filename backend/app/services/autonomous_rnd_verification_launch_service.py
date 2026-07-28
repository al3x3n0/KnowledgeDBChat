"""Approval-backed materialization of autonomous R&D verification tasks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.document import DocumentSource
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.research_note import ResearchNote
from app.models.tool_audit import ToolExecutionAudit
from app.models.user import User
from app.services.scientific_validation_service import (
    build_scientific_validation_recipe,
    get_scientific_sandbox_profile,
    get_scientific_validation_runtime_limits,
)

_LAUNCH_NAMESPACE = uuid.UUID("f090e324-858d-44b6-b32d-e775f465ed54")


class VerificationLaunchError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class VerificationLaunchResult:
    plan: ExperimentPlan
    run: ExperimentRun
    job: AgentJob
    audit: ToolExecutionAudit
    created: bool


class AutonomousRnDVerificationLaunchService:
    """Create one idempotent, budget-bounded local verification execution."""

    async def launch(
        self,
        *,
        parent_job: AgentJob,
        task: Dict[str, Any],
        current_user: User,
        db: AsyncSession,
        research_note_id: uuid.UUID,
        source_id: uuid.UUID,
        sandbox_profile_id: str,
        commands: List[str],
        repeat_count: int,
        timeout_seconds: int,
        max_runtime_minutes: int,
        budget_limit: float,
        approval_note: str,
    ) -> VerificationLaunchResult:
        if parent_job.user_id != current_user.id:
            raise VerificationLaunchError(
                "Parent agent job was not found", status_code=404
            )
        if str(parent_job.status or "").lower() not in {"completed", "failed"}:
            raise VerificationLaunchError(
                "Verification tasks can only launch from finalized jobs"
            )
        task_id = str(task.get("id") or "").strip()
        evidence_id = str(task.get("evidence_id") or "").strip()
        if not task_id or not evidence_id:
            raise VerificationLaunchError("Verification task is malformed")
        if task.get("approval_required") is not True:
            raise VerificationLaunchError(
                "Verification task does not declare an approval boundary"
            )

        plan_id, run_id, job_id, audit_id = self._ids(
            current_user.id, parent_job.id, task_id
        )
        existing = await self._existing_launch(
            db=db,
            user_id=current_user.id,
            plan_id=plan_id,
            run_id=run_id,
            job_id=job_id,
            audit_id=audit_id,
        )
        if existing is not None:
            return existing

        note = await db.get(ResearchNote, research_note_id)
        if note is None or note.user_id != current_user.id:
            raise VerificationLaunchError(
                "Research note was not found", status_code=404
            )
        source = await db.get(DocumentSource, source_id)
        if source is None or not bool(source.is_active):
            raise VerificationLaunchError(
                "Active experiment source was not found", status_code=404
            )
        profile = await get_scientific_sandbox_profile(
            db,
            sandbox_profile_id,
            track_type=self._track_type(task),
        )
        if not profile:
            raise VerificationLaunchError(
                "Enabled scientific sandbox profile was not found", status_code=404
            )
        profile_owner = str(profile.get("created_by_user_id") or "").strip()
        if profile_owner and profile_owner != str(current_user.id):
            raise VerificationLaunchError(
                "Scientific sandbox profile was not found", status_code=404
            )

        normalized_commands = self._normalize_commands(commands)
        minimum_repeats = int(
            ((task.get("experiment_spec") or {}).get("repeat_count") or 2)
        )
        if repeat_count < minimum_repeats:
            raise VerificationLaunchError(
                f"repeat_count must be at least {minimum_repeats} for this task"
            )
        if float(budget_limit) > float(profile.get("budget_limit_default") or 0):
            raise VerificationLaunchError(
                "budget_limit exceeds the selected sandbox profile limit"
            )
        profile_timeout = int(profile.get("timeout_seconds") or timeout_seconds)
        runtime_limits = get_scientific_validation_runtime_limits()
        maximum_timeout = min(
            profile_timeout,
            int(runtime_limits.get("max_timeout_seconds") or profile_timeout),
        )
        if timeout_seconds > maximum_timeout:
            raise VerificationLaunchError(
                f"timeout_seconds exceeds the allowed limit of {maximum_timeout}"
            )
        worst_case_seconds = len(normalized_commands) * repeat_count * timeout_seconds
        if worst_case_seconds > max_runtime_minutes * 60:
            raise VerificationLaunchError(
                "Requested command repetitions exceed max_runtime_minutes"
            )

        recipe = build_scientific_validation_recipe(
            track_type=self._track_type(task),
            objective=str(task.get("objective") or "Verify external evidence"),
            hypothesis_title=f"Verify {evidence_id}"[:240],
            hypothesis_text=(
                "Independently verify or reject the external evidence referenced by "
                f"{evidence_id}."
            ),
            verification_commands=normalized_commands,
            supporting_evidence=[evidence_id],
        )
        accepted_commands = [
            str(item).strip()
            for item in recipe.get("commands") or []
            if str(item).strip()
        ]
        if accepted_commands != normalized_commands:
            raise VerificationLaunchError(
                "One or more commands are outside the selected verification recipe"
            )

        scientific_validation = {
            "validation_kind": "external_evidence_verification",
            "sandbox_profile_id": str(profile.get("id") or ""),
            "recipe_family": str(recipe.get("recipe_family") or ""),
            "recipe_id": str(recipe.get("recipe_id") or ""),
            "recipe_version": int(recipe.get("recipe_version") or 1),
            "artifact_collection_rules": list(
                (task.get("experiment_spec") or {}).get("required_artifact_kinds", [])
            ),
            "commands": accepted_commands,
            "decision_summary": str(recipe.get("decision_summary") or "")[:2000],
            "profile_snapshot": profile,
            "recipe_snapshot": recipe,
            "capability_check": {
                "ok": True,
                "required": ["repo_reconstruction"],
                "satisfied": ["repo_reconstruction"],
                "missing": [],
            },
            "originating_job_id": str(parent_job.id),
            "verification_task_id": task_id,
            "external_evidence_id": evidence_id,
            "budget_limit": float(budget_limit),
            "runtime_limit_minutes": int(max_runtime_minutes),
            "measurement_summary": {
                "status": "pending",
                "repeat_count": repeat_count,
                "artifact_inventory": list(
                    (task.get("experiment_spec") or {}).get(
                        "required_artifact_kinds", []
                    )
                ),
            },
        }
        resource_caps = (
            profile.get("resource_caps")
            if isinstance(profile.get("resource_caps"), dict)
            else {}
        )
        plan_body = {
            "objective": str(task.get("objective") or ""),
            "verification_task": task,
            "execution_handoff": {
                "source_id": str(source.id),
                "commands": accepted_commands,
                "repeat_count": repeat_count,
                "timeout_seconds": timeout_seconds,
                "sandbox_profile_id": str(profile.get("id") or ""),
            },
            "success_criteria": list(
                (task.get("experiment_spec") or {}).get("success_criteria", [])
            ),
        }
        plan = ExperimentPlan(
            id=plan_id,
            user_id=current_user.id,
            research_note_id=note.id,
            title=f"Verification: {evidence_id}"[:500],
            hypothesis_text=(
                f"Local evidence will verify or reject external evidence {evidence_id}."
            ),
            plan=plan_body,
            generator="autonomous_rnd_verification_planner",
            generator_details={
                "parent_job_id": str(parent_job.id),
                "verification_task_id": task_id,
                "external_evidence_id": evidence_id,
                "approval_note": approval_note[:2000],
            },
        )
        run_config = {
            "source_id": str(source.id),
            "commands": accepted_commands,
            "repeat_count": repeat_count,
            "timeout_seconds": timeout_seconds,
            "unsafe_code_exec_backend": str(profile.get("backend") or "docker"),
            "unsafe_code_exec_docker_image": str(profile.get("docker_image") or ""),
            "unsafe_code_exec_max_memory_mb": int(
                resource_caps.get("memory_mb") or 512
            ),
            "unsafe_code_exec_docker_cpus": float(resource_caps.get("cpus") or 1.0),
            "unsafe_code_exec_docker_pids_limit": int(
                resource_caps.get("pids_limit") or 128
            ),
            "scientific_validation": scientific_validation,
            "verification_origin": {
                "parent_job_id": str(parent_job.id),
                "verification_task_id": task_id,
                "external_evidence_id": evidence_id,
            },
        }
        run = ExperimentRun(
            id=run_id,
            user_id=current_user.id,
            experiment_plan_id=plan.id,
            name=f"Local verification: {evidence_id}"[:500],
            status="planned",
            progress=0,
            config=run_config,
            summary="Approved local verification of external-agent evidence.",
        )
        job = AgentJob(
            id=job_id,
            name=f"Verification Run: {evidence_id}"[:200],
            description="Approved deterministic local evidence verification.",
            job_type="analysis",
            goal="Run approved local verification commands and preserve evidence.",
            goal_criteria={
                "required_artifact_kinds": list(
                    (task.get("experiment_spec") or {}).get(
                        "required_artifact_kinds", []
                    )
                ),
                "repeat_count": repeat_count,
                "strict_completion": True,
            },
            config={
                **run_config,
                "deterministic_runner": "experiment_runner",
                "experiment_run_id": str(run.id),
                "experiment_plan_id": str(plan.id),
                "approval": {
                    "approved_by": str(current_user.id),
                    "note": approval_note[:2000],
                },
            },
            user_id=current_user.id,
            parent_job_id=parent_job.id,
            root_job_id=parent_job.root_job_id or parent_job.id,
            chain_depth=int(parent_job.chain_depth or 0) + 1,
            status=AgentJobStatus.PENDING.value,
            max_iterations=1,
            max_tool_calls=0,
            max_llm_calls=0,
            max_runtime_minutes=max_runtime_minutes,
        )
        run.agent_job_id = job.id
        audit = ToolExecutionAudit(
            id=audit_id,
            user_id=current_user.id,
            tool_name="autonomous_rnd:launch_verification_task",
            tool_input={
                "parent_job_id": str(parent_job.id),
                "verification_task_id": task_id,
                "research_note_id": str(note.id),
                "source_id": str(source.id),
                "sandbox_profile_id": str(profile.get("id") or ""),
                "command_count": len(accepted_commands),
                "repeat_count": repeat_count,
                "timeout_seconds": timeout_seconds,
                "max_runtime_minutes": max_runtime_minutes,
                "budget_limit": float(budget_limit),
            },
            tool_output={
                "experiment_plan_id": str(plan.id),
                "experiment_run_id": str(run.id),
                "agent_job_id": str(job.id),
            },
            policy_decision={
                "allowed": True,
                "require_approval": True,
                "approval_source": "explicit_api_confirmation",
            },
            status="completed",
            approval_required=True,
            approval_mode="owner_or_admin",
            approval_status="approved",
            approved_by=current_user.id,
            approved_at=datetime.utcnow(),
            approval_note=approval_note[:2000],
        )
        db.add_all([plan, run, job, audit])
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            existing = await self._existing_launch(
                db=db,
                user_id=current_user.id,
                plan_id=plan_id,
                run_id=run_id,
                job_id=job_id,
                audit_id=audit_id,
            )
            if existing is not None:
                return existing
            raise
        return VerificationLaunchResult(
            plan=plan,
            run=run,
            job=job,
            audit=audit,
            created=True,
        )

    async def _existing_launch(
        self,
        *,
        db: AsyncSession,
        user_id: uuid.UUID,
        plan_id: uuid.UUID,
        run_id: uuid.UUID,
        job_id: uuid.UUID,
        audit_id: uuid.UUID,
    ) -> VerificationLaunchResult | None:
        plan = await db.get(ExperimentPlan, plan_id)
        if plan is None:
            return None
        run = await db.get(ExperimentRun, run_id)
        job = await db.get(AgentJob, job_id)
        audit = await db.get(ToolExecutionAudit, audit_id)
        if (
            plan.user_id != user_id
            or run is None
            or run.user_id != user_id
            or job is None
            or job.user_id != user_id
            or audit is None
            or audit.user_id != user_id
        ):
            raise VerificationLaunchError(
                "Existing verification launch is incomplete", status_code=409
            )
        return VerificationLaunchResult(
            plan=plan,
            run=run,
            job=job,
            audit=audit,
            created=False,
        )

    @staticmethod
    def _ids(
        user_id: uuid.UUID, parent_job_id: uuid.UUID, task_id: str
    ) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, uuid.UUID]:
        launch_key = f"{user_id}:{parent_job_id}:{task_id}"
        return tuple(
            uuid.uuid5(_LAUNCH_NAMESPACE, f"{kind}:{launch_key}")
            for kind in ("plan", "run", "job", "audit")
        )

    @classmethod
    def launch_ids(
        cls, user_id: uuid.UUID, parent_job_id: uuid.UUID, task_id: str
    ) -> Dict[str, uuid.UUID]:
        plan_id, run_id, job_id, audit_id = cls._ids(user_id, parent_job_id, task_id)
        return {
            "experiment_plan_id": plan_id,
            "experiment_run_id": run_id,
            "agent_job_id": job_id,
            "audit_id": audit_id,
        }

    @staticmethod
    def _normalize_commands(commands: List[str]) -> List[str]:
        normalized = [str(item).strip() for item in commands if str(item).strip()]
        if not normalized:
            raise VerificationLaunchError("At least one command is required")
        if len(normalized) > 4:
            raise VerificationLaunchError("At most four commands are allowed")
        if len(set(normalized)) != len(normalized):
            raise VerificationLaunchError("Verification commands must be unique")
        return normalized

    @staticmethod
    def _track_type(task: Dict[str, Any]) -> str:
        capability = str(task.get("capability") or "").lower()
        if "compiler" in capability:
            return "compiler"
        if "microarchitecture" in capability or "performance" in capability:
            return "microarchitecture"
        return "generic"


autonomous_rnd_verification_launch_service = AutonomousRnDVerificationLaunchService()
