"""Present autonomous job models as stable API response contracts."""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from uuid import UUID

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.schemas.agent_job import AgentJobResponse
from app.services.agent_job_scheduler_state import extract_scheduler_state
from app.services.agent_scope_service import (
    normalize_scope_config,
    normalize_scope_keys_deep,
)
from app.services.operator_interventions import (
    derive_operator_interventions_with_outcomes,
)

from .relaunch_lineage import extract_parent_job_id


@dataclass(frozen=True)
class JobPresenterDependencies:
    extract_launch_mode: Callable[[dict], str]
    extract_promotion: Callable[[AgentJob], dict[str, Any]]
    extract_swarm_summary: Callable[..., Any]
    extract_goal_contract_summary: Callable[[AgentJob], dict | None]
    extract_approval_checkpoint: Callable[[AgentJob], dict | None]
    extract_executive_digest: Callable[[AgentJob], dict | None]


def present_job(
    job: AgentJob,
    *,
    deps: JobPresenterDependencies,
    relaunch_children_count: int = 0,
    current_user_id: str | None = None,
    user_lookup: dict[str, User] | None = None,
) -> AgentJobResponse:
    """Convert an AgentJob model to its public response schema."""
    config = job.config if isinstance(job.config, dict) else {}
    results = job.results if isinstance(job.results, dict) else {}
    now = datetime.utcnow()
    experiment_run = (
        results.get("experiment_run")
        if isinstance(results.get("experiment_run"), dict)
        else None
    )
    experiment_runs_raw = (
        results.get("experiment_runs")
        if isinstance(results.get("experiment_runs"), list)
        else []
    )
    experiment_runs = [row for row in experiment_runs_raw if isinstance(row, dict)]
    execution_strategy = (
        results.get("execution_strategy")
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
    scheduler_state = extract_scheduler_state(job)
    operator_interventions_raw = (
        execution_strategy.get("operator_interventions")
        if isinstance(execution_strategy.get("operator_interventions"), list)
        else []
    )
    effective_intervention_status = job.status
    if operator_interventions_raw:
        last_intervention = next(
            (
                row
                for row in reversed(operator_interventions_raw)
                if isinstance(row, dict)
            ),
            None,
        )
        if isinstance(last_intervention, dict):
            action = str(last_intervention.get("action") or "").strip().lower()
            derived_status = str(
                last_intervention.get("job_status_after") or ""
            ).strip()
            if derived_status and action in {"pause", "reject"}:
                effective_intervention_status = derived_status
    operator_interventions = derive_operator_interventions_with_outcomes(
        [row for row in operator_interventions_raw if isinstance(row, dict)],
        current_status=effective_intervention_status,
        completed_at=job.completed_at,
        status_values={
            "completed": AgentJobStatus.COMPLETED.value,
            "failed": AgentJobStatus.FAILED.value,
            "cancelled": AgentJobStatus.CANCELLED.value,
            "pending": AgentJobStatus.PENDING.value,
            "running": AgentJobStatus.RUNNING.value,
            "paused": AgentJobStatus.PAUSED.value,
        },
    )
    if operator_interventions and isinstance(
        execution_strategy.get("operator_interventions"), list
    ):
        execution_strategy["operator_interventions"] = operator_interventions

    promotion = deps.extract_promotion(job)
    promoted_profile_id = _optional_uuid(
        promotion.get("domain_research_profile_id")
        or promotion.get("promoted_domain_research_profile_id")
    )
    promoted_portfolio_id = _optional_uuid(
        promotion.get("research_portfolio_id")
        or promotion.get("promoted_research_portfolio_id")
    )
    return AgentJobResponse(
        id=job.id or uuid.uuid4(),
        name=job.name,
        description=job.description,
        job_type=job.job_type,
        goal=job.goal,
        goal_criteria=job.goal_criteria,
        config=normalize_scope_config(job.config),
        launch_mode=deps.extract_launch_mode(config) or None,
        relaunch_from_job_id=extract_parent_job_id(config),
        relaunch_children_count=max(0, int(relaunch_children_count or 0)),
        promotion_status=str(promotion.get("status") or "").strip() or None,
        promoted_domain_research_profile_id=promoted_profile_id,
        promoted_research_portfolio_id=promoted_portfolio_id,
        agent_definition_id=job.agent_definition_id,
        agent_definition_name=job.agent_definition.name
        if job.agent_definition
        else None,
        user_id=job.user_id,
        status=job.status,
        progress=int(job.progress or 0),
        current_phase=job.current_phase,
        phase_details=job.phase_details,
        iteration=int(job.iteration or 0),
        max_iterations=int(job.max_iterations or 0),
        max_tool_calls=int(job.max_tool_calls or 0),
        max_llm_calls=int(job.max_llm_calls or 0),
        max_runtime_minutes=int(job.max_runtime_minutes or 0),
        tool_calls_used=int(job.tool_calls_used or 0),
        llm_calls_used=int(job.llm_calls_used or 0),
        tokens_used=int(job.tokens_used or 0),
        error=job.error,
        error_count=int(job.error_count or 0),
        schedule_type=job.schedule_type,
        schedule_cron=job.schedule_cron,
        next_run_at=job.next_run_at,
        scheduler_state=scheduler_state,
        results=results or None,
        experiment_run=experiment_run,
        experiment_runs=experiment_runs or None,
        operator_interventions=operator_interventions or None,
        output_artifacts=job.output_artifacts,
        created_at=job.created_at or now,
        started_at=job.started_at,
        completed_at=job.completed_at,
        last_activity_at=job.last_activity_at,
        celery_task_id=job.celery_task_id,
        execution_lease_owner=job.execution_lease_owner,
        execution_lease_expires_at=job.execution_lease_expires_at,
        execution_lease_heartbeat_at=job.execution_lease_heartbeat_at,
        execution_fence=int(job.execution_fence or 0),
        parent_job_id=job.parent_job_id,
        root_job_id=job.root_job_id,
        chain_depth=int(job.chain_depth or 0),
        chain_triggered=bool(job.chain_triggered),
        chain_config=normalize_scope_keys_deep(job.chain_config),
        swarm_summary=deps.extract_swarm_summary(
            job, current_user_id=current_user_id, user_lookup=user_lookup
        ),
        goal_contract_summary=deps.extract_goal_contract_summary(job),
        approval_checkpoint=deps.extract_approval_checkpoint(job),
        executive_digest=deps.extract_executive_digest(job),
    )


def _optional_uuid(value: Any) -> UUID | None:
    text = str(value or "").strip()
    return UUID(text) if re.fullmatch(r"[0-9a-fA-F-]{36}", text) else None
