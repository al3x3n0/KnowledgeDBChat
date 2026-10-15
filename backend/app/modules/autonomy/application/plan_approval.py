"""Approving a plan a run proposed, from outside the HTTP layer.

A person approves through ``POST /agent-jobs/{id}/action``, whose
dependencies are assembled in the endpoint module, which nothing below the
API may import. A campaign approving a plan its own job proposed must take
the same path rather than a copy of it, or the two would drift exactly where
it matters -- the audit event, the cleared checkpoint, which stages start.
So this builds the dependencies a checkpoint *decision* needs from the
application layer and calls ``perform_checkpoint_decision``, the function
the endpoint calls. The parts only other actions use (relaunch, swarm,
visibility) are not available here and say so if anything asks for them.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobCheckpoint
from app.models.user import User
from app.modules.autonomy.application import (
    job_action_checkpoints,
    job_action_interventions,
    job_action_plan_state,
)
from app.modules.autonomy.application.job_action_checkpoint_decisions import (
    perform_checkpoint_decision,
)
from app.modules.autonomy.application.job_action_contracts import JobActionDependencies
from app.schemas.agent_job import AgentJobActionRequest
from app.services import job_dispatch


async def load_latest_checkpoint(
    job_id: UUID, db: AsyncSession
) -> Optional[AgentJobCheckpoint]:
    result = await db.execute(
        select(AgentJobCheckpoint)
        .where(AgentJobCheckpoint.job_id == job_id)
        .order_by(AgentJobCheckpoint.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


class _TaskByPath:
    """``.delay`` for a task named by path, so nothing here imports the worker."""

    def __init__(self, path: str):
        self.path = path

    def delay(self, *args: Any, **kwargs: Any) -> Any:
        return job_dispatch.send_now(self.path, *args, **kwargs)


def _not_for_decisions(name: str):
    def _refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{name} is not available outside the HTTP layer; a checkpoint "
            "decision should not need it"
        )

    return _refuse


def decision_dependencies() -> JobActionDependencies:
    """What a checkpoint decision needs, from the application layer only."""
    return JobActionDependencies(
        is_job_visible=_not_for_decisions("is_job_visible"),
        approval_payload_from_results=job_action_checkpoints.approval_payload_from_results,
        load_latest_checkpoint=load_latest_checkpoint,
        append_operator_intervention=job_action_interventions.append_operator_intervention,
        append_step_event=job_action_checkpoints.append_step_event,
        normalize_checkpoint_action_patch=(
            job_action_checkpoints.normalize_checkpoint_action_patch
        ),
        apply_checkpoint_action_patch=job_action_checkpoints.apply_checkpoint_action_patch,
        set_current_plan_step_status=job_action_plan_state.set_current_plan_step_status,
        append_approval_event=job_action_checkpoints.append_approval_event,
        sync_execution_strategy_state=job_action_checkpoints.sync_execution_strategy_state,
        quick_start_relaunch_dispatcher=None,
        infer_coding_swarm_preset_key=_not_for_decisions(
            "infer_coding_swarm_preset_key"
        ),
        extract_swarm_collaboration=_not_for_decisions("extract_swarm_collaboration"),
        build_swarm_collaboration_payload=_not_for_decisions(
            "build_swarm_collaboration_payload"
        ),
        store_swarm_collaboration=_not_for_decisions("store_swarm_collaboration"),
        execute_agent_job_task=_TaskByPath(job_dispatch.AGENT_JOB_TASK),
        generate_job_summary=_TaskByPath(
            "app.tasks.agent_job_tasks.generate_job_summary"
        ),
    )


async def approve_proposed_plan(
    job: AgentJob, db: AsyncSession, *, approver: User, note: str
) -> AgentJob:
    """Approve the plan this paused run proposed, as ``approver``, with ``note``
    recorded where a person's approval note would be."""
    return await perform_checkpoint_decision(
        job,
        "approve",
        AgentJobActionRequest(action="approve"),
        note,
        deps=decision_dependencies(),
        db=db,
        current_user=approver,
    )
