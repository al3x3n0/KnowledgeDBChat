"""Swarm collaboration actions for the autonomous-job action state machine."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

SWARM_ACTIONS = frozenset(
    {
        "assign_swarm_review",
        "clear_swarm_assignment",
        "update_swarm_review_note",
        "launch_tie_breaker",
        "promote_swarm_candidate",
    }
)


async def perform_swarm_action(
    job: AgentJob,
    action: str,
    action_payload: dict[str, Any],
    checkpoint_note: str | None,
    *,
    deps: JobActionDependencies,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    if action == "assign_swarm_review":
        if not deps.infer_coding_swarm_preset_key(job):
            raise JobActionError(
                status_code=400,
                detail="Assignment is only available on coding swarm jobs",
            )
        assigned_user_id = str(
            action_payload.get("assigned_user_id") or current_user.id
        ).strip()
        try:
            assigned_user = await db.get(User, UUID(assigned_user_id))
        except Exception:
            assigned_user = None
        if assigned_user is None or not bool(
            getattr(assigned_user, "is_active", False)
        ):
            raise JobActionError(
                status_code=422,
                detail="Assigned user not found",
            )
        collaboration = deps.extract_swarm_collaboration(job)
        collaboration = deps.build_swarm_collaboration_payload(
            owner_user_id=collaboration.get("owner_user_id") or job.user_id,
            visibility="shared",
            shared_with_user_ids=[
                *list(collaboration.get("shared_with_user_ids") or []),
                assigned_user_id,
            ],
            assigned_user_id=assigned_user_id,
            assigned_by_user_id=str(current_user.id),
            assigned_at=datetime.utcnow().isoformat(),
            review_note=str(collaboration.get("review_note") or "").strip() or None,
        )
        deps.store_swarm_collaboration(job, collaboration)
        job.add_log_entry(
            {
                "phase": "swarm_review_assigned",
                "result": {"assigned_user_id": assigned_user_id},
            }
        )

    elif action == "clear_swarm_assignment":
        if not deps.infer_coding_swarm_preset_key(job):
            raise JobActionError(
                status_code=400,
                detail="Assignment is only available on coding swarm jobs",
            )
        collaboration = deps.extract_swarm_collaboration(job)
        collaboration = deps.build_swarm_collaboration_payload(
            owner_user_id=collaboration.get("owner_user_id") or job.user_id,
            visibility="shared"
            if bool(collaboration.get("shared_review"))
            else "private",
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
            review_note=str(collaboration.get("review_note") or "").strip() or None,
        )
        deps.store_swarm_collaboration(job, collaboration)
        job.add_log_entry(
            {"phase": "swarm_assignment_cleared", "reason": "user_request"}
        )

    elif action == "update_swarm_review_note":
        if not deps.infer_coding_swarm_preset_key(job):
            raise JobActionError(
                status_code=400,
                detail="Review notes are only available on coding swarm jobs",
            )
        collaboration = deps.extract_swarm_collaboration(job)
        collaboration = deps.build_swarm_collaboration_payload(
            owner_user_id=collaboration.get("owner_user_id") or job.user_id,
            visibility="shared"
            if bool(collaboration.get("shared_review"))
            else "private",
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
            assigned_user_id=str(collaboration.get("assigned_user_id") or "").strip()
            or None,
            assigned_by_user_id=str(
                collaboration.get("assigned_by_user_id") or ""
            ).strip()
            or None,
            assigned_at=str(collaboration.get("assigned_at") or "").strip() or None,
            review_note=str(action_payload.get("review_note") or "").strip() or None,
        )
        deps.store_swarm_collaboration(job, collaboration)
        job.add_log_entry(
            {"phase": "swarm_review_note_updated", "reason": "user_request"}
        )

    elif action == "launch_tie_breaker":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
            AgentJobStatus.PAUSED.value,
        ]:
            raise JobActionError(
                status_code=400,
                detail="Can only launch a tie-breaker from completed, failed, cancelled, or paused fan-in jobs",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        fan_in = (
            results_payload.get("swarm_fan_in")
            if isinstance(results_payload.get("swarm_fan_in"), dict)
            else {}
        )
        cfg = job.config if isinstance(job.config, dict) else {}
        inherited = (
            cfg.get("inherited_data")
            if isinstance(cfg.get("inherited_data"), dict)
            else {}
        )
        swarm_payload = (
            inherited.get("swarm") if isinstance(inherited.get("swarm"), dict) else {}
        )
        if (
            not fan_in
            or not swarm_payload
            or not bool(cfg.get("coding_swarm_enabled") or fan_in)
        ):
            raise JobActionError(
                status_code=400,
                detail="Bug triage swarm tie-breaker is only available on coding swarm fan-in jobs with inherited sibling data",
            )
        executor = AutonomousAgentExecutor()
        new_job = await executor._launch_bug_triage_swarm_tie_breaker_job(
            fan_in_job=job,
            db=db,
            merged=fan_in,
            swarm_payload=swarm_payload,
        )
        if new_job is None:
            raise JobActionError(
                status_code=422,
                detail="Failed to launch verifier tie-breaker",
            )
        fan_in["review_state"] = "tie_break_running"
        fan_in["review_required"] = False
        fan_in["review_reason"] = str(
            fan_in.get("review_reason") or "Verifier tie-breaker running."
        )
        fan_in["tie_breaker_job_id"] = str(new_job.id)
        fan_in["tie_breaker_attempted"] = True
        results_payload["swarm_fan_in"] = fan_in
        deps.append_operator_intervention(
            results_payload,
            action="launch_tie_breaker",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=job.status,
            metadata={"new_job_id": str(new_job.id)},
        )
        job.results = results_payload
        flag_modified(job, "results")
        job.add_log_entry(
            {
                "phase": "tie_breaker_requested",
                "reason": "user_request",
                "result": {"new_job_id": str(new_job.id)},
            }
        )
        await db.commit()
        deps.execute_agent_job_task.delay(str(new_job.id), str(current_user.id))
        return new_job

    elif action == "promote_swarm_candidate":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
            AgentJobStatus.PAUSED.value,
        ]:
            raise JobActionError(
                status_code=400,
                detail="Can only promote a swarm candidate from completed, failed, cancelled, or paused fan-in jobs",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        fan_in = (
            results_payload.get("swarm_fan_in")
            if isinstance(results_payload.get("swarm_fan_in"), dict)
            else {}
        )
        cfg = job.config if isinstance(job.config, dict) else {}
        if not fan_in or not bool(cfg.get("coding_swarm_enabled") or fan_in):
            raise JobActionError(
                status_code=400,
                detail="Manual promotion is only available on bug triage swarm fan-in jobs",
            )
        candidate_rows = (
            fan_in.get("candidate_paths")
            if isinstance(fan_in.get("candidate_paths"), list)
            else []
        )
        candidate_job_id = str(action_payload.get("candidate_job_id") or "").strip()
        if not candidate_job_id and candidate_rows:
            try:
                candidate_index = int(action_payload.get("candidate_index", 0) or 0)
            except Exception:
                candidate_index = 0
            if 0 <= candidate_index < len(candidate_rows) and isinstance(
                candidate_rows[candidate_index], dict
            ):
                candidate_job_id = str(
                    candidate_rows[candidate_index].get("job_id") or ""
                ).strip()
        executor = AutonomousAgentExecutor()
        new_job = await executor._launch_bug_triage_swarm_repair_job(
            fan_in_job=job,
            db=db,
            merged=fan_in,
            candidate_job_id=candidate_job_id,
            manual_promotion=True,
        )
        if new_job is None:
            raise JobActionError(
                status_code=422,
                detail="Failed to launch repair chain from the selected swarm candidate",
            )
        fan_in["repair_chain_job_id"] = str(new_job.id)
        fan_in["review_state"] = "manual_promotion"
        fan_in["review_required"] = False
        fan_in["promotion_reason"] = (
            f"Manually promoted swarm candidate {candidate_job_id[:8]} into the repair chain."
            if candidate_job_id
            else "Manually promoted the leading swarm candidate into the repair chain."
        )
        results_payload["swarm_fan_in"] = fan_in
        deps.append_operator_intervention(
            results_payload,
            action="promote_swarm_candidate",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=job.status,
            metadata={
                "new_job_id": str(new_job.id),
                "candidate_job_id": candidate_job_id or None,
            },
        )
        job.results = results_payload
        flag_modified(job, "results")
        job.add_log_entry(
            {
                "phase": "swarm_candidate_promoted",
                "reason": "user_request",
                "result": {
                    "new_job_id": str(new_job.id),
                    "candidate_job_id": candidate_job_id or None,
                },
            }
        )
        await db.commit()
        deps.execute_agent_job_task.delay(str(new_job.id), str(current_user.id))
        return new_job

    return job
