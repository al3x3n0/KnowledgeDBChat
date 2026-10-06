"""Autonomous-job tools: the ``scheduling`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)

#: How many scheduled jobs and notifications one run may leave behind it.
MAX_SCHEDULED_JOBS_PER_RUN = 10


def build_autonomous_scheduling_provider(executor: Any) -> FunctionToolProvider:
    """Scheduling helpers for AutonomousAgentExecutor."""

    async def _schedule_job(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timezone

        from sqlalchemy import func

        from app.models.agent_job import AgentJob as AgentJobModel
        from app.models.agent_job import AgentJobType

        job = ctx.job
        goal = str(params.get("goal", "")).strip()
        schedule_type = str(params.get("schedule_type", "")).strip().lower()
        if not goal:
            return {"error": "goal is required"}
        if schedule_type not in {"once", "recurring"}:
            return {"error": "schedule_type must be 'once' or 'recurring'"}
        job_type_param = str(params.get("job_type") or "research").strip().lower()
        known_job_types = {t.value for t in AgentJobType} | {"coding"}
        if job_type_param not in known_job_types:
            return {
                "error": f"job_type must be one of {sorted(known_job_types)}, "
                f"not {job_type_param!r}"
            }
        try:
            # A run that schedules jobs which schedule jobs has no natural
            # end, so one run may leave only so many behind it.
            already = (
                await ctx.db.execute(
                    select(func.count(AgentJobModel.id)).where(
                        AgentJobModel.parent_job_id == job.id,
                        AgentJobModel.schedule_type.isnot(None),
                    )
                )
            ).scalar() or 0
            if already >= MAX_SCHEDULED_JOBS_PER_RUN:
                return {
                    "error": f"This run has already scheduled {already} jobs "
                    f"(limit {MAX_SCHEDULED_JOBS_PER_RUN}). Cancel one first."
                }
            config_param = (
                params.get("config") if isinstance(params.get("config"), dict) else {}
            )
            next_run = None
            cron_expr = None
            if schedule_type == "once":
                run_at = str(params.get("run_at", "")).strip()
                if not run_at:
                    return {"error": "run_at is required for schedule_type=once"}
                next_run = datetime.fromisoformat(run_at)
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)
            else:
                from croniter import croniter

                cron_expr = str(params.get("cron", "")).strip()
                if not cron_expr:
                    return {"error": "cron is required for schedule_type=recurring"}
                if not croniter.is_valid(cron_expr):
                    return {"error": f"Invalid cron expression: {cron_expr}"}
                next_run = croniter(cron_expr, datetime.now(timezone.utc)).get_next(
                    datetime
                )
            new_job = AgentJobModel(
                user_id=job.user_id,
                # Required by the table; it was never set, so every call
                # was refused by the database.
                name=f"Scheduled: {goal}"[:200],
                goal=goal[:2000],
                job_type=job_type_param,
                schedule_type=schedule_type,
                schedule_cron=cron_expr,
                next_run_at=next_run,
                status="pending",
                config=config_param,
                parent_job_id=job.id,
            )
            # In a savepoint: this session is the run's, and a refused insert
            # would otherwise leave it unusable for every later tool.
            async with ctx.db.begin_nested():
                ctx.db.add(new_job)
                await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "id": str(new_job.id),
                    "goal": new_job.goal,
                    "job_type": new_job.job_type,
                    "schedule_type": schedule_type,
                    "next_run_at": next_run.isoformat(),
                    "cron": cron_expr,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to schedule job: {exc}"}

    async def _cancel_scheduled_job(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        cancel_job_id = str(params.get("job_id", "")).strip()
        if not cancel_job_id:
            return {"error": "job_id is required"}
        try:
            target = await ctx.db.get(AgentJobModel, _UUID(cancel_job_id))
            if not target:
                return {"error": f"Job not found: {cancel_job_id}"}
            if target.user_id != job.user_id:
                return {"error": "Not authorized to cancel this job"}
            if target.status == "running":
                return {"error": "Cannot cancel a currently running job"}
            if target.schedule_type is None:
                # Without this a finished one-shot job was rewritten to
                # "cancelled", losing the record that it had completed.
                return {"error": f"Job {cancel_job_id} is not a scheduled job"}
            target.status = "cancelled"
            target.next_run_at = None
            target.schedule_type = None
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "id": str(target.id),
                    "status": "cancelled",
                    "goal": target.goal,
                },
            }
        except ValueError:
            return {"error": f"Invalid job_id format: {cancel_job_id}"}
        except Exception as exc:
            return {"error": f"Failed to cancel job: {exc}"}

    return FunctionToolProvider(
        name="autonomous_scheduling_tools",
        modes={"autonomous"},
        handlers={
            "schedule_job": _schedule_job,
            "cancel_scheduled_job": _cancel_scheduled_job,
        },
    )
