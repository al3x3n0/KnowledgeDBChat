"""
Celery tasks for autonomous agent job execution.

Handles background execution of autonomous agent jobs, including:
- Job execution with progress tracking
- Scheduled/recurring job processing
- Job cleanup and maintenance
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from celery import current_task
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.research_inbox_follow_up_service import sync_follow_up_outcome_for_job
from sqlalchemy import select, and_, or_, delete
from sqlalchemy.orm import selectinload


def _scheduler_state(job: AgentJob) -> dict:
    results = job.results if isinstance(job.results, dict) else {}
    execution = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    state = execution.get("scheduler_state") if isinstance(execution.get("scheduler_state"), dict) else {}
    return {
        **state,
        "last_run_status": str(state.get("last_run_status") or "").strip() or None,
        "failure_streak": max(0, int(state.get("failure_streak", 0) or 0)),
    }


def _write_scheduler_state(job: AgentJob, state: dict) -> dict:
    results = dict(job.results) if isinstance(job.results, dict) else {}
    execution = dict(results.get("execution_strategy")) if isinstance(results.get("execution_strategy"), dict) else {}
    execution["scheduler_state"] = state
    results["execution_strategy"] = execution
    job.results = results
    return state


def _mark_scheduler_dispatched(job: AgentJob, *, dispatched_at: datetime) -> None:
    state = _scheduler_state(job)
    state["last_scheduled_at"] = dispatched_at.isoformat()
    state["last_dispatched_at"] = dispatched_at.isoformat()
    state["current_run_started_at"] = dispatched_at.isoformat()
    state["last_run_status"] = "queued"
    state["queue_reason"] = None
    state["backoff_until"] = None
    state["backoff_seconds"] = 0
    _write_scheduler_state(job, state)


def _record_scheduler_outcome(job: AgentJob, *, outcome: str, happened_at: datetime, queue_reason: str | None = None) -> None:
    state = _scheduler_state(job)
    state["last_run_status"] = outcome
    state["current_run_started_at"] = None
    state["queue_reason"] = queue_reason
    if outcome == AgentJobStatus.COMPLETED.value:
        state["failure_streak"] = 0
        state["last_successful_run_at"] = happened_at.isoformat()
        state["last_completed_run_at"] = happened_at.isoformat()
        state["backoff_until"] = None
        state["backoff_seconds"] = 0
    else:
        streak = max(0, int(state.get("failure_streak", 0) or 0)) + 1
        state["failure_streak"] = streak
        state["last_failure_at"] = happened_at.isoformat()
        if str(job.schedule_type or "").strip().lower() in {"recurring", "continuous"}:
            interval_minutes = 30
            try:
                interval_minutes = int(((job.config or {}).get("interval_minutes") or 30))
            except Exception:
                interval_minutes = 30
            interval_minutes = max(1, min(interval_minutes, 24 * 60))
            backoff_seconds = min(interval_minutes * 60 * (2 ** max(0, streak - 1)), 6 * 60 * 60)
            state["backoff_seconds"] = int(backoff_seconds)
            state["backoff_until"] = (happened_at + timedelta(seconds=backoff_seconds)).isoformat()
            job.next_run_at = happened_at + timedelta(seconds=backoff_seconds)
    _write_scheduler_state(job, state)


async def _publish_job_progress(
    job_id: str,
    progress: int,
    phase: str,
    status: str,
    iteration: int = 0,
    phase_details: Optional[str] = None,
    error: Optional[str] = None,
    execution_graph_runtime: Optional[dict] = None,
    scope_observability_runtime: Optional[dict] = None,
):
    """Publish job progress update to Redis for WebSocket subscribers."""
    import redis.asyncio as redis
    from app.core.config import settings

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        channel = f"agent_job:{job_id}:progress"

        message = {
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "phase": phase,
            "status": status,
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if phase_details:
            message["phase_details"] = phase_details
        if error:
            message["error"] = error
        if isinstance(execution_graph_runtime, dict) and execution_graph_runtime:
            message["execution_graph_runtime"] = execution_graph_runtime
        if isinstance(scope_observability_runtime, dict) and scope_observability_runtime:
            message["scope_observability_runtime"] = scope_observability_runtime

        await redis_client.publish(channel, json.dumps(message))
        await redis_client.close()
    except Exception as e:
        logger.warning(f"Failed to publish progress for agent job {job_id}: {e}")


async def _execute_agent_job_async(job_id: str, user_id: str):
    """Async implementation of agent job execution."""
    job_uuid = UUID(job_id)
    session_factory = create_celery_session()

    async with session_factory() as db:
        # Load job from database
        result = await db.execute(
            select(AgentJob)
            .options(selectinload(AgentJob.agent_definition))
            .where(AgentJob.id == job_uuid)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Agent job {job_id} not found")
            return

        # Check if cancelled
        if job.status == AgentJobStatus.CANCELLED.value:
            logger.info(f"Agent job {job_id} was cancelled")
            return

        # Store celery task ID
        if current_task:
            job.celery_task_id = current_task.request.id
            await db.commit()

        await _publish_job_progress(job_id, 0, "starting", "running")

        try:
            # Initialize executor
            executor = AutonomousAgentExecutor()

            # Progress callback that publishes to Redis
            async def progress_callback(progress_data: dict):
                # Check for cancellation
                await db.refresh(job)
                if job.status == AgentJobStatus.CANCELLED.value:
                    raise Exception("Job cancelled by user")

                await _publish_job_progress(
                    job_id=job_id,
                    progress=progress_data.get("progress", 0),
                    phase=progress_data.get("phase", "running"),
                    status="running",
                    iteration=progress_data.get("iteration", 0),
                    phase_details=progress_data.get("phase_details"),
                    execution_graph_runtime=(
                        progress_data.get("execution_graph_runtime")
                        if isinstance(progress_data.get("execution_graph_runtime"), dict)
                        else None
                    ),
                    scope_observability_runtime=(
                        progress_data.get("scope_observability_runtime")
                        if isinstance(progress_data.get("scope_observability_runtime"), dict)
                        else None
                    ),
                )

            # Execute the job
            result = await executor.execute_job(
                job_id=job_uuid,
                db=db,
                progress_callback=progress_callback,
            )

            # Publish completion
            final_status = result.get("status", "completed")
            _record_scheduler_outcome(
                job,
                outcome=str(final_status or AgentJobStatus.COMPLETED.value),
                happened_at=datetime.utcnow(),
                queue_reason=None,
            )
            await sync_follow_up_outcome_for_job(db, job)
            await db.commit()
            await _publish_job_progress(
                job_id=job_id,
                progress=result.get("progress", 100),
                phase="completed",
                status=final_status,
                iteration=result.get("iterations", 0),
            )

            logger.info(f"Agent job {job_id} completed with status: {final_status}")

        except Exception as e:
            # Update job as failed
            job.status = AgentJobStatus.FAILED.value
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            _record_scheduler_outcome(
                job,
                outcome=AgentJobStatus.FAILED.value,
                happened_at=job.completed_at,
                queue_reason="execution_failure",
            )
            await sync_follow_up_outcome_for_job(db, job)
            await db.commit()

            await _publish_job_progress(
                job_id=job_id,
                progress=job.progress,
                phase="error",
                status="failed",
                error=str(e),
            )

            logger.error(f"Agent job {job_id} failed: {e}")


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def execute_agent_job_task(self, job_id: str, user_id: str):
    """
    Celery task for executing an autonomous agent job.

    This task:
    1. Loads the agent job from the database
    2. Runs the autonomous execution loop
    3. Publishes progress to Redis for WebSocket subscribers
    4. Updates job status on completion/failure

    Args:
        job_id: UUID of the AgentJob
        user_id: UUID of the user who created the job
    """
    logger.info(f"Starting autonomous agent job execution for {job_id}")

    try:
        asyncio.run(_execute_agent_job_async(job_id, user_id))

    except Exception as e:
        logger.exception(f"Agent job task failed for {job_id}")

        async def _mark_failed():
            job_uuid = UUID(job_id)
            session_factory = create_celery_session()
            async with session_factory() as db:
                result = await db.execute(
                    select(AgentJob).where(AgentJob.id == job_uuid)
                )
                job = result.scalar_one_or_none()
                if job and job.status not in (
                    AgentJobStatus.COMPLETED.value,
                    AgentJobStatus.FAILED.value,
                    AgentJobStatus.CANCELLED.value,
                ):
                    job.status = AgentJobStatus.FAILED.value
                    job.error = f"Task error: {str(e)}"
                    job.completed_at = datetime.utcnow()
                    _record_scheduler_outcome(
                        job,
                        outcome=AgentJobStatus.FAILED.value,
                        happened_at=job.completed_at,
                        queue_reason="task_failure",
                    )
                    await sync_follow_up_outcome_for_job(db, job)
                    await db.commit()

        try:
            asyncio.run(_mark_failed())
        except Exception:
            logger.warning("Failed to persist agent job task failure status")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)


@celery_app.task
def process_scheduled_agent_jobs():
    """
    Process scheduled agent jobs that are due.

    This task scans for jobs with next_run_at in the past and queues them.
    Called periodically via Celery Beat.
    """
    logger.info("Scanning for scheduled agent jobs")

    async def _process_scheduled():
        session_factory = create_celery_session()
        async with session_factory() as db:
            now = datetime.utcnow()

            # Find jobs that are due
            result = await db.execute(
                select(AgentJob).where(
                    and_(
                        AgentJob.schedule_type.isnot(None),
                        AgentJob.next_run_at <= now,
                        AgentJob.status.in_([
                            AgentJobStatus.PENDING.value,
                            AgentJobStatus.COMPLETED.value,
                            AgentJobStatus.FAILED.value,
                        ]),
                    )
                )
                .with_for_update(skip_locked=True)
            )
            due_jobs = result.scalars().all()

            for job in due_jobs:
                logger.info(f"Queuing scheduled agent job {job.id}")

                # Reset transient run state so the job can execute again.
                job.status = AgentJobStatus.PENDING.value
                job.progress = 0
                job.current_phase = None
                job.phase_details = None
                job.started_at = None
                job.completed_at = None
                job.error = None
                job.celery_task_id = None
                job.last_activity_at = now
                _mark_scheduler_dispatched(job, dispatched_at=now)

                # Update next_run_at for recurring jobs
                if job.schedule_type == "recurring" and job.schedule_cron:
                    # Parse cron and calculate next run
                    try:
                        from croniter import croniter
                        cron = croniter(job.schedule_cron, now)
                        job.next_run_at = cron.get_next(datetime)
                    except Exception as e:
                        logger.error(f"Failed to calculate next run for job {job.id}: {e}")
                        job.next_run_at = None
                elif job.schedule_type == "continuous":
                    # Simple interval scheduling (minutes) stored in job.config.interval_minutes.
                    try:
                        interval = int(((job.config or {}).get("interval_minutes") or 30))
                    except Exception:
                        interval = 30
                    interval = max(1, min(interval, 24 * 60))
                    job.next_run_at = now + timedelta(minutes=interval)
                elif job.schedule_type == "once":
                    job.next_run_at = None

                await db.commit()
                execute_agent_job_task.delay(str(job.id), str(job.user_id))

            logger.info(f"Queued {len(due_jobs)} scheduled agent jobs")

    asyncio.run(_process_scheduled())


@celery_app.task
def resume_paused_agent_jobs():
    """
    Check and potentially resume paused agent jobs.

    For jobs that were paused due to rate limits or temporary issues,
    this task checks if they can be resumed.
    """
    logger.info("Checking paused agent jobs for resumption")

    async def _check_paused():
        session_factory = create_celery_session()
        async with session_factory() as db:
            # Find paused jobs that haven't been touched in a while
            cutoff = datetime.utcnow() - timedelta(minutes=5)

            result = await db.execute(
                select(AgentJob).where(
                    and_(
                        AgentJob.status == AgentJobStatus.PAUSED.value,
                        or_(
                            AgentJob.last_activity_at < cutoff,
                            AgentJob.last_activity_at.is_(None),
                        ),
                    )
                )
            )
            paused_jobs = result.scalars().all()

            resumed_count = 0
            for job in paused_jobs:
                # Paused jobs are not eligible for can_continue(); check resource limits directly.
                is_limited, reason = job.is_resource_limited()
                if is_limited:
                    logger.info(f"Skipping paused agent job {job.id} because of {reason}")
                    continue

                logger.info(f"Resuming paused agent job {job.id}")
                job.status = AgentJobStatus.PENDING.value
                job.error = None
                job.celery_task_id = None
                job.last_activity_at = datetime.utcnow()
                execute_agent_job_task.delay(str(job.id), str(job.user_id))
                resumed_count += 1

            await db.commit()
            logger.info(f"Resumed {resumed_count} paused agent jobs")

    asyncio.run(_check_paused())


@celery_app.task
def cleanup_old_agent_jobs(days: int = 30):
    """
    Cleanup task to remove old completed/failed agent jobs.

    Removes jobs older than the specified number of days.
    Scheduled to run periodically via Celery Beat.

    Args:
        days: Number of days to keep jobs
    """
    logger.info(f"Starting cleanup of agent jobs older than {days} days")

    async def _cleanup():
        session_factory = create_celery_session()
        async with session_factory() as db:
            from app.models.agent_job import AgentJobCheckpoint

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Find old completed/failed/cancelled jobs
            result = await db.execute(
                select(AgentJob).where(
                    and_(
                        AgentJob.created_at < cutoff_date,
                        AgentJob.status.in_([
                            AgentJobStatus.COMPLETED.value,
                            AgentJobStatus.FAILED.value,
                            AgentJobStatus.CANCELLED.value,
                        ])
                    )
                )
            )
            old_jobs = result.scalars().all()

            deleted_count = 0
            for job in old_jobs:
                try:
                    # Delete checkpoints before removing the job row.
                    await db.execute(
                        delete(AgentJobCheckpoint).where(AgentJobCheckpoint.job_id == job.id)
                    )

                    # Delete job
                    await db.delete(job)
                    deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cleanup agent job {job.id}: {e}")

            await db.commit()
            logger.info(f"Cleaned up {deleted_count} old agent jobs")

    asyncio.run(_cleanup())


@celery_app.task
def check_stalled_agent_jobs(timeout_minutes: int = 30):
    """
    Check for stalled agent jobs that haven't made progress.

    Jobs that have been running without activity for too long
    are marked as failed.

    Args:
        timeout_minutes: Minutes without activity before marking as stalled
    """
    logger.info(f"Checking for stalled agent jobs (timeout: {timeout_minutes} min)")

    async def _check_stalled():
        session_factory = create_celery_session()
        async with session_factory() as db:
            cutoff = datetime.utcnow() - timedelta(minutes=timeout_minutes)

            # Find running jobs with no recent activity
            result = await db.execute(
                select(AgentJob).where(
                    and_(
                        AgentJob.status == AgentJobStatus.RUNNING.value,
                        or_(
                            AgentJob.last_activity_at < cutoff,
                            AgentJob.last_activity_at.is_(None),
                        ),
                    )
                )
            )
            stalled_jobs = result.scalars().all()

            for job in stalled_jobs:
                logger.warning(f"Marking stalled agent job {job.id} as failed")
                job.status = AgentJobStatus.FAILED.value
                job.error = f"Job stalled - no activity for {timeout_minutes} minutes"
                job.completed_at = datetime.utcnow()
                job.celery_task_id = None
                _record_scheduler_outcome(
                    job,
                    outcome=AgentJobStatus.FAILED.value,
                    happened_at=job.completed_at,
                    queue_reason="stalled_run",
                )
                await sync_follow_up_outcome_for_job(db, job)

                await _publish_job_progress(
                    job_id=str(job.id),
                    progress=job.progress,
                    phase="stalled",
                    status="failed",
                    error=job.error,
                )

            await db.commit()
            logger.info(f"Marked {len(stalled_jobs)} stalled jobs as failed")

    asyncio.run(_check_stalled())


@celery_app.task
def generate_job_summary(job_id: str):
    """
    Generate a summary report for a completed agent job.

    Creates a human-readable summary of what the job accomplished.

    Args:
        job_id: UUID of the completed AgentJob
    """
    logger.info(f"Generating summary for agent job {job_id}")

    async def _generate_summary():
        from app.services.llm_service import LLMService, UserLLMSettings
        from app.models.memory import UserPreferences

        job_uuid = UUID(job_id)
        session_factory = create_celery_session()

        async with session_factory() as db:
            result = await db.execute(
                select(AgentJob).where(AgentJob.id == job_uuid)
            )
            job = result.scalar_one_or_none()

            if not job or job.status != AgentJobStatus.COMPLETED.value:
                logger.info(f"Job {job_id} not found or not completed")
                return

            # Generate summary using LLM
            llm_service = LLMService()
            # Best-effort: apply per-user LLM settings (provider/model/custom URL, etc.)
            user_settings = None
            try:
                prefs_res = await db.execute(select(UserPreferences).where(UserPreferences.user_id == job.user_id))
                prefs = prefs_res.scalar_one_or_none()
                user_settings = UserLLMSettings.from_preferences(prefs) if prefs else None
            except Exception:
                user_settings = None

            summary_prompt = f"""Generate a concise summary of this completed autonomous agent job:

Job Name: {job.name}
Job Type: {job.job_type}
Goal: {job.goal}

Results:
- Iterations: {job.iteration}
- Tool calls: {job.tool_calls_used}
- LLM calls: {job.llm_calls_used}
- Progress: {job.progress}%

Findings: {json.dumps(job.results.get('findings', [])[:10] if job.results else [], default=str)}

Provide a 2-3 sentence summary of what was accomplished."""

            try:
                summary = await llm_service.generate_response(
                    system_prompt="You are a helpful assistant that summarizes research and analysis results.",
                    user_message=summary_prompt,
                    user_settings=user_settings,
                    task_type="summarization",
                    user_id=job.user_id,
                    db=db,
                )

                # Store summary in results
                if job.results is None:
                    job.results = {}
                job.results["summary"] = summary

                await db.commit()
                logger.info(f"Generated summary for job {job_id}")

            except Exception as e:
                logger.error(f"Failed to generate summary for job {job_id}: {e}")

    asyncio.run(_generate_summary())
