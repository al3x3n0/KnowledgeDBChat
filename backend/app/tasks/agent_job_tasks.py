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
from uuid import UUID, uuid4

from celery import current_task
from loguru import logger
from sqlalchemy import and_, delete, or_, select
from sqlalchemy.orm import selectinload

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_execution_lease_service import (
    ExecutionLeaseLostError,
    agent_execution_lease_service,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.research_inbox_follow_up_service import sync_follow_up_outcome_for_job


def _scheduler_state(job: AgentJob) -> dict:
    results = job.results if isinstance(job.results, dict) else {}
    execution = (
        results.get("execution_strategy")
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
    state = (
        execution.get("scheduler_state")
        if isinstance(execution.get("scheduler_state"), dict)
        else {}
    )
    return {
        **state,
        "last_run_status": str(state.get("last_run_status") or "").strip() or None,
        "failure_streak": max(0, int(state.get("failure_streak", 0) or 0)),
    }


def _write_scheduler_state(job: AgentJob, state: dict) -> dict:
    results = dict(job.results) if isinstance(job.results, dict) else {}
    execution = (
        dict(results.get("execution_strategy"))
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
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


def _record_scheduler_outcome(
    job: AgentJob,
    *,
    outcome: str,
    happened_at: datetime,
    queue_reason: str | None = None,
) -> None:
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
                interval_minutes = int(
                    ((job.config or {}).get("interval_minutes") or 30)
                )
            except Exception:
                interval_minutes = 30
            interval_minutes = max(1, min(interval_minutes, 24 * 60))
            backoff_seconds = min(
                interval_minutes * 60 * (2 ** max(0, streak - 1)), 6 * 60 * 60
            )
            state["backoff_seconds"] = int(backoff_seconds)
            state["backoff_until"] = (
                happened_at + timedelta(seconds=backoff_seconds)
            ).isoformat()
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
        if (
            isinstance(scope_observability_runtime, dict)
            and scope_observability_runtime
        ):
            message["scope_observability_runtime"] = scope_observability_runtime

        await redis_client.publish(channel, json.dumps(message))
        await redis_client.close()
    except Exception as e:
        logger.warning(f"Failed to publish progress for agent job {job_id}: {e}")


async def _execute_agent_job_async(
    job_id: str,
    user_id: str,
    *,
    lease_owner_id: Optional[str] = None,
):
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

        owner_id = str(
            lease_owner_id
            or getattr(getattr(current_task, "request", None), "id", None)
            or f"worker:{uuid4()}"
        )
        lease_ttl = (
            (job.config or {}).get("execution_lease_ttl_seconds")
            if isinstance(job.config, dict)
            else None
        )
        lease = await agent_execution_lease_service.acquire(
            db=db,
            job_id=job_uuid,
            owner_id=owner_id,
            ttl_seconds=lease_ttl,
        )
        if lease is None:
            logger.info(
                f"Agent job {job_id} already has an active execution lease; "
                "duplicate delivery skipped"
            )
            return {
                "status": "lease_conflict",
                "job_id": job_id,
            }
        await db.refresh(job)

        # Store celery task ID
        job.celery_task_id = owner_id
        await db.commit()

        heartbeat_stop = asyncio.Event()
        lease_lost = asyncio.Event()
        heartbeat_interval = max(
            10,
            agent_execution_lease_service.normalize_ttl(lease_ttl) // 3,
        )

        async def _heartbeat_execution_lease() -> None:
            while not heartbeat_stop.is_set():
                try:
                    await asyncio.wait_for(
                        heartbeat_stop.wait(),
                        timeout=heartbeat_interval,
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                heartbeat_session_factory = create_celery_session()
                async with heartbeat_session_factory() as heartbeat_db:
                    renewed = await agent_execution_lease_service.renew(
                        db=heartbeat_db,
                        lease=lease,
                        ttl_seconds=lease_ttl,
                    )
                if renewed is None:
                    lease_lost.set()
                    logger.error(
                        f"Execution lease heartbeat lost for job {job_id} "
                        f"at fence {lease.fence}"
                    )
                    return

        heartbeat_task = asyncio.create_task(_heartbeat_execution_lease())

        await _publish_job_progress(
            job_id=job_id,
            progress=0,
            phase="starting",
            status="running",
        )

        try:
            # Initialize executor
            executor = AutonomousAgentExecutor()
            executor.execution_lease = lease

            # Progress callback that publishes to Redis
            async def progress_callback(progress_data: dict):
                if lease_lost.is_set():
                    raise ExecutionLeaseLostError(
                        f"Execution lease heartbeat lost for job {job_id}"
                    )
                await agent_execution_lease_service.assert_owned(
                    db=db,
                    lease=lease,
                )
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
                        if isinstance(
                            progress_data.get("execution_graph_runtime"), dict
                        )
                        else None
                    ),
                    scope_observability_runtime=(
                        progress_data.get("scope_observability_runtime")
                        if isinstance(
                            progress_data.get("scope_observability_runtime"), dict
                        )
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
            await agent_execution_lease_service.assert_owned(db=db, lease=lease)
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

        except ExecutionLeaseLostError as e:
            await db.rollback()
            logger.error(f"Abandoning stale execution for job {job_id}: {e}")
            return {
                "status": "lease_lost",
                "job_id": job_id,
                "fence": lease.fence,
            }
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
        finally:
            heartbeat_stop.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            try:
                await agent_execution_lease_service.release(db=db, lease=lease)
            except Exception as release_error:
                logger.warning(
                    f"Failed releasing execution lease for job {job_id}: "
                    f"{release_error}"
                )


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

    task_owner_id = str(
        getattr(getattr(self, "request", None), "id", None)
        or getattr(getattr(current_task, "request", None), "id", None)
        or f"worker:{uuid4()}"
    )
    execution_coro = _execute_agent_job_async(
        job_id,
        user_id,
        lease_owner_id=task_owner_id,
    )
    try:
        asyncio.run(execution_coro)

    except Exception as e:
        execution_coro.close()
        logger.exception(f"Agent job task failed for {job_id}")
        error_text = str(e)

        async def _mark_failed():
            job_uuid = UUID(job_id)
            session_factory = create_celery_session()
            async with session_factory() as db:
                result = await db.execute(
                    select(AgentJob).where(AgentJob.id == job_uuid)
                )
                job = result.scalar_one_or_none()
                lease_expired = False
                if job and job.execution_lease_expires_at is not None:
                    lease_deadline = job.execution_lease_expires_at
                    lease_now = (
                        datetime.now(lease_deadline.tzinfo)
                        if lease_deadline.tzinfo is not None
                        else datetime.utcnow()
                    )
                    lease_expired = lease_deadline <= lease_now
                lease_allows_failure_write = bool(
                    job
                    and (
                        not job.execution_lease_owner
                        or job.execution_lease_owner == task_owner_id
                        or lease_expired
                    )
                )
                if (
                    job
                    and lease_allows_failure_write
                    and job.status
                    not in (
                        AgentJobStatus.COMPLETED.value,
                        AgentJobStatus.FAILED.value,
                        AgentJobStatus.CANCELLED.value,
                    )
                ):
                    job.status = AgentJobStatus.FAILED.value
                    job.error = f"Task error: {error_text}"
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
                select(AgentJob)
                .where(
                    and_(
                        AgentJob.schedule_type.isnot(None),
                        AgentJob.next_run_at <= now,
                        AgentJob.status.in_(
                            [
                                AgentJobStatus.PENDING.value,
                                AgentJobStatus.COMPLETED.value,
                                AgentJobStatus.FAILED.value,
                            ]
                        ),
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
                        logger.error(
                            f"Failed to calculate next run for job {job.id}: {e}"
                        )
                        job.next_run_at = None
                elif job.schedule_type == "continuous":
                    # Simple interval scheduling (minutes) stored in job.config.interval_minutes.
                    try:
                        interval = int(
                            ((job.config or {}).get("interval_minutes") or 30)
                        )
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
                    logger.info(
                        f"Skipping paused agent job {job.id} because of {reason}"
                    )
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
                        AgentJob.status.in_(
                            [
                                AgentJobStatus.COMPLETED.value,
                                AgentJobStatus.FAILED.value,
                                AgentJobStatus.CANCELLED.value,
                            ]
                        ),
                    )
                )
            )
            old_jobs = result.scalars().all()

            deleted_count = 0
            for job in old_jobs:
                try:
                    # Delete checkpoints before removing the job row.
                    await db.execute(
                        delete(AgentJobCheckpoint).where(
                            AgentJobCheckpoint.job_id == job.id
                        )
                    )

                    # Delete job
                    await db.delete(job)
                    deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cleanup agent job {job.id}: {e}")

            await db.commit()
            logger.info(f"Cleaned up {deleted_count} old agent jobs")

    asyncio.run(_cleanup())


MAX_ORPHAN_RECOVERIES = 3
ORPHAN_RECOVERY_PHASE = "orphan_recovered"


def count_orphan_recoveries(job: AgentJob) -> int:
    """How many times this job has already been recovered from a lost worker."""
    entries = job.execution_log if isinstance(job.execution_log, list) else []
    return sum(
        1
        for entry in entries
        if isinstance(entry, dict) and entry.get("phase") == ORPHAN_RECOVERY_PHASE
    )


def is_orphaned(job: AgentJob, now: datetime) -> bool:
    """True when no worker holds this job any more.

    The lease is what distinguishes a job whose worker died from one that is
    genuinely wedged: a live lease is heartbeated by the worker running it, so
    if it has expired or was never taken, nobody is executing this job and it
    is safe to queue again. Without that distinction the only options are to
    fail every quiet job -- losing hours of work a checkpoint could restore --
    or to requeue every quiet job and risk two workers running one job.
    """
    expires_at = job.execution_lease_expires_at
    if expires_at is None:
        return True
    if expires_at.tzinfo is not None:
        expires_at = expires_at.replace(tzinfo=None)
    return expires_at < now


@celery_app.task
def check_stalled_agent_jobs(timeout_minutes: int = 30):
    """
    Check for stalled agent jobs that haven't made progress.

    A job whose worker died is queued again, resuming from its last
    checkpoint; a job that is quiet while its lease is still being
    heartbeated is genuinely stuck and is failed.

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

            now = datetime.utcnow()
            recovered_count = 0
            failed_count = 0
            for job in stalled_jobs:
                recoveries = count_orphan_recoveries(job)
                limited, limit_reason = job.is_resource_limited()
                if (
                    is_orphaned(job, now)
                    and not limited
                    and recoveries < MAX_ORPHAN_RECOVERIES
                ):
                    logger.warning(
                        f"Agent job {job.id} lost its worker; queueing again "
                        f"(recovery {recoveries + 1}/{MAX_ORPHAN_RECOVERIES})"
                    )
                    job.add_log_entry(
                        {
                            "phase": ORPHAN_RECOVERY_PHASE,
                            "reason": "execution lease expired with no worker",
                            "recovery_attempt": recoveries + 1,
                        }
                    )
                    job.status = AgentJobStatus.PENDING.value
                    job.celery_task_id = None
                    job.last_activity_at = now
                    # A job queued to run again has not completed and carries no
                    # error; leaving either set describes a finished run.
                    job.completed_at = None
                    job.error = None
                    execute_agent_job_task.delay(str(job.id), str(job.user_id))
                    recovered_count += 1
                    await _publish_job_progress(
                        job_id=str(job.id),
                        progress=job.progress,
                        phase=ORPHAN_RECOVERY_PHASE,
                        status="pending",
                    )
                    continue

                logger.warning(f"Marking stalled agent job {job.id} as failed")
                job.status = AgentJobStatus.FAILED.value
                if limited:
                    job.error = f"Job stalled and cannot continue: {limit_reason}"
                elif recoveries >= MAX_ORPHAN_RECOVERIES:
                    job.error = (
                        f"Job lost its worker {recoveries} times; not retried again"
                    )
                else:
                    job.error = (
                        f"Job stalled - no activity for {timeout_minutes} minutes "
                        "while its execution lease was still held"
                    )
                job.completed_at = now
                job.celery_task_id = None
                failed_count += 1
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
            logger.info(
                f"Stalled-job sweep: {recovered_count} requeued after losing a "
                f"worker, {failed_count} failed"
            )

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
        from app.models.memory import UserPreferences
        from app.services.llm_service import LLMService, UserLLMSettings

        job_uuid = UUID(job_id)
        session_factory = create_celery_session()

        async with session_factory() as db:
            result = await db.execute(select(AgentJob).where(AgentJob.id == job_uuid))
            job = result.scalar_one_or_none()

            if not job or job.status != AgentJobStatus.COMPLETED.value:
                logger.info(f"Job {job_id} not found or not completed")
                return

            # Generate summary using LLM
            llm_service = LLMService()
            # Best-effort: apply per-user LLM settings (provider/model/custom URL, etc.)
            user_settings = None
            try:
                prefs_res = await db.execute(
                    select(UserPreferences).where(
                        UserPreferences.user_id == job.user_id
                    )
                )
                prefs = prefs_res.scalar_one_or_none()
                user_settings = (
                    UserLLMSettings.from_preferences(prefs) if prefs else None
                )
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


@celery_app.task
def advance_research_campaigns(limit: int = 25):
    """Move every active campaign forward one step.

    A campaign holds a line of enquiry across many jobs and keeps all its state
    in the database, so running one is a matter of asking it to take a step
    often enough. This is that asking. Nothing is held between calls, which is
    why a restart costs only the time the machine was off.

    Each step launches at most one job per campaign, so this tick's cost is
    bounded by the number of active campaigns rather than by the size of their
    backlogs.
    """

    async def _advance():
        session_factory = create_celery_session()
        async with session_factory() as db:
            from app.services import research_campaign_service

            steps = await research_campaign_service.advance_all(db, limit=limit)
            await db.commit()

            launched = [s for s in steps if s.get("action") == "launched"]
            for step in launched:
                job_id = step.get("launched_job")
                campaign_id = step.get("campaign")
                if not job_id:
                    continue
                # The campaign records the job; a worker still has to run it.
                job = await db.get(AgentJob, UUID(str(job_id)))
                if job is None:
                    continue
                execute_agent_job_task.delay(str(job.id), str(job.user_id))
                logger.info(f"Campaign {campaign_id} launched job {job_id}")

            finished = [
                s for s in steps if s.get("action") in ("completed", "exhausted")
            ]
            for step in finished:
                logger.info(
                    f"Campaign {step.get('campaign')} {step.get('action')} "
                    f"after {step.get('jobs_launched')} jobs"
                )
            if steps:
                logger.info(
                    f"Advanced {len(steps)} campaign(s): "
                    f"{len(launched)} launched, {len(finished)} finished"
                )
            return {"advanced": len(steps), "launched": len(launched)}

    return asyncio.run(_advance())
