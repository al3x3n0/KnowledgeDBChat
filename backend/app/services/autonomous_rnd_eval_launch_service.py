"""Fan an evaluation suite out into real agent jobs, then grade them as trials.

Grading persisted trajectories already works, but it requires an operator to
launch every trial by hand and then bind job ids to tasks. This service closes
that loop: it creates ``trials`` jobs per suite task, tracks them as one launch,
and grades the whole set once every trial reaches a terminal state.

Fan-out is unattended agent execution, so it is opt-in
(``AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED``) and hard-capped on total trial jobs
(``AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS``).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.autonomous_rnd_eval_launch import (
    EVAL_LAUNCH_STATUS_COMPLETED,
    EVAL_LAUNCH_STATUS_FAILED,
    EVAL_LAUNCH_STATUS_RUNNING,
    AutonomousRndEvalLaunch,
)
from app.models.autonomous_rnd_eval_run import AutonomousRndEvalRun
from app.models.experiment import ExperimentRun
from app.services.agent_job_creation_service import (
    AgentJobSpec,
    agent_job_creation_service,
)
from app.services.autonomous_rnd_eval_run_service import autonomous_rnd_eval_run_service
from app.services.autonomous_rnd_eval_service import (
    AutonomousRnDEvalSuite,
    autonomous_rnd_eval_harness,
)
from app.services.autonomous_rnd_trajectory_service import (
    autonomous_rnd_trajectory_adapter,
)

EVAL_RUN_SOURCE_LAUNCH = "launch"

TERMINAL_JOB_STATUSES = {
    AgentJobStatus.COMPLETED.value,
    AgentJobStatus.FAILED.value,
    AgentJobStatus.CANCELLED.value,
}


class EvalLaunchError(RuntimeError):
    """Raised when a suite cannot be launched or finalized."""

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code


class AutonomousRnDEvalLaunchService:
    """Create trial jobs for a suite and grade them once they settle."""

    async def launch(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        suite: AutonomousRnDEvalSuite,
        trials_override: Optional[int] = None,
        label: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AutonomousRndEvalLaunch, List[AgentJob]]:
        if not settings.AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED:
            raise EvalLaunchError(
                "Evaluation launches are disabled; set "
                "AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED to enable them",
                status_code=403,
            )

        planned = [
            (task, self._trial_count(task, trials_override)) for task in suite.tasks
        ]
        total_jobs = sum(count for _, count in planned)
        cap = max(1, int(settings.AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS))
        if total_jobs > cap:
            raise EvalLaunchError(
                f"Launch would create {total_jobs} trial jobs, "
                f"above the AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS cap of {cap}",
                status_code=400,
            )

        launch_id = uuid4()
        jobs: List[AgentJob] = []
        task_bindings: Dict[str, List[str]] = {}
        for task, trial_count in planned:
            job_ids: List[str] = []
            for trial_index in range(trial_count):
                job = await agent_job_creation_service.create(
                    spec=self._trial_spec(
                        suite=suite,
                        task=task,
                        trial_index=trial_index,
                        launch_id=launch_id,
                        config_overrides=config_overrides,
                    ),
                    user_id=user_id,
                    db=db,
                )
                jobs.append(job)
                job_ids.append(str(job.id))
            task_bindings[task.id] = job_ids

        launch = AutonomousRndEvalLaunch(
            id=launch_id,
            user_id=user_id,
            suite_id=suite.id,
            suite_name=suite.name,
            suite_version=suite.version,
            label=(label.strip()[:200] or None) if label else None,
            status=EVAL_LAUNCH_STATUS_RUNNING,
            trials_per_task=max(count for _, count in planned),
            job_count=total_jobs,
            task_bindings=task_bindings,
        )
        db.add(launch)
        await db.flush()
        return launch, jobs

    def _trial_count(self, task, trials_override: Optional[int]) -> int:
        if trials_override is None:
            return task.trials
        return max(1, min(int(trials_override), 100))

    def _trial_spec(
        self,
        *,
        suite: AutonomousRnDEvalSuite,
        task,
        trial_index: int,
        launch_id: UUID,
        config_overrides: Optional[Dict[str, Any]],
    ) -> AgentJobSpec:
        config: Dict[str, Any] = dict(config_overrides or {})
        # The eval binding lives in config so a trial job is identifiable
        # without a second table, and so finalization can rebuild bindings.
        config["autonomous_rnd_eval"] = {
            "launch_id": str(launch_id),
            "suite_id": suite.id,
            "suite_version": suite.version,
            "task_id": task.id,
            "trial_index": trial_index,
            "seed": suite.seed + trial_index,
        }
        return AgentJobSpec(
            name=f"{suite.name} · {task.name} · trial {trial_index + 1}"[:200],
            description=f"Evaluation trial for suite {suite.id}, task {task.id}",
            job_type="analysis",
            goal=task.prompt or task.name,
            config=config,
            max_iterations=int(settings.AUTONOMOUS_RND_EVAL_TRIAL_MAX_ITERATIONS),
            max_runtime_minutes=int(
                settings.AUTONOMOUS_RND_EVAL_TRIAL_MAX_RUNTIME_MINUTES
            ),
        )

    async def _load_jobs(
        self, db: AsyncSession, *, launch: AutonomousRndEvalLaunch
    ) -> Dict[UUID, AgentJob]:
        job_ids: List[UUID] = []
        for raw_ids in (launch.task_bindings or {}).values():
            for raw in raw_ids:
                try:
                    job_ids.append(UUID(str(raw)))
                except (TypeError, ValueError):
                    continue
        if not job_ids:
            return {}
        jobs = (
            (
                await db.execute(
                    select(AgentJob).where(
                        AgentJob.id.in_(job_ids),
                        AgentJob.user_id == launch.user_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        return {job.id: job for job in jobs}

    async def progress(
        self, db: AsyncSession, *, launch: AutonomousRndEvalLaunch
    ) -> Dict[str, Any]:
        jobs_by_id = await self._load_jobs(db, launch=launch)
        statuses = [str(job.status or "") for job in jobs_by_id.values()]
        terminal = [status for status in statuses if status in TERMINAL_JOB_STATUSES]
        missing = int(launch.job_count or 0) - len(jobs_by_id)
        return {
            "job_count": int(launch.job_count or 0),
            # A deleted trial job can never reach a terminal state, so it is
            # counted as settled; otherwise finalization would wait forever.
            "settled_count": len(terminal) + max(0, missing),
            "missing_count": max(0, missing),
            "failed_count": sum(
                1 for status in statuses if status == AgentJobStatus.FAILED.value
            ),
            "is_ready": len(terminal) + max(0, missing) >= int(launch.job_count or 0),
        }

    async def finalize(
        self, db: AsyncSession, *, launch: AutonomousRndEvalLaunch
    ) -> Optional[AutonomousRndEvalRun]:
        """Grade a settled launch. Returns None while trials are still running."""
        if launch.status != EVAL_LAUNCH_STATUS_RUNNING:
            return (
                await db.get(AutonomousRndEvalRun, launch.run_id)
                if launch.run_id
                else None
            )

        progress = await self.progress(db, launch=launch)
        if not progress["is_ready"]:
            return None

        try:
            suite = autonomous_rnd_eval_harness.load_builtin_suite(launch.suite_id)
        except Exception as exc:  # noqa: BLE001 - surfaced on the launch record
            launch.status = EVAL_LAUNCH_STATUS_FAILED
            launch.error = f"Suite could not be loaded: {exc}"[:1000]
            launch.completed_at = datetime.now(timezone.utc)
            await db.flush()
            return None

        jobs_by_id = await self._load_jobs(db, launch=launch)
        runs_by_job: Dict[Any, List[ExperimentRun]] = defaultdict(list)
        if jobs_by_id:
            experiment_runs = (
                (
                    await db.execute(
                        select(ExperimentRun).where(
                            ExperimentRun.agent_job_id.in_(list(jobs_by_id)),
                            ExperimentRun.user_id == launch.user_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for run in experiment_runs:
                runs_by_job[run.agent_job_id].append(run)

        outcomes: Dict[str, List[Dict[str, Any]]] = {}
        for task_id, raw_ids in (launch.task_bindings or {}).items():
            task_outcomes: List[Dict[str, Any]] = []
            for raw in raw_ids:
                try:
                    job = jobs_by_id.get(UUID(str(raw)))
                except (TypeError, ValueError):
                    job = None
                if job is None:
                    # A missing trial is a failed trial, never a skipped one:
                    # dropping it would inflate pass_pow_k for the task.
                    task_outcomes.append(
                        {"status": "error", "error": "trial job is unavailable"}
                    )
                    continue
                task_outcomes.append(
                    autonomous_rnd_trajectory_adapter.build_outcome(
                        job,
                        experiment_runs=runs_by_job.get(job.id, []),
                    )
                )
            outcomes[task_id] = task_outcomes

        report = autonomous_rnd_eval_harness.grade_suite_outcomes(suite, outcomes)
        run = await autonomous_rnd_eval_run_service.record_run(
            db,
            user_id=launch.user_id,
            report=report,
            task_bindings={
                task_id: list(raw_ids)
                for task_id, raw_ids in (launch.task_bindings or {}).items()
            },
            source=EVAL_RUN_SOURCE_LAUNCH,
            label=launch.label,
        )
        launch.run_id = run.id
        launch.status = EVAL_LAUNCH_STATUS_COMPLETED
        launch.completed_at = datetime.now(timezone.utc)
        await db.flush()
        return run

    async def get_launch(
        self, db: AsyncSession, *, user_id: UUID, launch_id: UUID
    ) -> Optional[AutonomousRndEvalLaunch]:
        return (
            await db.execute(
                select(AutonomousRndEvalLaunch).where(
                    AutonomousRndEvalLaunch.id == launch_id,
                    AutonomousRndEvalLaunch.user_id == user_id,
                )
            )
        ).scalar_one_or_none()

    async def list_launches(
        self,
        db: AsyncSession,
        *,
        user_id: UUID,
        suite_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[AutonomousRndEvalLaunch]:
        bounded = max(1, min(int(limit or 20), 100))
        query = select(AutonomousRndEvalLaunch).where(
            AutonomousRndEvalLaunch.user_id == user_id
        )
        if suite_id:
            query = query.where(AutonomousRndEvalLaunch.suite_id == suite_id.strip())
        query = query.order_by(AutonomousRndEvalLaunch.created_at.desc()).limit(bounded)
        return list((await db.execute(query)).scalars().all())


autonomous_rnd_eval_launch_service = AutonomousRnDEvalLaunchService()
