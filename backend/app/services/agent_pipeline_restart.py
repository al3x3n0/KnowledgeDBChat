"""Restart a pipeline run from one of its stages.

A pipeline that stops four stages in has usually stopped for a reason that
belongs to one stage. Re-launching the whole thing re-ingests the paper,
re-writes the specification and re-runs the implementation to get back to the
place it already reached -- paying for four stages to retry the fifth, and
producing a *different* run, so the evidence the earlier stages established is
not the evidence the retry builds on.

The mechanism is deliberately not new. A pipeline stage already exists as a
chained child job, and `create_chained_job` already builds one from its
parent's `chain_config.child_jobs` entry, carrying the parent's results into
`config.inherited_data`. Restarting from stage N is therefore re-firing the
chain from the completed stage N-1, and the inheritance that makes stages
connect is the same code path that connected them the first time. Reimplementing
it here would be a second definition of what a stage inherits, and the two would
drift.

What this file adds on top is the refusals, because a restart is only worth
anything if the ground under it is real:

* the predecessor must have COMPLETED and MET ITS CONTRACT. Restarting
  `measure` on top of an `implement` stage that never verified anything is how
  a run reaches a benchmark of code nobody checked -- observed exactly so, and
  the chain gate refuses it for the same reason.
* the stage being restarted must not be running. Two runners on one stage is
  the failure the execution lease exists to prevent; there is no reason to
  create it deliberately.
* a correction, if given, is attached where the run will actually read it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus


class PipelineRestartError(Exception):
    """A restart that must not happen, with the reason a caller can act on."""

    def __init__(self, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class StageJob:
    """One stage of one run."""

    stage_id: str
    job: AgentJob

    @property
    def completed(self) -> bool:
        return str(self.job.status or "") == AgentJobStatus.COMPLETED.value

    @property
    def contract_met(self) -> bool:
        """Whether the stage produced what it promised.

        A stage can complete without meeting its contract -- it ran out of
        iterations, or gave up. `goal_contract_satisfied()` is the same
        question the chain gate asks before starting the next stage, asked
        here for the same reason.
        """
        return bool(self.job.goal_contract_satisfied())


def stage_of(job: AgentJob) -> str:
    """The stage id a job was created for, or "" if it is not a pipeline stage."""
    config = job.config if isinstance(job.config, dict) else {}
    return str(config.get("pipeline_stage") or "").strip()


async def load_run(root_job_id: uuid.UUID, db: AsyncSession) -> List[StageJob]:
    """Every stage job of one run, oldest first.

    Matched on `root_job_id` OR the id itself: the head of a chain is its own
    run but does not point at itself.
    """
    rows = await db.execute(
        select(AgentJob)
        .where(or_(AgentJob.id == root_job_id, AgentJob.root_job_id == root_job_id))
        .order_by(AgentJob.created_at)
    )
    stages: List[StageJob] = []
    for job in rows.scalars().all():
        stage_id = stage_of(job)
        if stage_id:
            stages.append(StageJob(stage_id=stage_id, job=job))
    return stages


def latest_per_stage(stages: List[StageJob]) -> Dict[str, StageJob]:
    """The most recent job for each stage.

    A run restarted before has more than one job for a stage, and the one that
    matters is the last -- restarting again must build on the newest attempt,
    not resurrect the one that was already superseded.
    """
    latest: Dict[str, StageJob] = {}
    for stage in stages:
        latest[stage.stage_id] = stage
    return latest


def _child_config_for(parent: AgentJob, stage_id: str) -> Optional[Dict[str, Any]]:
    """The parent's own definition of this child stage.

    Read from the parent rather than rebuilt from the pipeline spec, because
    the spec is not stored with the run: the bound chain IS the record of what
    this run was asked to do, and rebuilding from a spec that may have been
    edited since would restart a different pipeline under the same run id.
    """
    chain_config = parent.chain_config if isinstance(parent.chain_config, dict) else {}
    for child in chain_config.get("child_jobs") or []:
        if not isinstance(child, dict):
            continue
        config = child.get("config") if isinstance(child.get("config"), dict) else {}
        if str(config.get("pipeline_stage") or "").strip() == stage_id:
            return child
    return None


def _attach_correction(child_config: Dict[str, Any], note: str) -> None:
    """Put the operator's correction where the restarted run reads it.

    `operator_clues` is the same channel a resumed job uses, rendered into the
    stable thinking prompt beside the goal. A restart without this is a stage
    repeating what it already did: the reason it stopped is usually something
    only a person knows.
    """
    config = child_config.setdefault("config", {})
    clues = config.get("operator_clues")
    if not isinstance(clues, list):
        clues = []
    clues.append({"note": note[:2000], "iteration": 0})
    config["operator_clues"] = clues[-8:]


async def restart_from_stage(
    *,
    root_job_id: uuid.UUID,
    stage_id: str,
    executor: Any,
    db: AsyncSession,
    note: str = "",
) -> AgentJob:
    """Run `stage_id` again on the evidence the stages before it established."""
    stages = await load_run(root_job_id, db)
    if not stages:
        raise PipelineRestartError(
            f"No pipeline run found for job {root_job_id}", status_code=404
        )

    by_stage = latest_per_stage(stages)
    target = by_stage.get(stage_id)
    if target is None:
        raise PipelineRestartError(
            f"This run has no stage {stage_id!r}. Its stages are: "
            + ", ".join(sorted(by_stage))
        )

    if str(target.job.status or "") == AgentJobStatus.RUNNING.value:
        raise PipelineRestartError(
            f"Stage {stage_id!r} is still running. Cancel it first; two "
            "runners on one stage is what the execution lease exists to stop."
        )

    parent_id = target.job.parent_job_id
    if parent_id is None:
        raise PipelineRestartError(
            f"Stage {stage_id!r} is the head of this pipeline, so there is no "
            "earlier evidence to restart it on. Launch the pipeline again."
        )

    parent = next((s for s in stages if s.job.id == parent_id), None)
    if parent is None:
        raise PipelineRestartError(
            f"Stage {stage_id!r} has no parent stage in this run; it cannot be "
            "restarted on evidence that is not there."
        )

    if not parent.completed:
        raise PipelineRestartError(
            f"Stage {parent.stage_id!r} is {parent.job.status}, not completed. "
            f"Restarting {stage_id!r} on it would build on work that has not "
            "finished."
        )
    if not parent.contract_met:
        raise PipelineRestartError(
            f"Stage {parent.stage_id!r} completed without meeting its "
            f"contract, so the evidence {stage_id!r} assumes was never "
            "produced. Restart from that stage instead."
        )

    child_config = _child_config_for(parent.job, stage_id)
    if child_config is None:
        raise PipelineRestartError(
            f"Stage {parent.stage_id!r} does not define {stage_id!r} as its "
            "next stage in this run."
        )

    child_config = {**child_config, "config": dict(child_config.get("config") or {})}
    if note.strip():
        _attach_correction(child_config, note.strip())

    # The parent fired its chain once already; without this the orchestration
    # would treat the stage as started and do nothing, silently.
    parent.job.chain_triggered = False

    child = await executor.chain_orchestration_service.create_chained_job(
        executor, parent.job, child_config, db
    )
    if child is None:
        raise PipelineRestartError(
            f"Could not create a new job for stage {stage_id!r}.", status_code=500
        )

    # Commit, then dispatch, in that order and both of them. `create_chained_job`
    # only adds the row -- the chain trigger commits and queues afterwards, and
    # a restart that skipped either would leave a job that never runs. That is
    # exactly the bug this pipeline's launch endpoint had: a created job, a
    # success reported, and nothing queued.
    await db.commit()
    from app.tasks.agent_job_tasks import execute_agent_job_task

    execute_agent_job_task.delay(str(child.id), str(child.user_id))

    logger.info(
        f"Restarted pipeline run {root_job_id} from stage {stage_id} "
        f"as job {child.id} (parent {parent.job.id})"
    )
    return child
