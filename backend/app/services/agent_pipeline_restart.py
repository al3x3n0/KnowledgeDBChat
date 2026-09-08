"""Change a pipeline run in flight: restart a stage, or insert a new one.

Two operations, one idea. A run that stops four stages in has usually stopped
for a reason belonging to one place, and re-launching the whole pipeline pays
for every earlier stage again to reach the same point -- in a DIFFERENT run, so
the evidence the retry builds on is not the evidence that was established.

Restarting runs an existing stage again. Inserting adds a stage that was not in
the plan, between one that succeeded and one that could not, which is the only
move that fixes a gap without redoing the work either side of it.

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
from sqlalchemy.orm.attributes import flag_modified

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


async def _rerun_the_head(
    head: AgentJob, note: str, executor: Any, db: AsyncSession
) -> AgentJob:
    """Run the pipeline's first stage again, with a correction.

    A clone rather than a reset of the original: the first attempt and what it
    established stay readable, which is the same reason a restarted stage does
    not overwrite the one it supersedes. The chain config is carried over
    verbatim, so the stages after it re-derive exactly as they did the first
    time.
    """
    config = dict(head.config or {})
    if note.strip():
        clues = config.get("operator_clues")
        if not isinstance(clues, list):
            clues = []
        clues.append({"note": note.strip()[:2000], "iteration": 0})
        config["operator_clues"] = clues[-8:]

    child = AgentJob(
        id=uuid.uuid4(),
        name=head.name,
        description=head.description,
        goal=head.goal,
        job_type=head.job_type,
        status=AgentJobStatus.PENDING.value,
        config=config,
        chain_config=head.chain_config,
        max_iterations=head.max_iterations,
        max_tool_calls=head.max_tool_calls,
        max_llm_calls=head.max_llm_calls,
        max_runtime_minutes=head.max_runtime_minutes,
        results={},
        execution_log=[],
        iteration=0,
        error_count=0,
    )
    child.user_id = head.user_id
    # The rerun belongs to the same run, so the ledger and the stage view still
    # see one pipeline rather than two.
    child.root_job_id = head.root_job_id or head.id
    db.add(child)
    await db.commit()

    from app.tasks.agent_job_tasks import execute_agent_job_task

    execute_agent_job_task.delay(str(child.id), str(child.user_id))
    logger.info(f"Re-ran pipeline head {head.id} as job {child.id}")
    return child


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
        # The head has no predecessor to re-fire from, but it is exactly the
        # stage a run most often needs redone: a `mine` stage whose profile was
        # too coarse needs the PROFILE again, and profile is the head.
        # Refusing here made `may_revisit: [<head>]` a promise the system could
        # not keep -- the spec validated it, the tool accepted the request, and
        # the finaliser then quietly logged that it could not act.
        #
        # So the head is re-run from its own definition. Nothing is inherited
        # because there is nothing upstream to inherit; the correction is the
        # only new input, and the chain re-derives every stage after it.
        return await _rerun_the_head(target.job, note, executor, db)

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


# --------------------------------------------------------------- insertion


def _stage_ids(stages: List[StageJob]) -> set:
    return {s.stage_id for s in stages}


async def insert_stage_after(
    *,
    root_job_id: uuid.UUID,
    after_stage: str,
    stage: Dict[str, Any],
    executor: Any,
    db: AsyncSession,
    note: str = "",
) -> AgentJob:
    """Add a stage between one that succeeded and whatever followed it.

    The new stage takes over its parent's children, so everything downstream
    re-derives beneath it rather than being rebuilt by hand: profile -> mine
    becomes profile -> new -> mine, and `mine` runs again from the new stage's
    evidence when it completes.

    The refusals are the same ones a restart makes, for the same reason -- a
    stage inserted on top of work that did not finish is built on nothing --
    plus two of its own: the new stage needs a goal and a contract, because a
    stage with no contract cannot fail and so cannot be the fix for anything;
    and its id must be new, because reusing an existing one makes the run's own
    history ambiguous about which stage a finding came from.
    """
    stages = await load_run(root_job_id, db)
    if not stages:
        raise PipelineRestartError(
            f"No pipeline run found for job {root_job_id}", status_code=404
        )

    by_stage = latest_per_stage(stages)
    parent = by_stage.get(after_stage)
    if parent is None:
        raise PipelineRestartError(
            f"This run has no stage {after_stage!r}. Its stages are: "
            + ", ".join(sorted(by_stage))
        )
    if not parent.completed:
        raise PipelineRestartError(
            f"Stage {after_stage!r} is {parent.job.status}, not completed. A "
            "stage inserted after it would be built on work that has not "
            "finished."
        )
    if not parent.contract_met:
        raise PipelineRestartError(
            f"Stage {after_stage!r} completed without meeting its contract, so "
            "there is no established evidence to insert a stage on top of."
        )

    new_id = str(stage.get("id") or "").strip()
    if not new_id:
        raise PipelineRestartError("The new stage needs an id.")
    if new_id in _stage_ids(stages):
        raise PipelineRestartError(
            f"This run already has a stage {new_id!r}. Give the new one a "
            "different id, or restart the existing one instead -- reusing the "
            "id would leave the run unable to say which stage a finding came "
            "from."
        )
    goal = str(stage.get("goal") or "").strip()
    if not goal:
        raise PipelineRestartError("The new stage needs a goal.")
    contract = stage.get("contract")
    if not isinstance(contract, dict) or not contract:
        raise PipelineRestartError(
            "The new stage needs a contract. A stage with nothing to satisfy "
            "cannot fail, so it cannot be the fix for a stage that did."
        )

    # Any stage that was going to run next is superseded by this one, and a
    # stage still running would race it.
    parent_chain = (
        parent.job.chain_config if isinstance(parent.job.chain_config, dict) else {}
    )
    displaced = [
        str((c.get("config") or {}).get("pipeline_stage") or "").strip()
        for c in (parent_chain.get("child_jobs") or [])
        if isinstance(c, dict)
    ]
    for stage_id in displaced:
        existing = by_stage.get(stage_id)
        if existing and str(existing.job.status or "") == AgentJobStatus.RUNNING.value:
            raise PipelineRestartError(
                f"Stage {stage_id!r} is running and would be superseded by this "
                "insertion. Cancel it first."
            )

    config = {
        **(stage.get("config") or {}),
        "pipeline": (parent.job.config or {}).get("pipeline"),
        "pipeline_stage": new_id,
        "goal_contract": contract,
        # What this insertion moved, recorded on the job itself so a caller
        # reads it rather than deriving it a second time -- two derivations
        # disagree the first time the rule changes.
        "displaced_stages": [s for s in displaced if s],
    }
    if note.strip():
        config["operator_clues"] = [{"note": note.strip()[:2000], "iteration": 0}]

    child_config = {
        "name": f"{(parent.job.config or {}).get('pipeline') or 'pipeline'}: {new_id}",
        "goal": goal,
        "job_type": str(stage.get("job_type") or parent.job.job_type or "research"),
        "config": config,
        "max_iterations": int(stage.get("max_iterations") or 6),
        # The inserted stage takes over the parent's children, which is what
        # makes the rest of the chain re-derive beneath it instead of being
        # rebuilt by hand.
        "chain_config": {
            "trigger_condition": parent_chain.get("trigger_condition", "on_complete"),
            "child_jobs": parent_chain.get("child_jobs") or [],
        },
    }

    parent.job.chain_triggered = False
    child = await executor.chain_orchestration_service.create_chained_job(
        executor, parent.job, child_config, db
    )
    if child is None:
        raise PipelineRestartError(
            f"Could not create a job for stage {new_id!r}.", status_code=500
        )

    # The parent now points at the inserted stage alone; leaving the old
    # children there would fire them again beside it.
    parent.job.chain_config = {
        **parent_chain,
        "child_jobs": [child_config],
    }

    results = parent.job.results if isinstance(parent.job.results, dict) else {}
    inserted = results.get("stage_insertions")
    if not isinstance(inserted, list):
        inserted = []
    inserted.append(
        {
            "after": after_stage,
            "stage": new_id,
            "displaced": displaced,
            "reason": note.strip()[:2000],
        }
    )
    results["stage_insertions"] = inserted[-20:]
    parent.job.results = results
    flag_modified(parent.job, "results")
    parent.job.add_log_entry(
        {"phase": "stage_inserted", "stage": new_id, "after": after_stage}
    )

    await db.commit()
    from app.tasks.agent_job_tasks import execute_agent_job_task

    execute_agent_job_task.delay(str(child.id), str(child.user_id))
    logger.info(
        f"Inserted stage {new_id} after {after_stage} in run {root_job_id} "
        f"as job {child.id}"
    )
    return child
