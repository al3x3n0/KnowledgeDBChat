"""Restarting a pipeline from one of its stages.

A pipeline that stops four stages in has usually stopped for a reason that
belongs to one stage. Re-launching the whole thing re-ingests the paper,
re-specifies and re-implements to get back to where it already was -- paying
for four stages to retry the fifth, and producing a DIFFERENT run, so the
evidence the retry builds on is not the evidence that was established.

The refusals are the substance here. A restart is only worth anything if the
ground under it is real, and this project has already watched a benchmark stage
start on an implementation nothing had verified.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_pipeline_restart as restart

pytestmark = pytest.mark.unit


def _stage_job(stage, status, *, parent=None, root=None, contract=True, children=()):
    job = AgentJob(
        id=uuid.uuid4(),
        name=f"p: {stage}",
        goal=f"do {stage}",
        job_type="research",
        status=status,
        iteration=3,
        max_iterations=10,
        config={"pipeline": "p", "pipeline_stage": stage},
        results={
            "goal_contract": {"enabled": True, "satisfied": contract},
        },
        execution_log=[],
        chain_config={
            "trigger_condition": "on_complete",
            "child_jobs": [
                {"config": {"pipeline_stage": c}, "goal": f"do {c}"} for c in children
            ],
        },
    )
    job.parent_job_id = parent.id if parent else None
    job.root_job_id = root
    job.chain_triggered = True
    # Every real job has an owner, and the head rerun clones it onto the new
    # job; a helper that omits it hides a NOT NULL violation until then.
    job.user_id = uuid.uuid4()
    return job


class _Executor:
    def __init__(self):
        self.created = []

    class _Chain:
        def __init__(self, outer):
            self.outer = outer

        async def create_chained_job(self, executor, parent_job, child_config, db):
            self.outer.created.append((parent_job, child_config))
            child = AgentJob(
                id=uuid.uuid4(),
                name="child",
                goal="child",
                job_type="research",
                status=AgentJobStatus.PENDING.value,
                config=dict(child_config.get("config") or {}),
                results={},
                execution_log=[],
            )
            child.user_id = uuid.uuid4()
            return child

    @property
    def chain_orchestration_service(self):
        return _Executor._Chain(self)


@pytest.mark.asyncio
class TestItRefusesGroundThatIsNotThere:
    async def _run(self, monkeypatch, stages):
        async def _load(root_job_id, db):
            return stages

        monkeypatch.setattr(restart, "load_run", _load)

    async def test_a_stage_this_run_does_not_have(self, monkeypatch, db_session):
        root = uuid.uuid4()
        found = _stage_job("find", AgentJobStatus.COMPLETED.value, root=root)
        await self._run(monkeypatch, [restart.StageJob(stage_id="find", job=found)])

        with pytest.raises(restart.PipelineRestartError) as error:
            await restart.restart_from_stage(
                root_job_id=root,
                stage_id="measure",
                executor=_Executor(),
                db=db_session,
            )

        assert "no stage 'measure'" in error.value.detail
        assert "find" in error.value.detail, "say what the run does have"

    async def test_a_stage_that_is_still_running(self, monkeypatch, db_session):
        """Two runners on one stage is what the execution lease exists to
        prevent; there is no reason to create it deliberately."""
        root = uuid.uuid4()
        found = _stage_job("find", AgentJobStatus.COMPLETED.value, root=root)
        spec = _stage_job(
            "specify", AgentJobStatus.RUNNING.value, parent=found, root=root
        )
        await self._run(
            monkeypatch,
            [
                restart.StageJob(stage_id="find", job=found),
                restart.StageJob(stage_id="specify", job=spec),
            ],
        )

        with pytest.raises(restart.PipelineRestartError) as error:
            await restart.restart_from_stage(
                root_job_id=root,
                stage_id="specify",
                executor=_Executor(),
                db=db_session,
            )

        assert "still running" in error.value.detail

    async def test_a_predecessor_that_never_met_its_contract(
        self, monkeypatch, db_session
    ):
        """The one that matters. A stage can COMPLETE without producing what it
        promised, and restarting the next stage on it rebuilds on evidence that
        does not exist -- observed live as a benchmark of unverified code."""
        root = uuid.uuid4()
        impl = _stage_job(
            "implement",
            AgentJobStatus.COMPLETED.value,
            root=root,
            contract=False,
            children=("measure",),
        )
        meas = _stage_job(
            "measure", AgentJobStatus.PAUSED.value, parent=impl, root=root
        )
        await self._run(
            monkeypatch,
            [
                restart.StageJob(stage_id="implement", job=impl),
                restart.StageJob(stage_id="measure", job=meas),
            ],
        )

        with pytest.raises(restart.PipelineRestartError) as error:
            await restart.restart_from_stage(
                root_job_id=root,
                stage_id="measure",
                executor=_Executor(),
                db=db_session,
            )

        assert "without meeting its contract" in error.value.detail
        assert "Restart from that stage instead" in error.value.detail

    async def test_the_head_is_re_run_rather_than_refused(
        self, monkeypatch, db_session, test_user
    ):
        """This asserted a refusal until the head turned out to be the stage a
        run most often needs redone: a `mine` stage whose profile was too
        coarse needs the profile again, and profile is the head. Refusing it
        made `may_revisit: [<head>]` a promise the system could not keep -- the
        spec validated the edge and the tool accepted the request, and only the
        finaliser found out it could not act.
        """
        root = uuid.uuid4()
        found = _stage_job("find", AgentJobStatus.PAUSED.value)
        found.id = root
        found.user_id = test_user.id
        found.root_job_id = None
        await self._run(monkeypatch, [restart.StageJob(stage_id="find", job=found)])
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: None,
        )

        child = await restart.restart_from_stage(
            root_job_id=root, stage_id="find", executor=_Executor(), db=db_session
        )

        assert child.id != found.id
        assert child.goal == found.goal


@pytest.mark.asyncio
class TestARestartBuildsOnWhatWasEstablished:
    async def test_it_refires_the_chain_from_the_completed_predecessor(
        self, monkeypatch, db_session
    ):
        root = uuid.uuid4()
        meas = _stage_job(
            "measure",
            AgentJobStatus.COMPLETED.value,
            root=root,
            children=("compare",),
        )
        cmp_job = _stage_job(
            "compare", AgentJobStatus.PAUSED.value, parent=meas, root=root
        )
        stages = [
            restart.StageJob(stage_id="measure", job=meas),
            restart.StageJob(stage_id="compare", job=cmp_job),
        ]

        async def _load(root_job_id, db):
            return stages

        monkeypatch.setattr(restart, "load_run", _load)
        dispatched = []
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: dispatched.append(job_id),
        )
        executor = _Executor()

        child = await restart.restart_from_stage(
            root_job_id=root,
            stage_id="compare",
            executor=executor,
            db=db_session,
            note="the claimed speedup is in table 3",
        )

        parent_used, child_config = executor.created[0]
        assert parent_used is meas, "must build on the predecessor's evidence"
        # Without clearing this the orchestration treats the stage as already
        # started and does nothing, silently.
        assert meas.chain_triggered is False
        # The correction has to land where the run reads it.
        clues = child_config["config"]["operator_clues"]
        assert "table 3" in clues[0]["note"]
        assert dispatched == [str(child.id)], "a job that is never queued never runs"

    async def test_no_note_is_fine_and_attaches_nothing(self, monkeypatch, db_session):
        root = uuid.uuid4()
        meas = _stage_job(
            "measure",
            AgentJobStatus.COMPLETED.value,
            root=root,
            children=("compare",),
        )
        cmp_job = _stage_job(
            "compare", AgentJobStatus.PAUSED.value, parent=meas, root=root
        )

        async def _load(root_job_id, db):
            return [
                restart.StageJob(stage_id="measure", job=meas),
                restart.StageJob(stage_id="compare", job=cmp_job),
            ]

        monkeypatch.setattr(restart, "load_run", _load)
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: None,
        )
        executor = _Executor()

        await restart.restart_from_stage(
            root_job_id=root, stage_id="compare", executor=executor, db=db_session
        )

        _, child_config = executor.created[0]
        assert "operator_clues" not in child_config["config"]


class TestTheLatestAttemptIsTheOneThatCounts:
    def test_a_restarted_stage_supersedes_its_earlier_attempt(self):
        """A run restarted before has more than one job for a stage. Restarting
        again must build on the newest, not resurrect the superseded one."""
        first = _stage_job("compare", AgentJobStatus.PAUSED.value)
        second = _stage_job("compare", AgentJobStatus.COMPLETED.value)

        latest = restart.latest_per_stage(
            [
                restart.StageJob(stage_id="compare", job=first),
                restart.StageJob(stage_id="compare", job=second),
            ]
        )

        assert latest["compare"].job is second


@pytest.mark.asyncio
class TestTheHeadCanBeRerun:
    """`may_revisit: [<head>]` was a promise the system could not keep.

    Measured: a `mine` stage needed its PROFILE redone, and profile is the head
    of that pipeline. The spec validated the edge -- profile is genuinely an
    ancestor of mine -- and `request_stage_rerun` would have accepted the
    request, and then the finaliser would have quietly logged that it could not
    act, because the head has no predecessor to re-fire the chain from.

    The head is the stage a run most often needs redone, so refusing it left
    the feature useful mainly where it was least needed.
    """

    async def test_the_head_is_re_run_from_its_own_definition(
        self, monkeypatch, db_session, test_user
    ):
        from app.services import agent_pipeline_restart as restart

        root = uuid.uuid4()
        head = _stage_job("profile", AgentJobStatus.COMPLETED.value, children=("mine",))
        head.id = root
        head.user_id = test_user.id
        head.root_job_id = None

        async def _load(root_job_id, db):
            return [restart.StageJob(stage_id="profile", job=head)]

        monkeypatch.setattr(restart, "load_run", _load)
        dispatched = []
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: dispatched.append(job_id),
        )

        child = await restart.restart_from_stage(
            root_job_id=root,
            stage_id="profile",
            executor=object(),
            db=db_session,
            note="profile at instruction level: the block counts were too coarse",
        )

        assert child.id != head.id, "a clone, so the first attempt stays readable"
        assert child.goal == head.goal
        assert (
            child.chain_config == head.chain_config
        ), "the stages after it must re-derive exactly as they did before"
        # The correction is the only new input, and it must be readable.
        assert "instruction level" in child.config["operator_clues"][0]["note"]
        # Same run, so the stage view and the ledger still see one pipeline.
        assert child.root_job_id == root
        assert dispatched == [str(child.id)], "a job never queued never runs"

    async def test_a_head_rerun_with_no_note_carries_no_clue(
        self, monkeypatch, db_session, test_user
    ):
        from app.services import agent_pipeline_restart as restart

        root = uuid.uuid4()
        head = _stage_job("profile", AgentJobStatus.COMPLETED.value)
        head.id = root
        head.user_id = test_user.id
        head.root_job_id = None

        async def _load(root_job_id, db):
            return [restart.StageJob(stage_id="profile", job=head)]

        monkeypatch.setattr(restart, "load_run", _load)
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: None,
        )

        child = await restart.restart_from_stage(
            root_job_id=root, stage_id="profile", executor=object(), db=db_session
        )

        assert "operator_clues" not in child.config
