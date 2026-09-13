"""Inserting a stage the plan did not have, without redoing what worked.

A run stops for a reason belonging to one place, and the fix is often a step
nobody thought to include: a finer profile, a conversion, a check. The other
two repairs are worse there. Restarting the failed stage runs it again on the
inputs that already defeated it -- observed twice on one pipeline, where a
`mine` stage failed, its profile was redone, and it failed identically because
the profile was never the problem. Relaunching pays for every earlier stage to
reach the same point, in a different run, so the evidence the retry builds on
is not the evidence that was established.

The inserted stage takes over its parent's children, which is what makes the
rest of the chain re-derive beneath it instead of being rebuilt by hand.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_pipeline_restart as restart

pytestmark = pytest.mark.unit

GOOD_STAGE = {
    "id": "reprofile_fine",
    "goal": "Profile again at instruction level so the hot block's sequence is visible",
    "contract": {"required_finding_types": ["dynamic_profile"]},
}


def _job(stage, status, *, children=(), contract=True, parent=None):
    job = AgentJob(
        id=uuid.uuid4(),
        name=f"p: {stage}",
        goal=f"do {stage}",
        job_type="research",
        status=status,
        iteration=3,
        max_iterations=10,
        config={"pipeline": "p", "pipeline_stage": stage},
        results={"goal_contract": {"enabled": True, "satisfied": contract}},
        execution_log=[],
        chain_config={
            "trigger_condition": "on_complete",
            "child_jobs": [
                {"config": {"pipeline_stage": c}, "goal": f"do {c}"} for c in children
            ],
        },
    )
    job.parent_job_id = parent.id if parent else None
    job.user_id = uuid.uuid4()
    job.chain_triggered = True
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
                name=child_config["name"],
                goal=child_config["goal"],
                job_type=child_config["job_type"],
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
class TestItRefusesToBuildOnNothing:
    async def _with(self, monkeypatch, stages):
        async def _load(root_job_id, db):
            return stages

        monkeypatch.setattr(restart, "load_run", _load)

    async def test_a_parent_that_did_not_finish(self, monkeypatch, db_session):
        parent = _job("profile", AgentJobStatus.RUNNING.value, children=("mine",))
        await self._with(monkeypatch, [restart.StageJob("profile", parent)])

        with pytest.raises(restart.PipelineRestartError) as e:
            await restart.insert_stage_after(
                root_job_id=uuid.uuid4(),
                after_stage="profile",
                stage=GOOD_STAGE,
                executor=_Executor(),
                db=db_session,
            )

        assert "not completed" in e.value.detail

    async def test_a_parent_whose_contract_went_unmet(self, monkeypatch, db_session):
        """The rule a restart makes, for the same reason: a stage inserted on
        top of unestablished evidence is built on nothing."""
        parent = _job(
            "profile",
            AgentJobStatus.COMPLETED.value,
            children=("mine",),
            contract=False,
        )
        await self._with(monkeypatch, [restart.StageJob("profile", parent)])

        with pytest.raises(restart.PipelineRestartError) as e:
            await restart.insert_stage_after(
                root_job_id=uuid.uuid4(),
                after_stage="profile",
                stage=GOOD_STAGE,
                executor=_Executor(),
                db=db_session,
            )

        assert "without meeting its contract" in e.value.detail

    async def test_an_id_the_run_already_uses(self, monkeypatch, db_session):
        parent = _job("profile", AgentJobStatus.COMPLETED.value, children=("mine",))
        mine = _job("mine", AgentJobStatus.PAUSED.value, parent=parent)
        await self._with(
            monkeypatch,
            [restart.StageJob("profile", parent), restart.StageJob("mine", mine)],
        )

        with pytest.raises(restart.PipelineRestartError) as e:
            await restart.insert_stage_after(
                root_job_id=uuid.uuid4(),
                after_stage="profile",
                stage={**GOOD_STAGE, "id": "mine"},
                executor=_Executor(),
                db=db_session,
            )

        assert "already has a stage" in e.value.detail
        assert "which stage a finding came from" in e.value.detail

    async def test_a_stage_with_no_contract(self, monkeypatch, db_session):
        """A stage with nothing to satisfy cannot fail, so it cannot be the fix
        for a stage that did."""
        parent = _job("profile", AgentJobStatus.COMPLETED.value, children=("mine",))
        await self._with(monkeypatch, [restart.StageJob("profile", parent)])

        with pytest.raises(restart.PipelineRestartError) as e:
            await restart.insert_stage_after(
                root_job_id=uuid.uuid4(),
                after_stage="profile",
                stage={"id": "x", "goal": "do something"},
                executor=_Executor(),
                db=db_session,
            )

        assert "needs a contract" in e.value.detail

    async def test_a_running_stage_would_be_raced(self, monkeypatch, db_session):
        parent = _job("profile", AgentJobStatus.COMPLETED.value, children=("mine",))
        mine = _job("mine", AgentJobStatus.RUNNING.value, parent=parent)
        await self._with(
            monkeypatch,
            [restart.StageJob("profile", parent), restart.StageJob("mine", mine)],
        )

        with pytest.raises(restart.PipelineRestartError) as e:
            await restart.insert_stage_after(
                root_job_id=uuid.uuid4(),
                after_stage="profile",
                stage=GOOD_STAGE,
                executor=_Executor(),
                db=db_session,
            )

        assert "would be superseded" in e.value.detail


@pytest.mark.asyncio
class TestTheChainReDerivesBeneathIt:
    async def test_the_new_stage_takes_over_the_parents_children(
        self, monkeypatch, db_session
    ):
        parent = _job("profile", AgentJobStatus.COMPLETED.value, children=("mine",))
        mine = _job("mine", AgentJobStatus.PAUSED.value, parent=parent)

        async def _load(root_job_id, db):
            return [restart.StageJob("profile", parent), restart.StageJob("mine", mine)]

        monkeypatch.setattr(restart, "load_run", _load)
        dispatched = []
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: dispatched.append(job_id),
        )
        executor = _Executor()

        child = await restart.insert_stage_after(
            root_job_id=uuid.uuid4(),
            after_stage="profile",
            stage=GOOD_STAGE,
            executor=executor,
            db=db_session,
            note="the block counts were too coarse to mine",
        )

        used_parent, child_config = executor.created[0]
        assert used_parent is parent
        # The whole point: mine runs again beneath the new stage.
        assert [
            c["config"]["pipeline_stage"]
            for c in child_config["chain_config"]["child_jobs"]
        ] == ["mine"]
        # And the parent now points only at the insertion, or both would fire.
        assert [
            c["config"]["pipeline_stage"] for c in parent.chain_config["child_jobs"]
        ] == ["reprofile_fine"]
        assert parent.chain_triggered is False
        assert dispatched == [str(child.id)], "a job never queued never runs"

    async def test_the_correction_and_the_displacement_are_recorded(
        self, monkeypatch, db_session
    ):
        parent = _job("profile", AgentJobStatus.COMPLETED.value, children=("mine",))

        async def _load(root_job_id, db):
            return [restart.StageJob("profile", parent)]

        monkeypatch.setattr(restart, "load_run", _load)
        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
            lambda job_id, user_id: None,
        )

        child = await restart.insert_stage_after(
            root_job_id=uuid.uuid4(),
            after_stage="profile",
            stage=GOOD_STAGE,
            executor=_Executor(),
            db=db_session,
            note="too coarse to mine",
        )

        assert child.config["operator_clues"][0]["note"] == "too coarse to mine"
        assert child.config["displaced_stages"] == ["mine"]
        # A run that needed an extra stage is not the same as one that did not.
        entry = parent.results["stage_insertions"][-1]
        assert entry["after"] == "profile" and entry["stage"] == "reprofile_fine"
        assert "stage_inserted" in [e.get("phase") for e in parent.execution_log]
