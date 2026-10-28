"""A pipeline stage that ends short of its contract is retried once, by itself.

A stage that failed stopped the plan, and the note a person then wrote to
restart it was assembled from things the run already knew: what the contract
lacked, which tool produces it, why the attempt stopped. The retry writes that
note itself and goes through the restart path a person's restart takes.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.modules.autonomy.application import checkpoint_queue_jobs
from app.services import agent_pipeline_restart as restart
from app.services import agent_runtime_finalizer
from app.services import agent_stage_auto_retry as auto
from tests.test_agent_runtime_finalizer import (
    _DummyDb,
    _DummyJob,
    _state_with,
    _UnmetContractExecutor,
)
from tests.test_pipeline_restart_from_stage import _Executor, _stage_job

pytestmark = pytest.mark.unit

UNMET = {
    "enabled": True,
    "satisfied": False,
    "missing": ["finding_type:benchmark_measurement>=2"],
    "metrics": {},
}


def _job(status="completed", phase="", stage="measure", config=None):
    job = SimpleNamespace(
        id=uuid.uuid4(),
        status=status,
        current_phase=phase,
        job_type="research",
        config={"pipeline_stage": stage, **(config or {})} if stage else {},
        root_job_id=uuid.uuid4(),
        parent_job_id=None,
        log=[],
    )
    job.add_log_entry = job.log.append
    return job


class TestWhichEndingsQualify:
    def test_completed_with_the_contract_unmet(self):
        assert auto.qualifies(_job(), contract_unmet=True, blocked=None) == ""

    def test_blocked_on_input(self):
        job = _job(status="paused", phase="blocked_needs_input")
        assert auto.qualifies(job, contract_unmet=True, blocked={}) == ""

    def test_a_met_contract_is_not_retried(self):
        assert auto.qualifies(_job(), contract_unmet=False, blocked=None)

    def test_a_job_that_is_not_a_stage(self):
        assert "not a pipeline stage" in auto.qualifies(
            _job(stage=""), contract_unmet=True, blocked=None
        )

    @pytest.mark.parametrize(
        "status, phase",
        [("paused", "awaiting_approval"), ("cancelled", ""), ("failed", "")],
    )
    def test_endings_that_are_someone_elses_move(self, status, phase):
        assert auto.qualifies(
            _job(status=status, phase=phase), contract_unmet=True, blocked=None
        )

    def test_a_platform_blocker_is_not_retried(self):
        """A tool that never ran is the same on the second attempt; it is
        filed as coding work instead."""
        job = _job(status="paused", phase="blocked_needs_input")
        blocked = {"needs": {"answerable_by": "platform_change"}}
        assert "platform" in auto.qualifies(job, contract_unmet=True, blocked=blocked)

    def test_a_question_for_an_operator_is_still_retried(self):
        job = _job(status="paused", phase="blocked_needs_input")
        blocked = {"needs": {"answerable_by": "operator"}}
        assert auto.qualifies(job, contract_unmet=True, blocked=blocked) == ""


class TestTheNote:
    def test_it_names_what_is_missing_how_to_get_it_and_why_it_stopped(self):
        note = auto.build_note(
            job=_job(),
            state={
                "findings": [
                    {"type": "implementation_verified", "title": "fastmod passes"}
                ]
            },
            contract_eval=UNMET,
            blocked={
                "reason": "3 consecutive rounds produced no new findings",
                "needs": {"question": "What does benchmark_c_snippet accept here?"},
            },
        )

        assert note.startswith("AUTOMATIC RETRY")
        assert "finding_type:benchmark_measurement>=2" in note
        assert "benchmark_c_snippet" in note, "names the producing tool"
        assert "3 consecutive rounds" in note
        assert "What does benchmark_c_snippet accept" in note
        assert "fastmod passes" in note
        assert len(note) <= auto.MAX_NOTE_CHARS

    def test_a_validity_requirement_carries_its_remedy(self):
        contract = {
            "enabled": True,
            "satisfied": False,
            "missing": ["validity:records_method"],
            "metrics": {"validity": {"details": {}}},
        }
        note = auto.build_note(
            job=_job(), state={}, contract_eval=contract, blocked=None
        )
        assert "record_method" in note


@pytest.mark.asyncio
class TestTheRetry:
    @pytest.fixture
    def restarts(self, monkeypatch):
        calls = []
        stages = []

        async def _load(root_job_id, db):
            return stages

        async def _restart(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id=uuid.uuid4(), config=kwargs["extra_config"])

        monkeypatch.setattr(restart, "load_run", _load)
        monkeypatch.setattr(restart, "restart_from_stage", _restart)
        return SimpleNamespace(calls=calls, stages=stages)

    async def _retry(self, job, blocked=None):
        return await auto.retry(
            object(), job, state={}, contract_eval=UNMET, blocked=blocked, db=None
        )

    async def test_it_restarts_the_stage_marked_and_with_the_note(self, restarts):
        job = _job()
        child = await self._retry(job)

        (call,) = restarts.calls
        assert child is not None
        assert call["stage_id"] == "measure"
        assert call["root_job_id"] == job.root_job_id
        assert call["note"].startswith("AUTOMATIC RETRY")
        assert call["extra_config"] == {auto.RETRY_OF_KEY: str(job.id)}

    async def test_a_stage_is_retried_once(self, restarts):
        """The retry's own failure stops the plan, as before."""
        earlier = AgentJob(
            id=uuid.uuid4(),
            config={"pipeline_stage": "measure", auto.RETRY_OF_KEY: "x"},
        )
        restarts.stages.append(restart.StageJob(stage_id="measure", job=earlier))
        job = _job()

        assert await self._retry(job) is None
        assert restarts.calls == []
        assert job.log[-1]["phase"] == "stage_auto_retry_spent"

    async def test_another_stages_retry_does_not_spend_this_ones(self, restarts):
        other = AgentJob(
            id=uuid.uuid4(),
            config={"pipeline_stage": "implement", auto.RETRY_OF_KEY: "x"},
        )
        restarts.stages.append(restart.StageJob(stage_id="implement", job=other))

        assert await self._retry(_job()) is not None

    async def test_turned_off_it_does_nothing_and_says_nothing(
        self, restarts, monkeypatch
    ):
        from app.core.config import settings

        monkeypatch.setattr(settings, "AGENT_STAGE_AUTO_RETRIES", 0)
        job = _job()
        assert await self._retry(job) is None
        assert restarts.calls == [] and job.log == []

    async def test_a_stage_may_ask_for_more(self, restarts):
        earlier = AgentJob(
            id=uuid.uuid4(),
            config={"pipeline_stage": "measure", auto.RETRY_OF_KEY: "x"},
        )
        restarts.stages.append(restart.StageJob(stage_id="measure", job=earlier))

        assert await self._retry(_job(config={"auto_retries": 2})) is not None

    async def test_a_refused_restart_is_logged_not_raised(self, monkeypatch):
        async def _load(root_job_id, db):
            return []

        async def _refuse(**kwargs):
            raise restart.PipelineRestartError("Stage 'implement' is failed")

        monkeypatch.setattr(restart, "load_run", _load)
        monkeypatch.setattr(restart, "restart_from_stage", _refuse)
        job = _job()

        assert await self._retry(job) is None
        assert job.log[-1]["phase"] == "stage_auto_retry_refused"
        assert "implement" in job.log[-1]["reason"]


@pytest.mark.asyncio
async def test_the_marker_is_on_the_new_job_before_it_is_queued(
    monkeypatch, db_session
):
    """Set afterwards, a worker could start the retry without it, and it would
    not count against the limit."""
    root = uuid.uuid4()
    impl = _stage_job(
        "implement", AgentJobStatus.COMPLETED.value, root=root, children=("measure",)
    )
    meas = _stage_job("measure", AgentJobStatus.COMPLETED.value, parent=impl, root=root)

    async def _load(root_job_id, db):
        return [
            restart.StageJob(stage_id="implement", job=impl),
            restart.StageJob(stage_id="measure", job=meas),
        ]

    monkeypatch.setattr(restart, "load_run", _load)
    queued = []
    monkeypatch.setattr(
        "app.services.job_dispatch.enqueue_agent_job",
        lambda db, job_id, user_id: queued.append(job_id),
    )
    executor = _Executor()

    await restart.restart_from_stage(
        root_job_id=root,
        stage_id="measure",
        executor=executor,
        db=db_session,
        note="n",
        extra_config={auto.RETRY_OF_KEY: str(meas.id)},
    )

    ((_, child_config),) = executor.created
    assert child_config["config"][auto.RETRY_OF_KEY] == str(meas.id)
    assert child_config["config"]["pipeline_stage"] == "measure"


class _StageJob(_DummyJob):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.config = {"pipeline_stage": "measure"}
        self.root_job_id = uuid.uuid4()
        self.parent_job_id = uuid.uuid4()
        self.current_phase = None


@pytest.mark.asyncio
class TestTheFinalizer:
    async def _finalize(self, monkeypatch, job, *, child):
        monkeypatch.setattr(
            agent_runtime_finalizer.agent_job_memory_service,
            "extract_memories_from_job",
            AsyncMock(return_value=[]),
        )

        async def _retry(executor, job, **kwargs):
            return child

        monkeypatch.setattr(auto, "retry", _retry)
        executor = _UnmetContractExecutor()
        await agent_runtime_finalizer.finalize_job(
            executor, job, _state_with(40), _DummyDb()
        )
        return executor

    async def test_a_retried_stage_does_not_fire_its_chain(self, monkeypatch):
        job = _StageJob(status="running")
        child = SimpleNamespace(id=uuid.uuid4())

        executor = await self._finalize(monkeypatch, job, child=child)

        assert executor.trigger_calls == []
        assert job.status == AgentJobStatus.COMPLETED.value
        assert job.current_phase == auto.RETRIED_PHASE
        assert job.log_entries[-1] == {
            "phase": auto.RETRIED_PHASE,
            "stage": "measure",
            "job_id": str(child.id),
        }

    async def test_without_a_retry_the_stage_ends_as_before(self, monkeypatch):
        """The control: the chain is still told the stage completed, and the
        unmet contract is still in the log."""
        job = _StageJob(status="running")

        executor = await self._finalize(monkeypatch, job, child=None)

        assert executor.trigger_calls == ["complete"]
        phases = [e.get("phase") for e in job.log_entries]
        assert "completed_contract_unmet" in phases
        assert auto.RETRIED_PHASE not in phases


class TestTheQueue:
    def _completed(self, log):
        return SimpleNamespace(status="completed", execution_log=log)

    def test_a_retried_attempt_is_not_a_contract_unmet_row(self):
        job = self._completed(
            [{"phase": "completed_contract_unmet"}, {"phase": auto.RETRIED_PHASE}]
        )
        assert checkpoint_queue_jobs._finished_without_meeting_its_contract(job) == {}

    def test_an_attempt_not_retried_still_is(self):
        job = self._completed([{"phase": "completed_contract_unmet", "missing": []}])
        assert checkpoint_queue_jobs._finished_without_meeting_its_contract(job)

    def test_the_restated_phase_matches(self):
        assert checkpoint_queue_jobs.RETRIED_PHASE == auto.RETRIED_PHASE
