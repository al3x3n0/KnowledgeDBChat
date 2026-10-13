"""A run proposes the pipeline that should follow it; a person decides.

The step from one job to a plan was always a person's. `propose_pipeline`
lets a run take it, with the judging left where it was: the draft goes
through the same checker and the same pricing a person's pipeline does, and
nothing starts until someone approves.

The model is scripted (it returns a spec); everything that happens to that
spec is real -- the drafter's own checks, the binding, the job row, the
approval hold, and the chain runtime that starts the first stage.
"""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_self_planned_pipeline as planning
from app.services.agent_runtime_finalizer import (
    CHAIN_GATE_CHECKPOINT,
    hold_for_chain_approval,
)

pytestmark = pytest.mark.unit

SPEC = {
    "name": "fastmod-study",
    "stages": [
        {
            "id": "find",
            "goal": "Ingest the paper describing the algorithm",
            "contract": {"required_finding_types": ["papers_ingested"]},
        },
        {
            "id": "specify",
            "goal": "Read the paper into an implementable specification",
            "depends_on": ["find"],
            "assumes": ["papers_ingested"],
            "contract": {"required_finding_types": ["algorithm_spec"]},
        },
    ],
}
ASK = "Find the paper on fast modular reduction and write its algorithm down as a specification."


class _Model:
    """Returns whatever spec it was given, and records what it was asked."""

    def __init__(self, *specs):
        self.specs = list(specs)
        self.asked = []

    async def generate_structured(self, **kwargs):
        self.asked.append(kwargs["user_message"])
        return copy.deepcopy(
            self.specs.pop(0) if len(self.specs) > 1 else self.specs[0]
        )


async def _run(db, user, **fields):
    values = dict(
        name="explore fastmod",
        goal="See whether fast modular reduction applies to our kernel",
        job_type="research",
        user_id=user.id,
        status=AgentJobStatus.RUNNING.value,
        config={},
    )
    values.update(fields)
    job = AgentJob(**values)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _propose(db, job, model=None, state=None, description=ASK):
    return await planning.propose(
        job,
        description,
        db=db,
        llm_service=model or _Model(SPEC),
        state={} if state is None else state,
    )


# --------------------------------------------------------------------------
# proposing
# --------------------------------------------------------------------------


async def test_a_run_attaches_a_checked_and_priced_plan(db_session, test_user):
    job = await _run(db_session, test_user)
    model = _Model(SPEC)

    result = await _propose(db_session, job, model)

    assert result["success"] is True, result
    assert result["data"]["stages"] == ["find", "specify"]
    assert result["data"]["estimated_seconds"] > 0
    assert "not started" in result["data"]["note"]
    assert result["findings"][0]["type"] == "pipeline_proposal"
    # The drafter was told what this run was for.
    assert "fast modular reduction applies to our kernel" in model.asked[0]

    chain = job.chain_config
    assert chain["trigger_condition"] == "on_approval"
    (root,) = chain["child_jobs"]
    assert root["config"]["pipeline_stage"] == "find"
    assert (
        root["chain_config"]["child_jobs"][0]["config"]["pipeline_stage"] == "specify"
    )
    assert planning.proposal_of(job)["stages"] == ["find", "specify"]


async def test_nothing_is_created_by_proposing(db_session, test_user):
    job = await _run(db_session, test_user)

    await _propose(db_session, job)
    await db_session.commit()

    jobs = (await db_session.execute(select(AgentJob))).scalars().all()
    assert [j.id for j in jobs] == [job.id]


async def test_a_plan_the_checker_refuses_is_not_attached(db_session, test_user):
    """The stage requires evidence no tool produces."""
    broken = copy.deepcopy(SPEC)
    broken["stages"][1]["contract"] = {"required_finding_types": ["no_such_evidence"]}
    job = await _run(db_session, test_user)

    result = await _propose(db_session, job, _Model(broken))

    assert "does not pass the checker" in result["error"]
    assert result["data"]["problems"]
    assert not job.chain_config


async def test_a_plan_over_the_stage_limit_is_refused_with_the_figure(
    db_session, test_user, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_SELF_PLANNED_PIPELINE_MAX_STAGES", 1)
    job = await _run(db_session, test_user)

    result = await _propose(db_session, job)

    assert "2 stages" in result["error"] and "at most 1" in result["error"]
    assert not job.chain_config


async def test_a_plan_over_the_time_limit_is_refused_with_the_figure(
    db_session, test_user, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_SELF_PLANNED_PIPELINE_MAX_SECONDS", 1)
    job = await _run(db_session, test_user)

    result = await _propose(db_session, job)

    assert "at most 1s" in result["error"]
    assert result["data"]["estimated_seconds"] > 1
    assert not job.chain_config


@pytest.mark.parametrize("description", ["", "   ", "do more"])
async def test_a_plan_needs_a_description(db_session, test_user, description):
    job = await _run(db_session, test_user)
    model = _Model(SPEC)

    result = await _propose(db_session, job, model, description=description)

    assert "Describe the follow-up" in result["error"]
    assert model.asked == []


# --------------------------------------------------------------------------
# who may, and how often
# --------------------------------------------------------------------------


async def test_a_stage_of_a_pipeline_may_not_plan_one(db_session, test_user):
    parent = await _run(db_session, test_user)
    child = await _run(db_session, test_user, parent_job_id=parent.id)
    stage = await _run(db_session, test_user, config={"pipeline_stage": "find"})
    model = _Model(SPEC)

    for job in (child, stage):
        result = await _propose(db_session, job, model)
        assert "nobody's stage" in result["error"]
        assert not job.chain_config
    assert model.asked == []


async def test_a_chain_somebody_else_configured_is_not_overwritten(
    db_session, test_user
):
    theirs = {
        "trigger_condition": "on_complete",
        "child_jobs": [{"name": "next", "goal": "g", "job_type": "research"}],
    }
    job = await _run(db_session, test_user, chain_config=copy.deepcopy(theirs))

    result = await _propose(db_session, job)

    assert "already has work configured" in result["error"]
    assert job.chain_config == theirs


async def test_a_run_may_replace_its_own_plan(db_session, test_user):
    job = await _run(db_session, test_user)
    shorter = {"name": "just-find", "stages": [SPEC["stages"][0]]}
    state = {}

    await _propose(db_session, job, state=state)
    result = await _propose(db_session, job, _Model(shorter), state=state)

    assert result["success"] is True
    assert planning.proposal_of(job)["stages"] == ["find"]


async def test_a_run_cannot_keep_asking(db_session, test_user):
    job = await _run(db_session, test_user)
    state = {}
    model = _Model(SPEC)

    for _ in range(planning.MAX_PROPOSALS_PER_RUN):
        await _propose(db_session, job, model, state=state)
    result = await _propose(db_session, job, model, state=state)

    assert "already proposed a pipeline" in result["error"]
    assert len(model.asked) == planning.MAX_PROPOSALS_PER_RUN


async def test_it_can_be_turned_off(db_session, test_user, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_SELF_PLANNED_PIPELINES_ENABLED", False)
    job = await _run(db_session, test_user)

    result = await _propose(db_session, job)

    assert "AGENT_SELF_PLANNED_PIPELINES_ENABLED" in result["error"]


# --------------------------------------------------------------------------
# a person decides
# --------------------------------------------------------------------------


async def test_a_finished_run_waits_with_its_whole_plan_shown(db_session, test_user):
    job = await _run(db_session, test_user)
    await _propose(db_session, job)
    job.status = AgentJobStatus.COMPLETED.value
    job.results = {"summary": "It applies."}

    hold_for_chain_approval(job)

    assert job.status == AgentJobStatus.PAUSED.value
    checkpoint = job.results["approval_checkpoint"]
    assert checkpoint["checkpoint_type"] == CHAIN_GATE_CHECKPOINT
    assert "proposes a follow-up pipeline of 2 stages" in checkpoint["message"]
    assert "find → specify" in checkpoint["message"]
    assert checkpoint["waiting_stages"] == ["find", "specify"]
    plan = checkpoint["proposed_pipeline"]
    assert plan["estimated_seconds"] > 0
    assert plan["stage_goals"]["specify"].startswith("Read the paper")
    # What the run concluded is still there beside it.
    assert job.results["summary"] == "It applies."


async def test_it_appears_among_what_is_waiting_on_the_person(db_session, test_user):
    from app.modules.review.application.review_inbox import build_review_inbox

    job = await _run(db_session, test_user)
    await _propose(db_session, job)
    job.status = AgentJobStatus.COMPLETED.value
    job.results = {}
    hold_for_chain_approval(job)
    await db_session.commit()

    runs = next(
        s for s in await build_review_inbox(db_session, test_user) if s.key == "runs"
    )

    assert runs.count == 1
    assert runs.items[0].kind == "approval_checkpoint"


async def test_approving_starts_the_first_stage_and_only_that(db_session, test_user):
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = await _run(db_session, test_user)
    await _propose(db_session, job)
    job.status = AgentJobStatus.COMPLETED.value
    job.results = {"findings": []}
    await db_session.commit()

    with patch("app.tasks.agent_job_tasks.execute_agent_job_task.delay") as delay:
        started = await AutonomousAgentExecutor()._trigger_chained_jobs(
            job, "approval", db_session
        )
    await db_session.commit()

    assert len(started) == 1
    children = (
        (
            await db_session.execute(
                select(AgentJob).where(AgentJob.parent_job_id == job.id)
            )
        )
        .scalars()
        .all()
    )
    (first,) = children
    assert first.config["pipeline_stage"] == "find"
    assert first.config["goal_contract"]["required_finding_types"] == [
        "papers_ingested"
    ]
    # The rest of the plan travels with it, to start when it is done.
    assert first.chain_config["child_jobs"][0]["config"]["pipeline_stage"] == "specify"
    assert delay.called


async def test_completing_does_not_start_the_plan(db_session, test_user):
    """Only approval does: the trigger a run writes is `on_approval`."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = await _run(db_session, test_user)
    await _propose(db_session, job)
    job.status = AgentJobStatus.COMPLETED.value
    await db_session.commit()

    with patch("app.tasks.agent_job_tasks.execute_agent_job_task.delay") as delay:
        started = await AutonomousAgentExecutor()._trigger_chained_jobs(
            job, "complete", db_session
        )

    assert list(started) == [] and not delay.called


# --------------------------------------------------------------------------
# the tool a run calls
# --------------------------------------------------------------------------


async def test_the_tool_is_declared_and_answers_through_its_handler(
    db_session, test_user
):
    from app.agent_core.tool_specs import spec_for
    from app.services.agent_tool_providers.base import AgentToolExecutionContext
    from app.services.agent_tool_providers.collaboration import (
        build_autonomous_collaboration_provider,
    )

    spec = spec_for("propose_pipeline")
    assert spec.effects == "write" and spec.job_types is None
    job = await _run(db_session, test_user)
    handler = build_autonomous_collaboration_provider(
        SimpleNamespace(llm_service=_Model(SPEC))
    )._handlers["propose_pipeline"]
    state = {}

    result = await handler(
        {"description": ASK},
        AgentToolExecutionContext(
            mode="autonomous",
            db=db_session,
            service=None,
            user_id=test_user.id,
            job=job,
            state=state,
        ),
    )

    assert result["success"] is True, result
    assert state["pipeline_proposal"]["stages"] == ["find", "specify"]
    assert "spec" not in state["pipeline_proposal"]
