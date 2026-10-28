"""A campaign follows the plan its own job proposed.

A campaign launches one job per question and read that one job: its status
to settle the question, its findings to decide what to ask next. A job may
now propose a pipeline, and then neither holds. The job pauses for approval,
which the campaign saw only as "still running"; and once approved the work
is done by the stages, whose findings the campaign never read.

The model is scripted. The proposal, the approval hold, the chain runtime
that creates each stage and the campaign's own step are real.
"""

import copy
from datetime import timedelta
from unittest.mock import patch
from uuid import UUID

import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.research_campaign import CampaignItemStatus, CampaignStatus
from app.services import agent_self_planned_pipeline as planning
from app.services import research_campaign_service as campaigns
from app.services.agent_runtime_finalizer import hold_for_chain_approval
from app.utils.datetimes import utc_now

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
    async def generate_structured(self, **kwargs):
        return copy.deepcopy(SPEC)


async def _campaign(db, user, **overrides):
    kwargs = dict(
        user_id=user.id,
        name="fastmod",
        goal="find out whether fast modular reduction is worth adopting",
        items=[{"title": "survey the method"}, {"title": "measure it"}],
        max_jobs=5,
    )
    kwargs.update(overrides)
    campaign = await campaigns.create_campaign(db, **kwargs)
    await db.commit()
    return campaign


async def _launched(db, campaign):
    step = await campaigns.advance(db, campaign)
    await db.commit()
    return await db.get(AgentJob, UUID(step["launched_job"]))


async def _proposes_and_completes(db, job):
    """The job proposes a plan, completes, and is held for approval."""
    result = await planning.propose(job, ASK, db=db, llm_service=_Model(), state={})
    assert result.get("success") is True, result
    job.status = AgentJobStatus.COMPLETED.value
    job.completed_at = utc_now()
    job.results = {
        "goal_contract": {"satisfied": True},
        "findings": [{"type": "note", "title": "the method exists"}],
    }
    hold_for_chain_approval(job)
    await db.commit()
    assert job.status == AgentJobStatus.PAUSED.value


async def _start_next(db, job, event):
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    with patch("app.tasks.agent_job_tasks.execute_agent_job_task.delay"):
        started = await AutonomousAgentExecutor()._trigger_chained_jobs(job, event, db)
    job.chain_triggered = True
    await db.commit()
    return list(started)


async def _stage_of(db, parent):
    return (
        (await db.execute(select(AgentJob).where(AgentJob.parent_job_id == parent.id)))
        .scalars()
        .one()
    )


async def _ends(db, job, *, status="completed", findings=()):
    job.status = status
    job.completed_at = utc_now()
    job.results = {
        "goal_contract": {"satisfied": status == "completed"},
        "findings": list(findings),
    }
    await db.commit()


async def _approved_and_run(db, job, *, second="completed"):
    """Approve the plan and run both stages to their end."""
    job.status = AgentJobStatus.COMPLETED.value
    await _start_next(db, job, "approval")
    find = await _stage_of(db, job)
    await _ends(
        db, find, findings=[{"type": "papers_ingested", "title": "Lemire 2019"}]
    )
    await _start_next(db, find, "complete")
    specify = await _stage_of(db, find)
    await _ends(
        db,
        specify,
        status=second,
        findings=(
            [{"type": "algorithm_spec", "title": "fastmod u32"}]
            if second == "completed"
            else []
        ),
    )
    return find, specify


# --------------------------------------------------------------------------
# waiting
# --------------------------------------------------------------------------


async def test_a_plan_waiting_for_approval_is_named(db_session, test_user):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)

    step = await campaigns.advance(db_session, campaign)

    assert step["action"] == "waiting"
    assert step["waiting_on"] == "approval"
    assert step["launched_job"] is None
    summary = await campaigns.summarize(db_session, campaign)
    assert summary["waiting_on"] == "approval"


async def test_a_job_at_work_is_told_apart_from_one_waiting_on_a_person(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)

    assert (await campaigns.advance(db_session, campaign))["waiting_on"] == "job"

    job.status = AgentJobStatus.PAUSED.value
    job.current_phase = "blocked_needs_input"
    await db_session.commit()

    assert (await campaigns.advance(db_session, campaign))["waiting_on"] == "person"


async def test_the_question_is_not_settled_while_its_stages_run(db_session, test_user):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    job.status = AgentJobStatus.COMPLETED.value
    await _start_next(db_session, job, "approval")

    step = await campaigns.advance(db_session, campaign)

    # The job the campaign launched has completed; its first stage has not.
    assert step["action"] == "waiting" and step["waiting_on"] == "job"
    assert step["settled"] == 0
    assert campaign.jobs_launched == 1


async def test_a_stage_that_just_ended_is_given_time_to_start_the_next(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    job.status = AgentJobStatus.COMPLETED.value
    await _start_next(db_session, job, "approval")
    find = await _stage_of(db_session, job)
    # Completed, and the stage after it not yet created.
    await _ends(db_session, find, findings=[{"type": "papers_ingested", "title": "x"}])

    assert (await campaigns.advance(db_session, campaign))["settled"] == 0

    # Long after, a successor that never came is not waited for.
    find.completed_at = utc_now() - timedelta(hours=1)
    await db_session.commit()

    assert (await campaigns.advance(db_session, campaign))["settled"] == 1


# --------------------------------------------------------------------------
# settling
# --------------------------------------------------------------------------


async def test_the_stages_findings_and_jobs_are_the_campaigns(db_session, test_user):
    campaign = await _campaign(
        db_session, test_user, job_template={"spawn_items_from": ["algorithm_spec"]}
    )
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    await _approved_and_run(db_session, job)

    step = await campaigns.advance(db_session, campaign)
    await db_session.commit()

    assert step["settled"] == 1
    items = {i.title: i for i in await campaigns._items(db_session, campaign)}
    done = items["survey the method"]
    assert done.status == CampaignItemStatus.DONE
    assert done.outcome["finding_counts"] == {
        "note": 1,
        "papers_ingested": 1,
        "algorithm_spec": 1,
    }
    assert [s["stage"] for s in done.outcome["stages"]] == ["find", "specify"]
    assert all(s["contract_satisfied"] for s in done.outcome["stages"])
    # What a stage found becomes the next question, as a job's finding would.
    assert step["discovered"] == 1 and "fastmod u32" in items
    # One job for the question, two for its stages, one for the next question.
    assert step["action"] == "launched"
    assert campaign.jobs_launched == 4


async def test_a_stage_that_failed_is_on_the_record(db_session, test_user):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    await _approved_and_run(db_session, job, second="failed")

    await campaigns.advance(db_session, campaign)

    item = (await campaigns._items(db_session, campaign))[0]
    assert item.outcome["stages"][1] == {
        "stage": "specify",
        "job_status": "failed",
        "contract_satisfied": False,
    }


async def test_the_conclusion_is_drawn_from_what_the_stages_found(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    await _approved_and_run(db_session, job)

    findings = await campaigns._all_findings(db_session, campaign)

    assert {f["type"] for f in findings} == {
        "note",
        "papers_ingested",
        "algorithm_spec",
    }
    assert {f["subject"] for f in findings} == {"survey the method"}


async def test_inherited_stage_evidence_is_not_another_campaign_result(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    find, specify = await _approved_and_run(db_session, job)
    inherited = {
        **find.results["findings"][0],
        "inherited": True,
        "inherited_from_job_id": str(find.id),
    }
    specify.results = {
        **specify.results,
        "findings": [inherited, *specify.results["findings"]],
    }
    await db_session.commit()

    findings = await campaigns._all_findings(db_session, campaign)
    assert [f["type"] for f in findings].count("papers_ingested") == 1
    await campaigns.advance(db_session, campaign)
    done = next(
        i
        for i in await campaigns._items(db_session, campaign)
        if i.title == "survey the method"
    )
    assert done.outcome["finding_counts"]["papers_ingested"] == 1


async def test_separate_measurements_with_the_same_title_both_count(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    find, specify = await _approved_and_run(db_session, job)
    for stage in (find, specify):
        stage.results = {
            "goal_contract": {"satisfied": True},
            "findings": [
                {"type": "measurement", "title": "baseline"},
                {"type": "measurement", "title": "baseline"},
            ],
        }
    await db_session.commit()
    findings = await campaigns._all_findings(db_session, campaign)
    assert len([f for f in findings if f["type"] == "measurement"]) == 2


async def test_stages_spend_the_budget(db_session, test_user):
    campaign = await _campaign(db_session, test_user, max_jobs=3)
    job = await _launched(db_session, campaign)
    await _proposes_and_completes(db_session, job)
    await _approved_and_run(db_session, job)

    step = await campaigns.advance(db_session, campaign)

    # One question used all three jobs; the second question was never asked.
    assert campaign.jobs_launched == 3
    assert step["action"] == "exhausted"
    assert campaign.status == CampaignStatus.EXHAUSTED


# --------------------------------------------------------------------------
# proposing inside a budget
# --------------------------------------------------------------------------


async def test_a_plan_larger_than_what_the_campaign_has_left_is_refused(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user, max_jobs=2)
    job = await _launched(db_session, campaign)

    result = await planning.propose(
        job, ASK, db=db_session, llm_service=_Model(), state={}
    )

    assert "1 job(s) left" in result["error"] and "2 stages" in result["error"]
    assert not job.chain_config


async def test_a_run_outside_any_campaign_is_not_asked_about_one(db_session, test_user):
    job = AgentJob(
        name="alone",
        goal="See whether fast modular reduction applies to our kernel",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        config={},
    )
    db_session.add(job)
    await db_session.commit()

    result = await planning.propose(
        job, ASK, db=db_session, llm_service=_Model(), state={}
    )

    assert result.get("success") is True, result
