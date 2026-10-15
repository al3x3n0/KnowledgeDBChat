"""A campaign approves a plan its own job proposed, within limits it was given.

A campaign stopped for a person at every plan, so one left to run alone
stalled at its first proposal. With ``plan_approval`` it approves a plan of
no more stages and no longer an estimate than its limits allow, through the
same decision a person's approval goes through, and records that it did.

The model is scripted. The proposal, the approval hold, the decision, the
chain runtime that starts the first stage, and the campaign's step are real.
"""

import copy
from unittest.mock import patch
from uuid import UUID

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.models.agent_job import AgentJob, AgentJobStatus
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
ASK = "Find the paper on fast modular reduction and write its algorithm down."
GENEROUS = {"max_stages": 4, "max_minutes": 120}


class _Model:
    async def generate_structured(self, **kwargs):
        return copy.deepcopy(SPEC)


async def _campaign(db, user, **overrides):
    kwargs = dict(
        user_id=user.id,
        name="fastmod",
        goal="find out whether fast modular reduction is worth adopting",
        items=[{"title": "survey the method"}],
        max_jobs=5,
    )
    kwargs.update(overrides)
    campaign = await campaigns.create_campaign(db, **kwargs)
    await db.commit()
    return campaign


async def _held_with_a_plan(db, campaign):
    """The campaign's job proposes a plan, completes, and is held."""
    step = await campaigns.advance(db, campaign)
    await db.commit()
    job = await db.get(AgentJob, UUID(step["launched_job"]))
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
    return job


async def _step(db, campaign):
    """One campaign tick, as the scheduled task runs it."""
    from app.modules.autonomy.application import plan_approval

    with patch("app.tasks.agent_job_tasks.execute_agent_job_task.delay"):
        step = await campaigns.advance(
            db, campaign, approve=plan_approval.approve_proposed_plan
        )
        await db.commit()
    return step


async def _stages(db, job):
    return list(
        (await db.execute(select(AgentJob).where(AgentJob.parent_job_id == job.id)))
        .scalars()
        .all()
    )


class TestWithinItsLimits:
    async def test_it_approves_and_the_first_stage_starts(self, db_session, test_user):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        job = await _held_with_a_plan(db_session, campaign)

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert decision["approved"] and decision["stages"] == 2
        await db_session.refresh(job)
        assert job.status == AgentJobStatus.COMPLETED.value
        (first,) = await _stages(db_session, job)
        assert first.config["pipeline_stage"] == "find"
        assert step["waiting_on"] == "job"

    async def test_the_approval_is_recorded_as_the_campaign_s(
        self, db_session, test_user
    ):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        job = await _held_with_a_plan(db_session, campaign)

        await _step(db_session, campaign)

        await db_session.refresh(job)
        log = [e for e in job.execution_log if e.get("phase") == "chain_gate_decision"]
        assert log and "Approved by campaign 'fastmod'" in log[-1]["note"]
        # Where a person's approval is recorded too.
        approvals = (
            (job.results.get("execution_strategy") or {}).get("approval_checkpoints")
            or {}
        ).get("approvals") or []
        assert approvals and approvals[-1]["approved_by"] == str(test_user.id)
        assert "Approved by campaign" in approvals[-1]["note"]

    async def test_the_item_says_who_approved_what(self, db_session, test_user):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        await _held_with_a_plan(db_session, campaign)

        await _step(db_session, campaign)

        summary = await campaigns.summarize(db_session, campaign)
        assert summary["plan_approval"] == GENEROUS
        assert summary["plan_held"] is None


class TestOutsideThem:
    async def test_without_limits_a_person_approves_as_before(
        self, db_session, test_user
    ):
        campaign = await _campaign(db_session, test_user)
        job = await _held_with_a_plan(db_session, campaign)

        step = await _step(db_session, campaign)

        assert step["plan_decisions"] == [] and step["waiting_on"] == "approval"
        await db_session.refresh(job)
        assert job.status == AgentJobStatus.PAUSED.value

    async def test_too_many_stages_waits_for_a_person_and_says_why(
        self, db_session, test_user
    ):
        campaign = await _campaign(
            db_session, test_user, plan_approval={"max_stages": 1, "max_minutes": 120}
        )
        job = await _held_with_a_plan(db_session, campaign)

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert not decision["approved"]
        assert "2 stages" in decision["reason"] and "at most 1" in decision["reason"]
        await db_session.refresh(job)
        assert job.status == AgentJobStatus.PAUSED.value
        summary = await campaigns.summarize(db_session, campaign)
        assert "at most 1" in summary["plan_held"]

    async def test_too_long_an_estimate_waits_too(self, db_session, test_user):
        campaign = await _campaign(
            db_session, test_user, plan_approval={"max_stages": 4, "max_minutes": 1}
        )
        await _held_with_a_plan(db_session, campaign)

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert not decision["approved"] and "min" in decision["reason"]

    async def test_a_plan_bigger_than_the_budget_left_waits(
        self, db_session, test_user
    ):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        await _held_with_a_plan(db_session, campaign)
        # Jobs spent elsewhere since the plan was proposed.
        campaign.jobs_launched = int(campaign.max_jobs) - 1
        await db_session.commit()

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert not decision["approved"] and "job(s) left" in decision["reason"]

    async def test_an_approval_that_is_not_a_proposed_plan_is_a_person_s(
        self, db_session, test_user
    ):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        step = await campaigns.advance(db_session, campaign)
        await db_session.commit()
        job = await db_session.get(AgentJob, UUID(step["launched_job"]))
        # A tool asked for approval mid-run.
        job.status = AgentJobStatus.PAUSED.value
        job.current_phase = "awaiting_approval"
        job.results = {
            "approval_checkpoint": {
                "checkpoint_type": "action_approval",
                "action": {"tool": "delete_document"},
            }
        }
        await db_session.commit()

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert not decision["approved"]
        await db_session.refresh(job)
        assert job.status == AgentJobStatus.PAUSED.value

    async def test_a_deactivated_owner_approves_nothing(self, db_session, test_user):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        job = await _held_with_a_plan(db_session, campaign)
        test_user.is_active = False
        await db_session.commit()

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert not decision["approved"]
        await db_session.refresh(job)
        assert job.status == AgentJobStatus.PAUSED.value


class TestTheLimitsThemselves:
    async def test_a_limit_over_the_ceiling_is_refused_with_the_figure(
        self, db_session, test_user, monkeypatch
    ):
        monkeypatch.setattr(settings, "CAMPAIGN_PLAN_APPROVAL_MAX_STAGES", 4)

        with pytest.raises(ValueError, match="at most 4"):
            await _campaign(
                db_session,
                test_user,
                plan_approval={"max_stages": 9, "max_minutes": 60},
            )

    async def test_a_ceiling_lowered_later_still_binds(
        self, db_session, test_user, monkeypatch
    ):
        campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
        job = await _held_with_a_plan(db_session, campaign)
        monkeypatch.setattr(settings, "CAMPAIGN_PLAN_APPROVAL_MAX_STAGES", 1)

        step = await _step(db_session, campaign)

        (decision,) = step["plan_decisions"]
        assert not decision["approved"] and "at most 1" in decision["reason"]
        await db_session.refresh(job)
        assert job.status == AgentJobStatus.PAUSED.value

    def test_no_limits_means_none(self):
        assert campaigns.checked_plan_approval(None) is None
        assert campaigns.checked_plan_approval({}) is None
        with pytest.raises(ValueError):
            campaigns.checked_plan_approval({"max_stages": 0, "max_minutes": 5})


class TestTheRoute:
    def test_a_campaign_can_be_given_limits(self, client, auth_headers):
        response = client.post(
            "/api/v1/research/campaigns",
            headers=auth_headers,
            json={
                "name": "c",
                "goal": "g",
                "plan_approval": {"max_stages": 2, "max_minutes": 30},
            },
        )

        assert response.status_code == 201, response.text
        assert response.json()["plan_approval"] == {"max_stages": 2, "max_minutes": 30}

    def test_limits_over_the_ceiling_are_a_400(self, client, auth_headers):
        response = client.post(
            "/api/v1/research/campaigns",
            headers=auth_headers,
            json={
                "name": "c",
                "goal": "g",
                "plan_approval": {"max_stages": 40, "max_minutes": 30},
            },
        )

        assert response.status_code == 400
        assert "at most" in response.json()["detail"]


def test_the_scheduled_tick_passes_the_approver():
    """Without it a campaign's limits would be stored and never used."""
    import inspect

    from app.tasks import agent_job_tasks

    source = inspect.getsource(agent_job_tasks)
    assert "approve=plan_approval.approve_proposed_plan" in source


async def test_a_tick_with_no_approver_leaves_the_plan_for_a_person(
    db_session, test_user
):
    campaign = await _campaign(db_session, test_user, plan_approval=GENEROUS)
    job = await _held_with_a_plan(db_session, campaign)

    with patch("app.tasks.agent_job_tasks.execute_agent_job_task.delay"):
        step = await campaigns.advance(db_session, campaign)

    assert step["plan_decisions"] == [] and step["waiting_on"] == "approval"
    await db_session.refresh(job)
    assert job.status == AgentJobStatus.PAUSED.value
