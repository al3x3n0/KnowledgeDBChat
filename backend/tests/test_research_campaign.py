"""A line of enquiry that survives the process running it.

Job chaining fires children at completion and then nobody is watching, and a
restart ends the sequence silently. A campaign holds the sequence instead, and
is advanced a step at a time so that all the state that matters lives in the
database rather than in a running program.

Idempotence is the property under test throughout: calling a step twice must
not do the work twice, because the caller is a scheduler and schedulers repeat.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.models.agent_job import AgentJob
from app.models.research_campaign import CampaignItemStatus, CampaignStatus
from app.services import research_campaign_service as campaigns


async def _campaign(db, **overrides):
    kwargs = dict(
        user_id=uuid4(),
        name="fusion study",
        goal="find an instruction worth proposing",
        items=[{"title": "profile the kernel"}, {"title": "cost the top candidate"}],
        max_jobs=5,
    )
    kwargs.update(overrides)
    campaign = await campaigns.create_campaign(db, **kwargs)
    await db.commit()
    return campaign


async def _finish(db, job_id, *, status="completed", results=None):
    # advance() reports ids as strings; the column is a UUID.
    job = await db.get(AgentJob, UUID(str(job_id)))
    job.status = status
    job.results = (
        results if results is not None else {"goal_contract": {"satisfied": True}}
    )
    await db.commit()
    return job


@pytest.mark.asyncio
async def test_a_campaign_needs_a_goal(db_session):
    with pytest.raises(ValueError) as excinfo:
        await campaigns.create_campaign(
            db_session, user_id=uuid4(), name="nameless", goal="  "
        )

    assert "goal" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_step_launches_one_item(db_session):
    campaign = await _campaign(db_session)

    step = await campaigns.advance(db_session, campaign)

    assert step["action"] == "launched"
    assert step["launched_job"]
    assert step["running"] == 1
    assert step["pending"] == 1, "the other item waits its turn"


@pytest.mark.asyncio
async def test_a_second_step_does_not_relaunch_the_same_item(db_session):
    """The caller is a scheduler, and schedulers repeat."""
    campaign = await _campaign(db_session)
    await campaigns.advance(db_session, campaign)
    await db_session.commit()

    second = await campaigns.advance(db_session, campaign)

    assert second["launched_job"] is None
    assert second["action"] == "waiting"
    assert campaign.jobs_launched == 1


@pytest.mark.asyncio
async def test_a_finished_job_settles_its_item_and_frees_the_next(db_session):
    campaign = await _campaign(db_session)
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, launched["launched_job"])

    step = await campaigns.advance(db_session, campaign)

    assert step["settled"] == 1
    assert step["action"] == "launched", "the next item starts once the first settles"
    items = await campaigns._items(db_session, campaign)
    assert any(i.status == CampaignItemStatus.DONE for i in items)


@pytest.mark.asyncio
async def test_a_failed_job_marks_its_item_failed_without_stopping_the_campaign(
    db_session,
):
    campaign = await _campaign(db_session)
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, launched["launched_job"], status="failed")

    step = await campaigns.advance(db_session, campaign)

    items = await campaigns._items(db_session, campaign)
    assert any(i.status == CampaignItemStatus.FAILED for i in items)
    assert step["action"] == "launched", "one failure does not end the enquiry"


@pytest.mark.asyncio
async def test_a_campaign_completes_when_the_work_runs_out(db_session):
    campaign = await _campaign(db_session, items=[{"title": "only item"}])
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, launched["launched_job"])

    step = await campaigns.advance(db_session, campaign)

    assert step["action"] == "completed"
    assert campaign.status == CampaignStatus.COMPLETED
    assert campaign.completed_at is not None


@pytest.mark.asyncio
async def test_running_out_of_budget_is_not_the_same_as_finishing(db_session):
    """A campaign that stopped early looks exactly like one that finished
    unless someone writes down which it was."""
    campaign = await _campaign(
        db_session, items=[{"title": "a"}, {"title": "b"}], max_jobs=1
    )
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, launched["launched_job"])

    step = await campaigns.advance(db_session, campaign)

    assert step["action"] == "exhausted"
    assert campaign.status == CampaignStatus.EXHAUSTED


@pytest.mark.asyncio
async def test_findings_can_become_further_work(db_session):
    campaign = await _campaign(
        db_session,
        items=[{"title": "profile the kernel"}],
        job_template={"spawn_items_from": ["fusion_candidate"]},
    )
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        launched["launched_job"],
        results={
            "goal_contract": {"satisfied": True},
            "findings": [
                {"type": "fusion_candidate", "title": "fcmgt.2s + bit.8b"},
                {"type": "fusion_candidate", "title": "sxtl.8h + smlal.4s"},
                {"type": "document", "title": "an unrelated document"},
            ],
        },
    )

    step = await campaigns.advance(db_session, campaign)

    assert step["discovered"] == 2
    items = await campaigns._items(db_session, campaign)
    discovered = [i for i in items if i.origin == "discovered"]
    assert {i.title for i in discovered} == {"fcmgt.2s + bit.8b", "sxtl.8h + smlal.4s"}
    assert all(i.origin == "seed" or i.origin == "discovered" for i in items)


@pytest.mark.asyncio
async def test_findings_of_other_types_do_not_become_work(db_session):
    campaign = await _campaign(
        db_session,
        items=[{"title": "one"}],
        job_template={"spawn_items_from": ["fusion_candidate"]},
    )
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        launched["launched_job"],
        results={"findings": [{"type": "document", "title": "a document"}]},
    )

    step = await campaigns.advance(db_session, campaign)

    assert step["discovered"] == 0


@pytest.mark.asyncio
async def test_the_same_finding_twice_makes_one_item(db_session):
    campaign = await _campaign(
        db_session,
        items=[{"title": "one"}],
        job_template={"spawn_items_from": ["fusion_candidate"]},
    )
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        launched["launched_job"],
        results={
            "findings": [
                {"type": "fusion_candidate", "title": "same shape"},
                {"type": "fusion_candidate", "title": "same shape"},
            ]
        },
    )

    step = await campaigns.advance(db_session, campaign)

    assert step["discovered"] == 1


@pytest.mark.asyncio
async def test_a_job_that_vanished_does_not_strand_its_item(db_session):
    campaign = await _campaign(db_session, items=[{"title": "one"}])
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    job = await db_session.get(AgentJob, UUID(launched["launched_job"]))
    await db_session.delete(job)
    await db_session.commit()

    await campaigns.advance(db_session, campaign)

    items = await campaigns._items(db_session, campaign)
    assert items[0].status == CampaignItemStatus.FAILED
    assert "no longer exists" in items[0].outcome["error"]


@pytest.mark.asyncio
async def test_the_job_carries_the_campaign_goal_and_its_own_part(db_session):
    campaign = await _campaign(
        db_session, items=[{"title": "cost it", "detail": "at n1"}]
    )
    launched = await campaigns.advance(db_session, campaign)

    job = await db_session.get(AgentJob, UUID(launched["launched_job"]))

    assert campaign.goal in job.goal
    assert "cost it" in job.goal
    assert "at n1" in job.goal
    assert job.config["campaign_id"] == str(campaign.id)


@pytest.mark.asyncio
async def test_a_finished_campaign_is_left_alone(db_session):
    campaign = await _campaign(db_session, items=[{"title": "one"}])
    campaign.status = CampaignStatus.CANCELLED
    await db_session.commit()

    step = await campaigns.advance(db_session, campaign)

    assert step["action"] == "none"
    assert campaign.jobs_launched == 0


@pytest.mark.asyncio
async def test_a_summary_says_what_the_campaign_has_done(db_session):
    campaign = await _campaign(db_session)
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, launched["launched_job"])
    await campaigns.advance(db_session, campaign)

    summary = await campaigns.summarize(db_session, campaign)

    assert summary["items"] == 2
    assert summary["by_status"][CampaignItemStatus.DONE] == 1
    assert summary["items_meeting_contract"] == 1
    assert summary["budget"] == 5


@pytest.mark.asyncio
async def test_advancing_all_skips_campaigns_that_are_not_active(db_session):
    active = await _campaign(db_session, name="active one")
    done = await _campaign(db_session, name="finished one")
    done.status = CampaignStatus.COMPLETED
    await db_session.commit()

    results = await campaigns.advance_all(db_session)

    advanced = {row["campaign"] for row in results}
    assert str(active.id) in advanced
    assert str(done.id) not in advanced


def test_the_scheduler_is_told_to_advance_campaigns():
    """A mechanism nothing calls on a timer is not an unattended programme."""
    from app.core.celery import celery_app

    schedule = celery_app.conf.beat_schedule
    entry = schedule.get("advance-research-campaigns")

    assert entry, "campaigns must be advanced on a timer to run unattended"
    assert entry["task"] == "app.tasks.agent_job_tasks.advance_research_campaigns"


@pytest.mark.asyncio
async def test_a_tick_launches_the_job_and_leaves_it_ready_to_run(db_session):
    """The campaign records the job; a worker still has to be told about it."""
    campaign = await _campaign(db_session, items=[{"title": "only item"}])

    steps = await campaigns.advance_all(db_session)
    await db_session.commit()

    launched = [s for s in steps if s.get("action") == "launched"]
    assert len(launched) == 1
    job = await db_session.get(AgentJob, UUID(launched[0]["launched_job"]))
    assert job.status == "pending", "a launched job waits for a worker"
    assert job.user_id == campaign.user_id


@pytest.mark.asyncio
async def test_a_tick_over_many_campaigns_launches_one_job_each(db_session):
    """Cost per tick is bounded by how many campaigns are active, not by how
    much work is on their backlogs."""
    for index in range(3):
        await _campaign(
            db_session,
            name=f"campaign {index}",
            items=[{"title": "a"}, {"title": "b"}, {"title": "c"}],
        )

    steps = await campaigns.advance_all(db_session)

    launched = [s for s in steps if s.get("action") == "launched"]
    assert len(launched) == 3, "one job per campaign, not one per item"


# --- judgement: which item to do next, and which line to stop -----------------


@pytest.mark.asyncio
async def test_discovered_work_records_where_it_came_from(db_session):
    """Without lineage a cold line cannot be told from a cold item."""
    campaign = await _campaign(
        db_session,
        items=[{"title": "profile the kernel"}],
        job_template={"spawn_items_from": ["fusion_candidate"]},
    )
    launched = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        launched["launched_job"],
        results={"findings": [{"type": "fusion_candidate", "title": "fcmgt + bit"}]},
    )

    await campaigns.advance(db_session, campaign)

    items = await campaigns._items(db_session, campaign)
    parent = next(i for i in items if i.origin == "seed")
    child = next(i for i in items if i.origin == "discovered")
    assert child.parent_item_id == parent.id
    assert child.generation == 1
    assert parent.generation == 0


@pytest.mark.asyncio
async def test_the_campaign_prefers_work_from_a_job_that_produced_something(db_session):
    """The point of judgement: not the oldest item, the better one."""
    campaign = await _campaign(
        db_session,
        items=[{"title": "a stale seed"}, {"title": "profile the kernel"}],
        job_template={"spawn_items_from": ["fusion_candidate"]},
        max_jobs=9,
    )
    # Run the second seed, not the first, by finishing whatever launches and
    # letting the discovered item compete with the remaining seed.
    first = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        first["launched_job"],
        results={
            "goal_contract": {"satisfied": True},
            "findings": [{"type": "fusion_candidate", "title": "a real candidate"}],
        },
    )

    step = await campaigns.advance(db_session, campaign)

    assert step["chose"]["title"] == "a real candidate", step["chose"]
    assert "met its contract" in step["chose"]["why"]


@pytest.mark.asyncio
async def test_a_line_that_twice_produced_nothing_is_abandoned(db_session):
    campaign = await _campaign(
        db_session,
        items=[{"title": "root"}],
        job_template={
            "spawn_items_from": ["fusion_candidate"],
            "target_finding_types": ["codegen_measurement"],
        },
        max_jobs=9,
    )
    barren = {
        "goal_contract": {"satisfied": False},
        "findings": [{"type": "fusion_candidate", "title": "offshoot"}],
    }
    step = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, step["launched_job"], results=barren)

    # The offshoot runs and is equally barren, spawning one of its own.
    step = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        step["launched_job"],
        results={
            "goal_contract": {"satisfied": False},
            "findings": [{"type": "fusion_candidate", "title": "grand offshoot"}],
        },
    )

    step = await campaigns.advance(db_session, campaign)

    items = await campaigns._items(db_session, campaign)
    dropped = [i for i in items if i.status == CampaignItemStatus.DROPPED]
    assert [i.title for i in dropped] == ["grand offshoot"]
    assert "line abandoned" in dropped[0].priority_reason


@pytest.mark.asyncio
async def test_a_campaign_that_never_said_what_it_wanted_abandons_nothing(db_session):
    """Undeclared targets is how a campaign opts out of giving up."""
    campaign = await _campaign(
        db_session,
        items=[{"title": "root"}],
        job_template={"spawn_items_from": ["fusion_candidate"]},
        max_jobs=9,
    )
    barren = {
        "goal_contract": {"satisfied": False},
        "findings": [{"type": "fusion_candidate", "title": "offshoot"}],
    }
    for _ in range(2):
        step = await campaigns.advance(db_session, campaign)
        await db_session.commit()
        await _finish(db_session, step["launched_job"], results=barren)
    await campaigns.advance(db_session, campaign)

    items = await campaigns._items(db_session, campaign)
    assert not [i for i in items if i.status == CampaignItemStatus.DROPPED]


@pytest.mark.asyncio
async def test_a_seed_is_never_abandoned_however_bad_the_record(db_session):
    """A cold line may be cold because the questions were hard."""
    campaign = await _campaign(
        db_session,
        items=[{"title": "seed one"}, {"title": "seed two"}],
        job_template={"target_finding_types": ["codegen_measurement"]},
        max_jobs=9,
    )
    step = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        step["launched_job"],
        results={"goal_contract": {"satisfied": False}},
    )

    await campaigns.advance(db_session, campaign)

    items = await campaigns._items(db_session, campaign)
    assert not [i for i in items if i.status == CampaignItemStatus.DROPPED]


@pytest.mark.asyncio
async def test_dropping_the_last_work_completes_rather_than_hangs(db_session):
    campaign = await _campaign(
        db_session,
        items=[{"title": "root"}],
        job_template={
            "spawn_items_from": ["fusion_candidate"],
            "target_finding_types": ["codegen_measurement"],
        },
        max_jobs=9,
    )
    barren = {
        "goal_contract": {"satisfied": False},
        "findings": [{"type": "fusion_candidate", "title": "offshoot"}],
    }
    step = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(db_session, step["launched_job"], results=barren)
    step = await campaigns.advance(db_session, campaign)
    await db_session.commit()
    await _finish(
        db_session,
        step["launched_job"],
        results={"goal_contract": {"satisfied": False}, "findings": []},
    )

    step = await campaigns.advance(db_session, campaign)

    assert step["action"] == "completed"


@pytest.mark.asyncio
async def test_the_launch_records_what_it_thought_and_why(db_session):
    campaign = await _campaign(db_session, items=[{"title": "only item"}])

    await campaigns.advance(db_session, campaign)

    item = (await campaigns._items(db_session, campaign))[0]
    assert item.priority is not None
    assert item.priority_reason
    assert item.launched_at is not None


@pytest.mark.asyncio
async def test_a_summary_shows_what_it_would_do_next_and_why(db_session):
    """An operator should be able to disagree before a job is spent."""
    campaign = await _campaign(db_session)

    summary = await campaigns.summarize(db_session, campaign)

    assert len(summary["next_up"]) == 2
    assert all(
        row["why"] is None or isinstance(row["why"], str) for row in summary["next_up"]
    )
