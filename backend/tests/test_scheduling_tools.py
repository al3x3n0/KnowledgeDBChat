"""`schedule_job` and `cancel_scheduled_job`, called through the real handlers.

The handlers are run against the in-memory database and judged on the
`agent_jobs` rows they leave behind: a scheduled job is only scheduled if the
row `process_scheduled_agent_jobs` scans for exists, with the owner, the
schedule and the next run it was asked for.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJob
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_scheduling_provider,
)
from app.services.auth_service import AuthService

pytestmark = pytest.mark.unit


def _handler(tool_name):
    provider = build_autonomous_scheduling_provider(SimpleNamespace())
    return provider._handlers[tool_name]


def _ctx(db, job):
    return AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(job.user_id),
        job=job,
        state={},
    )


async def _job(db, user, **overrides):
    """A stored job; by default the running job that is calling the tool."""
    fields = {
        "name": "Calling job",
        "goal": "Watch the prefetcher literature",
        "job_type": "research",
        "user_id": user.id,
        "status": "running",
    }
    fields.update(overrides)
    job = AgentJob(**fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _other_user(db):
    return await AuthService().create_user(
        username="someone_else",
        email="else@example.com",
        password="otherpassword123",
        full_name="Someone Else",
        db=db,
    )


async def _schedule(db, job, params):
    return await _handler("schedule_job")(params, _ctx(db, job))


async def _cancel(db, job, params):
    return await _handler("cancel_scheduled_job")(params, _ctx(db, job))


async def _scheduled_rows(db, caller):
    """Every job row other than the caller's own."""
    rows = (await db.execute(select(AgentJob))).scalars().all()
    return [row for row in rows if row.id != caller.id]


def _as_utc(value):
    """SQLite hands back naive datetimes; the handler stores UTC."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# schedule_job: refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, names",
    [
        ({"schedule_type": "once", "run_at": "2030-01-01T09:00:00"}, "goal"),
        (
            {"goal": "   ", "schedule_type": "once", "run_at": "2030-01-01T09:00:00"},
            "goal",
        ),
        ({"goal": "Monitor arXiv"}, "schedule_type"),
        ({"goal": "Monitor arXiv", "schedule_type": "daily"}, "schedule_type"),
        # `continuous` is a real schedule type, but not one this tool offers.
        ({"goal": "Monitor arXiv", "schedule_type": "continuous"}, "schedule_type"),
        ({"goal": "Monitor arXiv", "schedule_type": "once"}, "run_at"),
        ({"goal": "Monitor arXiv", "schedule_type": "recurring"}, "cron"),
        (
            {"goal": "Monitor arXiv", "schedule_type": "recurring", "cron": "nope"},
            "cron",
        ),
        (
            {
                "goal": "Monitor arXiv",
                "schedule_type": "recurring",
                "cron": "99 99 * * *",
            },
            "cron",
        ),
    ],
)
async def test_schedule_job_refuses_and_names_what_is_wrong(
    db_session, test_user, params, names
):
    caller = await _job(db_session, test_user)

    result = await _schedule(db_session, caller, params)

    assert "success" not in result
    assert names in result["error"]
    assert await _scheduled_rows(db_session, caller) == []


async def test_schedule_job_refuses_a_run_at_that_is_not_a_datetime(
    db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {"goal": "Monitor arXiv", "schedule_type": "once", "run_at": "next tuesday"},
    )

    assert "success" not in result
    assert "next tuesday" in result["error"]
    assert await _scheduled_rows(db_session, caller) == []


async def test_schedule_job_refuses_a_job_type_that_does_not_exist(
    db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "Monitor arXiv",
            "schedule_type": "once",
            "run_at": "2030-01-01T09:00:00+00:00",
            "job_type": "world_domination",
        },
    )

    assert "success" not in result
    # A refusal of the value, not the database rejecting the row for
    # another reason (its message quotes the whole INSERT, job_type included).
    assert not result["error"].startswith("Failed to schedule job")
    assert "job_type" in result["error"]


# ---------------------------------------------------------------------------
# schedule_job: what is stored
# ---------------------------------------------------------------------------


async def test_schedule_once_stores_a_pending_job_the_scheduler_will_find(
    db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "Summarise this week's cache papers",
            "schedule_type": "once",
            "run_at": "2030-04-01T09:00:00+00:00",
        },
    )

    assert result.get("success") is True, result
    stored = await _scheduled_rows(db_session, caller)
    assert [str(row.id) for row in stored] == [result["data"]["id"]]
    row = stored[0]
    assert row.user_id == test_user.id
    assert row.parent_job_id == caller.id
    assert row.goal == "Summarise this week's cache papers"
    assert row.name
    # What process_scheduled_agent_jobs selects on.
    assert row.status == "pending"
    assert row.schedule_type == "once"
    assert row.schedule_cron is None
    assert _as_utc(row.next_run_at) == datetime(2030, 4, 1, 9, 0, tzinfo=timezone.utc)
    assert result["data"]["schedule_type"] == "once"
    assert result["data"]["cron"] is None
    assert datetime.fromisoformat(result["data"]["next_run_at"]) == datetime(
        2030, 4, 1, 9, 0, tzinfo=timezone.utc
    )


async def test_schedule_once_reads_a_naive_datetime_as_utc(db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {"goal": "Run later", "schedule_type": "once", "run_at": "2030-04-01T09:00:00"},
    )

    assert result.get("success") is True, result
    assert result["data"]["next_run_at"] == "2030-04-01T09:00:00+00:00"


async def test_schedule_once_keeps_the_offset_it_was_given(db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "Run later",
            "schedule_type": "once",
            "run_at": "2030-04-01T09:00:00+02:00",
        },
    )

    assert result.get("success") is True, result
    assert datetime.fromisoformat(result["data"]["next_run_at"]) == datetime(
        2030, 4, 1, 7, 0, tzinfo=timezone.utc
    )


async def test_schedule_recurring_stores_the_cron_and_its_next_firing(
    db_session, test_user
):
    caller = await _job(db_session, test_user)
    before = datetime.now(timezone.utc)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "Monitor ML papers",
            "job_type": "monitor",
            "schedule_type": "recurring",
            "cron": "0 9 * * 1",
        },
    )

    assert result.get("success") is True, result
    (row,) = await _scheduled_rows(db_session, caller)
    assert row.user_id == test_user.id
    assert row.status == "pending"
    assert row.job_type == "monitor"
    assert row.schedule_type == "recurring"
    assert row.schedule_cron == "0 9 * * 1"
    next_run = _as_utc(row.next_run_at)
    assert before < next_run <= before + timedelta(days=7, minutes=1)
    assert (next_run.weekday(), next_run.hour, next_run.minute) == (0, 9, 0)
    assert result["data"]["cron"] == "0 9 * * 1"
    assert result["data"]["job_type"] == "monitor"


async def test_schedule_job_defaults_to_research_and_an_empty_config(
    db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "Run later",
            "schedule_type": "once",
            "run_at": "2030-04-01T09:00:00+00:00",
            "config": "not a mapping",
        },
    )

    assert result.get("success") is True, result
    (row,) = await _scheduled_rows(db_session, caller)
    assert row.job_type == "research"
    assert row.config == {}


async def test_schedule_job_keeps_the_config_it_was_given(db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "Run later",
            "schedule_type": "once",
            "run_at": "2030-04-01T09:00:00+00:00",
            "config": {"max_iterations": 50},
        },
    )

    assert result.get("success") is True, result
    (row,) = await _scheduled_rows(db_session, caller)
    assert row.config == {"max_iterations": 50}


async def test_schedule_job_bounds_the_goal_it_stores(db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _schedule(
        db_session,
        caller,
        {
            "goal": "G" * 3000,
            "schedule_type": "once",
            "run_at": "2030-04-01T09:00:00+00:00",
        },
    )

    assert result.get("success") is True, result
    (row,) = await _scheduled_rows(db_session, caller)
    assert row.goal == "G" * 2000
    assert len(row.name) <= 200


async def test_schedule_job_is_owned_by_the_calling_jobs_owner_not_the_context_user(
    db_session, test_user
):
    """The new job belongs to whoever owns the job that asked for it."""
    other = await _other_user(db_session)
    caller = await _job(db_session, test_user)
    ctx = _ctx(db_session, caller)
    ctx.user_id = str(other.id)

    result = await _handler("schedule_job")(
        {
            "goal": "Run later",
            "schedule_type": "once",
            "run_at": "2030-04-01T09:00:00+00:00",
        },
        ctx,
    )

    assert result.get("success") is True, result
    (row,) = await _scheduled_rows(db_session, caller)
    assert row.user_id == test_user.id


async def test_a_refused_schedule_leaves_the_session_usable(db_session, test_user):
    """A tool that returns `{"error": ...}` must not break the run's session.

    The insert is made to fail the way a constraint fails: a job with no
    owner. The session is the run's, and every later tool shares it.
    """
    caller = await _job(db_session, test_user)
    caller_id = caller.id
    ownerless = SimpleNamespace(id=caller_id, user_id=None, name="ownerless")

    result = await _schedule(
        db_session,
        ownerless,
        {"goal": "Run later", "schedule_type": "once", "run_at": "2030-04-01T09:00:00"},
    )

    assert result["error"].startswith("Failed to schedule job")
    rows = (await db_session.execute(select(AgentJob.id))).scalars().all()
    assert rows == [caller_id]


async def test_one_run_cannot_schedule_an_unbounded_number_of_jobs(
    db_session, test_user
):
    caller = await _job(db_session, test_user)
    ctx = _ctx(db_session, caller)
    handler = _handler("schedule_job")

    results = []
    for n in range(60):
        params = {
            "goal": f"Recurring job {n}",
            "schedule_type": "recurring",
            "cron": "*/5 * * * *",
        }
        results.append(await handler(params, ctx))
        if n == 0 and results[0].get("success") is not True:
            pytest.fail(f"cannot judge a cap while nothing schedules: {results[0]}")

    assert results[0].get("success") is True, results[0]
    assert any("error" in result for result in results)


# ---------------------------------------------------------------------------
# cancel_scheduled_job
# ---------------------------------------------------------------------------


async def test_cancel_requires_a_job_id(db_session, test_user):
    caller = await _job(db_session, test_user)

    for params in ({}, {"job_id": "  "}):
        result = await _cancel(db_session, caller, params)
        assert "success" not in result
        assert "job_id is required" in result["error"]


async def test_cancel_refuses_an_id_that_is_not_a_uuid(db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _cancel(db_session, caller, {"job_id": "not-a-uuid"})

    assert "success" not in result
    assert "Invalid job_id" in result["error"]
    assert "not-a-uuid" in result["error"]


async def test_cancel_reports_a_job_that_does_not_exist(db_session, test_user):
    caller = await _job(db_session, test_user)
    missing = str(uuid4())

    result = await _cancel(db_session, caller, {"job_id": missing})

    assert "success" not in result
    assert "not found" in result["error"].lower()
    assert missing in result["error"]


async def test_cancel_stops_a_one_time_job_from_ever_running(db_session, test_user):
    caller = await _job(db_session, test_user)
    target = await _job(
        db_session,
        test_user,
        name="Later",
        goal="Run once later",
        status="pending",
        schedule_type="once",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )

    result = await _cancel(db_session, caller, {"job_id": str(target.id)})

    assert result == {
        "success": True,
        "data": {"id": str(target.id), "status": "cancelled", "goal": "Run once later"},
    }
    await db_session.commit()
    await db_session.refresh(target)
    # Out of every set process_scheduled_agent_jobs selects from.
    assert target.status == "cancelled"
    assert target.next_run_at is None


async def test_cancel_stops_a_recurring_job_from_firing_again(db_session, test_user):
    caller = await _job(db_session, test_user)
    # A recurring job between firings rests in `completed` with a next run.
    target = await _job(
        db_session,
        test_user,
        name="Weekly",
        status="completed",
        schedule_type="recurring",
        schedule_cron="0 9 * * 1",
        next_run_at=datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc),
    )

    result = await _cancel(db_session, caller, {"job_id": str(target.id)})

    assert result.get("success") is True, result
    await db_session.commit()
    await db_session.refresh(target)
    assert target.status == "cancelled"
    assert target.next_run_at is None
    assert target.schedule_type is None


async def test_cancel_accepts_surrounding_whitespace_in_the_id(db_session, test_user):
    caller = await _job(db_session, test_user)
    target = await _job(
        db_session,
        test_user,
        status="pending",
        schedule_type="once",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )

    result = await _cancel(db_session, caller, {"job_id": f"  {target.id}\n"})

    assert result.get("success") is True, result


async def test_cancel_refuses_another_users_job_and_leaves_it_scheduled(
    db_session, test_user
):
    other = await _other_user(db_session)
    caller = await _job(db_session, test_user)
    theirs = await _job(
        db_session,
        other,
        name="Theirs",
        goal="Someone else's private goal",
        status="pending",
        schedule_type="recurring",
        schedule_cron="0 9 * * 1",
        next_run_at=datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc),
    )

    result = await _cancel(db_session, caller, {"job_id": str(theirs.id)})

    assert "success" not in result
    assert "error" in result
    # Nothing about the other user's job comes back with the refusal.
    assert "Someone else's private goal" not in str(result)
    await db_session.commit()
    await db_session.refresh(theirs)
    assert theirs.status == "pending"
    assert theirs.schedule_type == "recurring"
    assert theirs.schedule_cron == "0 9 * * 1"
    assert theirs.next_run_at is not None


async def test_cancel_is_scoped_to_the_calling_jobs_owner_not_the_context_user(
    db_session, test_user
):
    """A context naming another user does not lend that user's authority."""
    other = await _other_user(db_session)
    caller = await _job(db_session, test_user)
    theirs = await _job(
        db_session,
        other,
        status="pending",
        schedule_type="once",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    ctx = _ctx(db_session, caller)
    ctx.user_id = str(other.id)

    result = await _handler("cancel_scheduled_job")({"job_id": str(theirs.id)}, ctx)

    assert "success" not in result
    await db_session.commit()
    await db_session.refresh(theirs)
    assert theirs.status == "pending"


async def test_cancel_refuses_a_job_that_is_running(db_session, test_user):
    caller = await _job(db_session, test_user)
    target = await _job(
        db_session,
        test_user,
        name="Busy",
        status="running",
        schedule_type="recurring",
        schedule_cron="0 9 * * 1",
        next_run_at=datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc),
    )

    result = await _cancel(db_session, caller, {"job_id": str(target.id)})

    assert "success" not in result
    assert "running" in result["error"]
    await db_session.commit()
    await db_session.refresh(target)
    assert target.status == "running"
    assert target.schedule_type == "recurring"
    assert target.next_run_at is not None


async def test_cancel_refuses_a_job_that_was_never_scheduled(db_session, test_user):
    """The tool cancels schedules; a finished one-shot run is not one."""
    caller = await _job(db_session, test_user)
    finished = await _job(
        db_session,
        test_user,
        name="Done",
        status="completed",
        schedule_type=None,
        next_run_at=None,
    )

    result = await _cancel(db_session, caller, {"job_id": str(finished.id)})

    await db_session.commit()
    await db_session.refresh(finished)
    assert finished.status == "completed"
    assert "success" not in result


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


class TestSchedulingToolSchemas:
    """Tests for scheduling tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "schedule_job" in names
        assert "cancel_scheduled_job" in names

    def test_schedule_job_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("schedule_job")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "goal" in required
        assert "schedule_type" in required

    def test_cancel_scheduled_job_requires_job_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("cancel_scheduled_job")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "job_id" in required

    def test_schedule_job_has_cron_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("schedule_job")
        assert "cron" in tool["parameters"]["properties"]

    def test_schedule_job_has_run_at_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("schedule_job")
        assert "run_at" in tool["parameters"]["properties"]

    def test_every_declared_parameter_is_one_the_handler_reads(self):
        """A parameter the model is offered and the handler ignores is a lie."""
        import inspect

        from app.services import agent_tool_dispatch
        from app.services.agent_tools import get_tool_by_name

        source = inspect.getsource(
            agent_tool_dispatch.build_autonomous_scheduling_provider
        )
        for tool_name in ("schedule_job", "cancel_scheduled_job"):
            for param in get_tool_by_name(tool_name)["parameters"]["properties"]:
                assert f'params.get("{param}"' in source, (tool_name, param)


class TestSchedulingToolRegistry:
    """Tests for scheduling tool registry classification."""

    def test_schedule_job_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("schedule_job")
        assert meta is not None
        assert meta.effects == "write"

    def test_cancel_scheduled_job_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("cancel_scheduled_job")
        assert meta is not None
        assert meta.effects == "write"

    def test_scheduling_tools_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["schedule_job", "cancel_scheduled_job"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
