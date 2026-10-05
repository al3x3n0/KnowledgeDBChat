"""`send_notification` and `send_email_alert`, called through the real handlers.

Both are judged on the `notifications` row they leave behind and on what
reaches the Redis publisher, which is the only edge replaced here.
"""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJob
from app.models.notification import Notification
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_notification_visualization_provider,
)
from app.services.auth_service import AuthService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def published(monkeypatch):
    """Stand in for Redis: record what would have been pushed to a socket."""
    sent = []
    monkeypatch.setattr(
        "app.tasks.job_support.publish_sync",
        lambda channel, message: sent.append((channel, message)),
    )
    return sent


def _handler(tool_name):
    provider = build_autonomous_notification_visualization_provider(SimpleNamespace())
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
    fields = {
        "name": "Research Agent",
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


async def _notify(db, job, params):
    return await _handler("send_notification")(params, _ctx(db, job))


async def _email(db, job, params):
    return await _handler("send_email_alert")(params, _ctx(db, job))


async def _stored(db):
    return (await db.execute(select(Notification))).scalars().all()


def _fail_first_flush(monkeypatch, db):
    """Make the write of the notification fail once, as a database would."""
    real_flush = db.flush
    calls = {"n": 0}

    async def flush(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is unavailable")
        return await real_flush(*args, **kwargs)

    monkeypatch.setattr(db, "flush", flush)


# ---------------------------------------------------------------------------
# send_notification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, names",
    [
        ({"message": "something happened"}, "title"),
        ({"title": "   ", "message": "something happened"}, "title"),
        ({"title": "Alert"}, "message"),
        ({"title": "Alert", "message": "\n \t"}, "message"),
    ],
)
async def test_send_notification_refuses_and_names_what_is_missing(
    db_session, test_user, published, params, names
):
    job = await _job(db_session, test_user)

    result = await _notify(db_session, job, params)

    assert "success" not in result
    assert result["error"] == f"{names} is required"
    assert await _stored(db_session) == []
    assert published == []


async def test_send_notification_stores_a_notification_for_the_job_owner(
    db_session, test_user
):
    job = await _job(db_session, test_user)

    result = await _notify(
        db_session,
        job,
        {
            "title": "  Job Complete  ",
            "message": "Research finished with 10 papers found",
            "priority": "high",
            "action_url": "/autonomous-agents?job=1",
        },
    )

    assert result.get("success") is True, result
    assert result["data"]["delivered"] is True
    assert result["data"]["priority"] == "high"
    (row,) = await _stored(db_session)
    assert str(row.id) == result["data"]["notification_id"]
    assert row.user_id == test_user.id
    assert row.notification_type == "agent_job_alert"
    assert row.title == "Job Complete"
    assert row.message == "Research finished with 10 papers found"
    assert row.priority == "high"
    assert row.action_url == "/autonomous-agents?job=1"
    assert row.related_entity_type == "agent_job"
    assert row.related_entity_id == job.id
    assert row.data == {
        "source_job_id": str(job.id),
        "source_job_name": "Research Agent",
    }
    assert row.is_read is False
    assert row.is_dismissed is False


async def test_send_notification_pushes_to_the_owners_channel(
    db_session, test_user, published
):
    job = await _job(db_session, test_user)

    result = await _notify(
        db_session, job, {"title": "Job Complete", "message": "Done", "priority": "low"}
    )

    assert result.get("success") is True, result
    ((channel, message),) = published
    assert channel == f"notifications:{test_user.id}"
    assert message["type"] == "notification"
    pushed = message["notification"]
    assert pushed["id"] == result["data"]["notification_id"]
    assert pushed["title"] == "Job Complete"
    assert pushed["message"] == "Done"
    assert pushed["priority"] == "low"
    assert pushed["notification_type"] == "agent_job_alert"
    assert pushed["related_entity_id"] == str(job.id)
    assert pushed["is_read"] is False


async def test_send_notification_survives_a_publisher_that_is_down(
    db_session, test_user, monkeypatch
):
    """The bell reads the row; a dead socket push must not lose it."""

    def _down(channel, message):
        raise ConnectionError("redis is not listening")

    monkeypatch.setattr("app.tasks.job_support.publish_sync", _down)
    job = await _job(db_session, test_user)

    result = await _notify(db_session, job, {"title": "Alert", "message": "Stored"})

    assert result.get("success") is True, result
    assert result["data"]["delivered"] is True
    (row,) = await _stored(db_session)
    assert row.message == "Stored"


@pytest.mark.parametrize(
    "given, stored",
    [
        (None, "normal"),
        ("low", "low"),
        ("normal", "normal"),
        ("high", "high"),
        ("urgent", "urgent"),
        ("  URGENT ", "urgent"),
        ("CRITICAL", "normal"),
        ("", "normal"),
    ],
)
async def test_send_notification_priority(db_session, test_user, given, stored):
    job = await _job(db_session, test_user)
    params = {"title": "Alert", "message": "Something happened"}
    if given is not None:
        params["priority"] = given

    result = await _notify(db_session, job, params)

    assert result.get("success") is True, result
    assert result["data"]["priority"] == stored
    (row,) = await _stored(db_session)
    assert row.priority == stored


async def test_send_notification_bounds_what_it_stores(db_session, test_user):
    job = await _job(db_session, test_user)

    result = await _notify(
        db_session,
        job,
        {
            "title": "A" * 300,
            "message": "B" * 3000,
            "action_url": "/" + "c" * 700,
        },
    )

    assert result.get("success") is True, result
    (row,) = await _stored(db_session)
    assert row.title == "A" * 200
    assert row.message == "B" * 2000
    assert row.action_url == "/" + "c" * 499


async def test_send_notification_without_an_action_url_stores_none(
    db_session, test_user
):
    job = await _job(db_session, test_user)

    for action_url in (None, "   "):
        params = {"title": "Alert", "message": "Something happened"}
        if action_url is not None:
            params["action_url"] = action_url
        result = await _notify(db_session, job, params)
        assert result.get("success") is True, result

    assert [row.action_url for row in await _stored(db_session)] == [None, None]


async def test_send_notification_goes_to_the_jobs_owner_and_nobody_else(
    db_session, test_user, published
):
    """A context naming another user does not redirect the notification."""
    other = await _other_user(db_session)
    job = await _job(db_session, test_user)
    ctx = _ctx(db_session, job)
    ctx.user_id = str(other.id)

    result = await _handler("send_notification")(
        {"title": "Alert", "message": "For the owner only"}, ctx
    )

    assert result.get("success") is True, result
    assert [row.user_id for row in await _stored(db_session)] == [test_user.id]
    assert [channel for channel, _ in published] == [f"notifications:{test_user.id}"]


async def test_send_notification_names_the_job_it_came_from(db_session, test_user):
    """Two jobs of one user each notify about themselves."""
    first = await _job(db_session, test_user, name="First")
    second = await _job(db_session, test_user, name="Second")

    await _notify(db_session, first, {"title": "One", "message": "from first"})
    await _notify(db_session, second, {"title": "Two", "message": "from second"})

    by_title = {row.title: row for row in await _stored(db_session)}
    assert by_title["One"].related_entity_id == first.id
    assert by_title["One"].data["source_job_name"] == "First"
    assert by_title["Two"].related_entity_id == second.id
    assert by_title["Two"].data["source_job_id"] == str(second.id)


async def test_send_notification_that_was_not_stored_is_not_a_success(
    db_session, test_user, monkeypatch, published
):
    job = await _job(db_session, test_user)
    ctx = _ctx(db_session, job)
    _fail_first_flush(monkeypatch, db_session)

    result = await _handler("send_notification")(
        {"title": "Alert", "message": "Never written"}, ctx
    )

    assert published == []
    assert result.get("success") is not True, result
    assert "error" in result


async def test_one_run_cannot_send_an_unbounded_number_of_notifications(
    db_session, test_user
):
    job = await _job(db_session, test_user)
    ctx = _ctx(db_session, job)
    handler = _handler("send_notification")

    results = [
        await handler({"title": f"Alert {n}", "message": "Again"}, ctx)
        for n in range(100)
    ]

    assert results[0].get("success") is True, results[0]
    assert any("error" in result for result in results)
    assert len(await _stored(db_session)) < 100


# ---------------------------------------------------------------------------
# send_email_alert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, names",
    [
        ({"body": "something happened"}, "subject"),
        ({"subject": "  ", "body": "something happened"}, "subject"),
        ({"subject": "Alert"}, "body"),
        ({"subject": "Alert", "body": "   "}, "body"),
    ],
)
async def test_send_email_alert_refuses_and_names_what_is_missing(
    db_session, test_user, published, params, names
):
    job = await _job(db_session, test_user)

    result = await _email(db_session, job, params)

    assert "success" not in result
    assert result["error"] == f"{names} is required"
    assert await _stored(db_session) == []
    assert published == []


async def test_send_email_alert_falls_back_to_a_notification_and_says_so(
    db_session, test_user, published
):
    """No SMTP exists in this deployment, so the alert lands in the bell."""
    job = await _job(db_session, test_user)

    result = await _email(
        db_session,
        job,
        {
            "subject": "Urgent: Job Failed",
            "body": "The research job hit an error.",
            "priority": "urgent",
        },
    )

    assert result.get("success") is True, result
    # It must not claim an email went out.
    assert result["data"]["delivery_method"] == "in_app_notification"
    assert "SMTP" in result["data"]["note"]
    assert result["data"]["delivered"] is True
    (row,) = await _stored(db_session)
    assert str(row.id) == result["data"]["notification_id"]
    assert row.user_id == test_user.id
    assert row.notification_type == "agent_job_alert"
    assert row.title == "[Email] Urgent: Job Failed"
    assert row.message == "The research job hit an error."
    assert row.priority == "urgent"
    assert row.related_entity_type == "agent_job"
    assert row.related_entity_id == job.id
    assert row.data["intended_delivery"] == "email"
    assert row.data["source_job_id"] == str(job.id)
    assert [channel for channel, _ in published] == [f"notifications:{test_user.id}"]


@pytest.mark.parametrize(
    "given, stored",
    [(None, "normal"), ("HIGH", "high"), ("whenever", "normal")],
)
async def test_send_email_alert_priority(db_session, test_user, given, stored):
    job = await _job(db_session, test_user)
    params = {"subject": "Alert", "body": "Something happened"}
    if given is not None:
        params["priority"] = given

    result = await _email(db_session, job, params)

    assert result.get("success") is True, result
    (row,) = await _stored(db_session)
    assert row.priority == stored


async def test_send_email_alert_bounds_what_it_stores(db_session, test_user):
    job = await _job(db_session, test_user)

    result = await _email(db_session, job, {"subject": "S" * 400, "body": "B" * 3000})

    assert result.get("success") is True, result
    (row,) = await _stored(db_session)
    assert row.title == "[Email] " + "S" * 180
    assert len(row.title) <= Notification.title.type.length
    assert row.message == "B" * 2000


async def test_send_email_alert_goes_to_the_jobs_owner_and_nobody_else(
    db_session, test_user, published
):
    other = await _other_user(db_session)
    job = await _job(db_session, test_user)
    ctx = _ctx(db_session, job)
    ctx.user_id = str(other.id)

    result = await _handler("send_email_alert")(
        {"subject": "Alert", "body": "For the owner only"}, ctx
    )

    assert result.get("success") is True, result
    assert [row.user_id for row in await _stored(db_session)] == [test_user.id]
    assert [channel for channel, _ in published] == [f"notifications:{test_user.id}"]


async def test_send_email_alert_that_was_not_stored_is_not_a_success(
    db_session, test_user, monkeypatch
):
    job = await _job(db_session, test_user)
    ctx = _ctx(db_session, job)
    _fail_first_flush(monkeypatch, db_session)

    result = await _handler("send_email_alert")(
        {"subject": "Alert", "body": "Never written"}, ctx
    )

    assert result.get("success") is not True, result
    assert "error" in result


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


class TestNotificationToolSchemas:
    """Tests for notification tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "send_notification" in names
        assert "send_email_alert" in names

    def test_send_notification_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("send_notification")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "title" in required
        assert "message" in required

    def test_send_email_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("send_email_alert")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "subject" in required
        assert "body" in required


class TestNotificationToolRegistry:
    """Tests for notification tool registry classification."""

    def test_send_notification_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("send_notification")
        assert meta is not None
        assert meta.effects == "write"

    def test_send_email_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("send_email_alert")
        assert meta is not None
        assert meta.effects == "write"

    def test_notification_tools_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["send_notification", "send_email_alert"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
