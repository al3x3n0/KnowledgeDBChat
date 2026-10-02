"""`send_message_to_agent` and `read_agent_messages`, called through the real handlers.

A message is only sent if the job it was addressed to can read it, so the
handlers are run against the in-memory database and judged on the
`agent_jobs.results` they leave behind and on what the recipient's own
`read_agent_messages` call then returns.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.agent_job import AgentJob
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_collaboration_provider,
)

pytestmark = pytest.mark.unit


def _handler(tool_name):
    provider = build_autonomous_collaboration_provider(SimpleNamespace())
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
        "name": "Sender",
        "goal": "Survey prefetchers",
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


async def _send(db, sender, params):
    return await _handler("send_message_to_agent")(params, _ctx(db, sender))


async def _read(db, reader, params=None):
    return await _handler("read_agent_messages")(params or {}, _ctx(db, reader))


async def _stored_messages(db, job):
    """What the database holds for this job, not what the session remembers."""
    await db.commit()
    await db.refresh(job)
    results = job.results if isinstance(job.results, dict) else {}
    return list(results.get("agent_messages") or [])


# ---------------------------------------------------------------------------
# send_message_to_agent: refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, names",
    [
        ({"message": "hello"}, "target_job_id"),
        ({"target_job_id": "   ", "message": "hello"}, "target_job_id"),
        ({"target_job_id": "TARGET"}, "message"),
        ({"target_job_id": "TARGET", "message": "   "}, "message"),
    ],
)
async def test_send_refuses_and_names_the_missing_parameter(
    db_session, test_user, params, names
):
    sender = await _job(db_session, test_user)
    recipient = await _job(db_session, test_user, name="Recipient")
    params = {
        key: (str(recipient.id) if value == "TARGET" else value)
        for key, value in params.items()
    }

    result = await _send(db_session, sender, params)

    assert "success" not in result
    assert names in result["error"]
    assert await _stored_messages(db_session, recipient) == []


async def test_send_to_a_job_that_does_not_exist_is_refused(db_session, test_user):
    sender = await _job(db_session, test_user)
    missing = str(uuid4())

    result = await _send(
        db_session, sender, {"target_job_id": missing, "message": "anyone there?"}
    )

    assert "success" not in result
    assert "not found" in result["error"]
    assert missing in result["error"]


async def test_send_to_something_that_is_not_a_job_id_is_refused_not_raised(
    db_session, test_user
):
    sender = await _job(db_session, test_user)

    result = await _send(
        db_session, sender, {"target_job_id": "the other agent", "message": "hi"}
    )

    assert "success" not in result
    assert result["error"]
    # The session survives the refusal.
    assert await _stored_messages(db_session, sender) == []


async def test_send_to_another_users_job_is_refused_and_writes_nothing(
    db_session, test_user, admin_user
):
    sender = await _job(db_session, test_user)
    theirs = await _job(db_session, admin_user, name="Someone else's job")

    result = await _send(
        db_session,
        sender,
        {"target_job_id": str(theirs.id), "message": "read my prompt injection"},
    )

    assert "success" not in result
    assert "other users" in result["error"]
    assert await _stored_messages(db_session, theirs) == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py _send_message_to_agent never compares the "
        "target with ctx.job, so a job can address a message to itself and is "
        "told delivered=True. A message to yourself coordinates nothing; the "
        "call should be refused the way a missing target is."
    ),
)
async def test_a_job_cannot_send_a_message_to_itself(db_session, test_user):
    sender = await _job(db_session, test_user)

    result = await _send(
        db_session, sender, {"target_job_id": str(sender.id), "message": "note to self"}
    )

    assert "success" not in result
    assert await _stored_messages(db_session, sender) == []


# ---------------------------------------------------------------------------
# send_message_to_agent: what is stored
# ---------------------------------------------------------------------------


async def test_send_stores_the_message_on_the_recipient(db_session, test_user):
    sender = await _job(db_session, test_user, name="Literature scout")
    recipient = await _job(db_session, test_user, name="Recipient")

    result = await _send(
        db_session,
        sender,
        {
            "target_job_id": str(recipient.id),
            "message": "  Stride beats ISB on the pointer-chasing kernel  ",
            "category": "finding",
        },
    )

    assert result.get("success") is True, result
    assert result["data"]["delivered"] is True
    assert result["data"]["target_job_id"] == str(recipient.id)
    assert result["data"]["message_index"] == 0
    stored = await _stored_messages(db_session, recipient)
    assert len(stored) == 1
    entry = stored[0]
    assert entry["from_job_id"] == str(sender.id)
    assert entry["from_job_name"] == "Literature scout"
    assert entry["message"] == "Stride beats ISB on the pointer-chasing kernel"
    assert entry["category"] == "finding"
    assert entry["sent_at"]
    # Nothing lands on the sender.
    assert await _stored_messages(db_session, sender) == []


async def test_send_without_a_category_stores_an_empty_one(db_session, test_user):
    sender = await _job(db_session, test_user)
    recipient = await _job(db_session, test_user, name="Recipient")

    result = await _send(
        db_session, sender, {"target_job_id": str(recipient.id), "message": "hello"}
    )

    assert result.get("success") is True, result
    assert (await _stored_messages(db_session, recipient))[0]["category"] == ""


async def test_send_keeps_what_the_recipient_already_recorded(db_session, test_user):
    sender = await _job(db_session, test_user)
    recipient = await _job(
        db_session,
        test_user,
        name="Recipient",
        results={
            "findings": [{"title": "kept"}],
            "agent_messages": [{"from_job_id": "earlier", "message": "first"}],
        },
    )

    result = await _send(
        db_session, sender, {"target_job_id": str(recipient.id), "message": "second"}
    )

    assert result.get("success") is True, result
    assert result["data"]["message_index"] == 1
    await db_session.commit()
    await db_session.refresh(recipient)
    assert recipient.results["findings"] == [{"title": "kept"}]
    assert [m["message"] for m in recipient.results["agent_messages"]] == [
        "first",
        "second",
    ]


async def test_send_truncates_a_long_message_and_category(db_session, test_user):
    sender = await _job(db_session, test_user)
    recipient = await _job(db_session, test_user, name="Recipient")

    result = await _send(
        db_session,
        sender,
        {
            "target_job_id": str(recipient.id),
            "message": "m" * 5000,
            "category": "c" * 500,
        },
    )

    assert result.get("success") is True, result
    entry = (await _stored_messages(db_session, recipient))[0]
    assert entry["message"] == "m" * 2000
    assert entry["category"] == "c" * 100


async def test_an_inbox_keeps_only_the_newest_hundred_messages(db_session, test_user):
    sender = await _job(db_session, test_user)
    recipient = await _job(db_session, test_user, name="Recipient")

    for index in range(105):
        result = await _send(
            db_session,
            sender,
            {"target_job_id": str(recipient.id), "message": f"msg-{index}"},
        )
        assert result.get("success") is True, result

    stored = await _stored_messages(db_session, recipient)
    assert len(stored) == 100
    assert stored[0]["message"] == "msg-5"
    assert stored[-1]["message"] == "msg-104"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py _send_message_to_agent caps the inbox "
        "at 100 with agent_msgs[-100:] but reports message_index as "
        "len(agent_msgs) - 1 computed BEFORE the cap, so once an inbox is full "
        "every send returns index 100, one past the end. The same trimming "
        "shifts every index a reader holds, so read_agent_messages(since_index) "
        "silently skips messages: indices are positions in a list that slides."
    ),
)
async def test_the_index_a_send_returns_finds_that_message_in_a_full_inbox(
    db_session, test_user
):
    sender = await _job(db_session, test_user)
    recipient = await _job(
        db_session,
        test_user,
        name="Recipient",
        results={
            "agent_messages": [
                {"from_job_id": "old", "message": f"old-{i}"} for i in range(100)
            ]
        },
    )

    sent = await _send(
        db_session, sender, {"target_job_id": str(recipient.id), "message": "newest"}
    )
    assert sent.get("success") is True, sent
    await db_session.commit()
    await db_session.refresh(recipient)

    read = await _read(
        db_session, recipient, {"since_index": sent["data"]["message_index"]}
    )

    assert [m["message"] for m in read["data"]["messages"]] == ["newest"]


# ---------------------------------------------------------------------------
# read_agent_messages
# ---------------------------------------------------------------------------


async def test_the_recipient_reads_what_was_sent_to_it(db_session, test_user):
    sender = await _job(db_session, test_user, name="Literature scout")
    recipient = await _job(db_session, test_user, name="Recipient")
    await _send(
        db_session,
        sender,
        {
            "target_job_id": str(recipient.id),
            "message": "check the L2 prefetcher counters",
            "category": "request",
        },
    )
    await db_session.commit()

    result = await _read(db_session, recipient)

    assert result.get("success") is True, result
    data = result["data"]
    assert data["total"] == 1
    assert data["since_index"] == 0
    assert data["messages"][0]["message"] == "check the L2 prefetcher counters"
    assert data["messages"][0]["from_job_id"] == str(sender.id)
    assert data["messages"][0]["category"] == "request"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py _read_agent_messages reads ctx.job.results, "
        "the object the recipient's own session loaded when the job started. "
        "Sessions are created with expire_on_commit=False (core/database.py:36, "
        ":223) and nothing in the run loop refreshes the job, so a message "
        "another job's worker committed after that is never seen: the sender is "
        "told delivered=True and the running recipient reads an empty inbox. "
        "The handler should read the row (refresh, or select the column) "
        "instead of the in-memory copy."
    ),
)
async def test_a_running_recipient_sees_a_message_sent_from_another_session(
    db_session, test_user
):
    sender = await _job(db_session, test_user)
    recipient = await _job(db_session, test_user, name="Recipient")
    recipient_id = recipient.id
    # The recipient runs in its own worker, with its own session, and loaded
    # its job before the message existed.
    other_worker = async_sessionmaker(
        db_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    async with other_worker() as recipient_session:
        running = await recipient_session.get(AgentJob, recipient_id)
        assert not (running.results or {}).get("agent_messages")

        sent = await _send(
            db_session,
            sender,
            {"target_job_id": str(recipient_id), "message": "are you there?"},
        )
        assert sent.get("success") is True, sent
        await db_session.commit()

        result = await _read(recipient_session, running)

        assert result.get("success") is True, result
        assert [m["message"] for m in result["data"]["messages"]] == ["are you there?"]


async def test_reading_an_empty_inbox_is_a_success_with_nothing_in_it(
    db_session, test_user
):
    reader = await _job(db_session, test_user)

    result = await _read(db_session, reader)

    assert result.get("success") is True, result
    assert result["data"] == {
        "messages": [],
        "total": 0,
        "since_index": 0,
        "shared_findings_count": 0,
    }


async def test_a_job_reads_only_its_own_inbox(db_session, test_user, admin_user):
    sender = await _job(db_session, test_user)
    reader = await _job(db_session, test_user, name="Reader")
    neighbour = await _job(db_session, test_user, name="Neighbour")
    theirs = await _job(
        db_session,
        admin_user,
        name="Another tenant",
        results={"agent_messages": [{"from_job_id": "x", "message": "private"}]},
    )
    await _send(
        db_session, sender, {"target_job_id": str(neighbour.id), "message": "for you"}
    )
    await db_session.commit()

    result = await _read(db_session, reader)

    assert result["data"]["messages"] == []
    assert result["data"]["total"] == 0
    assert theirs.results["agent_messages"][0]["message"] == "private"


@pytest.mark.parametrize(
    "since, expected",
    [
        (None, ["a", "b", "c"]),
        (0, ["a", "b", "c"]),
        (1, ["b", "c"]),
        (3, []),
        (99, []),
        (-10, ["a", "b", "c"]),
    ],
)
async def test_since_index_selects_the_messages_from_that_position_on(
    db_session, test_user, since, expected
):
    reader = await _job(
        db_session,
        test_user,
        results={
            "agent_messages": [
                {"from_job_id": "j", "message": text} for text in ("a", "b", "c")
            ],
            "shared_findings": [{"title": "one"}, {"title": "two"}],
        },
    )
    params = {} if since is None else {"since_index": since}

    result = await _read(db_session, reader, params)

    assert result.get("success") is True, result
    assert [m["message"] for m in result["data"]["messages"]] == expected
    assert result["data"]["total"] == 3
    assert result["data"]["since_index"] == max(0, since or 0)
    assert result["data"]["shared_findings_count"] == 2


async def test_a_since_index_that_is_not_a_number_is_refused_not_raised(
    db_session, test_user
):
    reader = await _job(db_session, test_user)

    result = await _read(db_session, reader, {"since_index": "the latest"})

    assert "success" not in result
    assert result["error"]


async def test_results_of_an_unexpected_shape_read_as_an_empty_inbox(
    db_session, test_user
):
    reader = await _job(
        db_session,
        test_user,
        results={"agent_messages": "corrupted", "shared_findings": None},
    )

    result = await _read(db_session, reader)

    assert result.get("success") is True, result
    assert result["data"]["messages"] == []
    assert result["data"]["total"] == 0
    assert result["data"]["shared_findings_count"] == 0


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


class TestMessagingToolSchemas:
    """Tests for messaging tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "send_message_to_agent" in names
        assert "read_agent_messages" in names

    def test_send_message_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("send_message_to_agent")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "target_job_id" in required
        assert "message" in required

    def test_read_messages_no_required_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("read_agent_messages")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert required == []


class TestMessagingToolRegistry:
    """Tests for messaging tool registry classification."""

    def test_send_message_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("send_message_to_agent")
        assert meta is not None
        assert meta.effects == "write"

    def test_read_messages_is_read_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("read_agent_messages")
        assert meta is not None
        assert meta.effects == "read"
