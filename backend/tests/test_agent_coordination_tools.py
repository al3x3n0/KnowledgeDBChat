"""`delegate_subtask`, `wait_for_subtask`, `share_findings` and `request_review`,
called through the real handlers.

The handlers are run against the in-memory database and judged on the
`agent_jobs` rows they leave behind and on what the job at the other end can
then see: a subtask is only delegated if a runnable child row exists and was
queued, and a finding is only shared if a sibling can read it.

The Celery `delay` is the one edge replaced; what was queued is recorded and
checked against the task's real signature.
"""

import asyncio
import inspect
from contextlib import contextmanager
from uuid import uuid4

import pytest
from sqlalchemy import event, select, update

from app.models.agent_job import AgentJob
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_collaboration_provider,
)
from app.tasks.agent_job_tasks import execute_agent_job_task

pytestmark = pytest.mark.unit


class _Executor:
    """The executor's only part these handlers touch: saving a checkpoint."""

    def __init__(self):
        self.checkpoints = []

    async def _save_checkpoint(self, job, state, db):
        self.checkpoints.append(str(job.id))


def _handler(tool_name):
    provider = build_autonomous_collaboration_provider(_Executor())
    return provider._handlers[tool_name]


def _ctx(db, job, state=None):
    return AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(job.user_id),
        job=job,
        state={} if state is None else state,
    )


async def _call(tool_name, db, job, params, state=None):
    return await _handler(tool_name)(params, _ctx(db, job, state))


async def _job(db, user, **overrides):
    """A stored job; by default the running job that is calling the tool."""
    fields = {
        "name": "Calling job",
        "goal": "Characterise the prefetcher",
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


async def _other_rows(db, *known):
    """Every job row the test did not create itself."""
    known_ids = {job.id for job in known}
    rows = (await db.execute(select(AgentJob))).scalars().all()
    return [row for row in rows if row.id not in known_ids]


async def _stored(db, job):
    """The job's results as the database holds them."""
    await db.commit()
    await db.refresh(job)
    return job.results if isinstance(job.results, dict) else {}


@pytest.fixture(autouse=True)
def queued(monkeypatch):
    """Record what would have been sent to Celery instead of sending it."""
    calls = []

    def _delay(*args, **kwargs):
        # A call the task's signature cannot take would fail in the worker.
        inspect.signature(execute_agent_job_task.run).bind(*args, **kwargs)
        calls.append((args, kwargs))

    monkeypatch.setattr(execute_agent_job_task, "delay", _delay)
    return calls


@pytest.fixture
def no_sleep(monkeypatch):
    """Polling waits are counted, not waited for."""
    naps = []

    async def _sleep(seconds, *args, **kwargs):
        naps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return naps


@contextmanager
def _inserts_fail():
    """Make every `agent_jobs` INSERT fail in the database itself."""

    def _blank_the_name(mapper, connection, target):
        target.name = None  # NOT NULL

    event.listen(AgentJob, "before_insert", _blank_the_name)
    try:
        yield
    finally:
        event.remove(AgentJob, "before_insert", _blank_the_name)


# ---------------------------------------------------------------------------
# delegate_subtask
# ---------------------------------------------------------------------------


async def test_delegate_creates_a_runnable_child_and_queues_it(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user, name="Parent study")
    state = {"findings": [{"title": "ISB issues no prefetches"}]}

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {
            "name": "Measure stride",
            "goal": "Measure the stride prefetcher on the same kernel",
            "job_type": "analysis",
        },
        state,
    )

    assert result.get("success") is True, result
    await db_session.commit()
    children = await _other_rows(db_session, caller)
    assert [str(row.id) for row in children] == [result["data"]["child_job_id"]]
    child = children[0]
    assert child.user_id == test_user.id
    assert child.parent_job_id == caller.id
    assert child.root_job_id == caller.id
    assert child.chain_depth == 1
    assert child.status == "pending"
    assert child.name == "Measure stride"
    assert child.goal == "Measure the stride prefetcher on the same kernel"
    assert child.job_type == "analysis"
    assert result["data"]["status"] == "pending"
    # The parent can later wait on exactly this child.
    assert state["delegated_subtask_ids"] == [str(child.id)]
    # And a worker was asked to run it, for the same owner.
    assert queued == [((str(child.id), str(test_user.id)), {})]


@pytest.mark.parametrize(
    "params",
    [{"name": "Subtask"}, {"name": "Subtask", "goal": "   "}],
)
async def test_delegate_refuses_a_subtask_with_no_goal(
    db_session, test_user, queued, params
):
    caller = await _job(db_session, test_user)

    result = await _call("delegate_subtask", db_session, caller, params)

    assert "success" not in result
    assert "goal" in result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_delegate_falls_back_to_custom_for_a_job_type_it_does_not_offer(
    db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Fix it", "goal": "Patch the repository", "job_type": "coding"},
    )

    assert result.get("success") is True, result
    child = (await _other_rows(db_session, caller))[0]
    assert child.job_type == "custom"


async def test_delegate_gives_the_child_no_more_budget_than_the_parent_has_left(
    db_session, test_user
):
    caller = await _job(
        db_session,
        test_user,
        max_iterations=10,
        iteration=7,
        max_tool_calls=8,
        max_llm_calls=4,
        max_runtime_minutes=12,
    )

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Do a part of it", "max_iterations": 50},
    )

    assert result.get("success") is True, result
    assert result["data"]["max_iterations"] == 3
    child = (await _other_rows(db_session, caller))[0]
    assert child.max_iterations == 3
    assert child.max_tool_calls == 8
    assert child.max_llm_calls == 4
    assert child.max_runtime_minutes == 12


async def test_delegate_never_creates_a_child_with_a_negative_budget(
    db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Do a part of it", "max_iterations": -5},
    )

    children = await _other_rows(db_session, caller)
    assert "error" in result or (
        children[0].max_iterations >= 1
        and children[0].max_tool_calls >= 1
        and children[0].max_llm_calls >= 1
    )


async def test_delegate_refuses_a_max_iterations_that_is_not_a_number(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user)

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Do a part of it", "max_iterations": "a few"},
    )

    assert "success" not in result
    assert result["error"]
    assert queued == []


async def test_delegate_is_refused_at_the_maximum_depth(db_session, test_user, queued):
    caller = await _job(db_session, test_user, chain_depth=3)

    result = await _call(
        "delegate_subtask", db_session, caller, {"name": "Sub", "goal": "Go deeper"}
    )

    assert "success" not in result
    assert "depth" in result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_a_grandchild_records_its_depth_and_the_root_of_the_tree(
    db_session, test_user
):
    root = await _job(db_session, test_user, name="Root")
    caller = await _job(
        db_session,
        test_user,
        name="Middle",
        parent_job_id=root.id,
        root_job_id=root.id,
        chain_depth=2,
    )

    result = await _call(
        "delegate_subtask", db_session, caller, {"name": "Leaf", "goal": "Last level"}
    )

    assert result.get("success") is True, result
    child = (await _other_rows(db_session, root, caller))[0]
    assert child.chain_depth == 3
    assert child.parent_job_id == caller.id
    assert child.root_job_id == root.id


async def test_a_parent_may_delegate_five_subtasks_and_no_more(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user)
    state = {}

    for index in range(5):
        result = await _call(
            "delegate_subtask",
            db_session,
            caller,
            {"name": f"Sub {index}", "goal": f"Part {index}"},
            state,
        )
        assert result.get("success") is True, result
    sixth = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub 5", "goal": "One too many"},
        state,
    )

    assert "success" not in sixth
    assert "5" in sixth["error"]
    assert len(await _other_rows(db_session, caller)) == 5
    assert len(queued) == 5
    assert len(state["delegated_subtask_ids"]) == 5


async def test_delegate_passes_config_and_the_parents_findings_to_the_child(
    db_session, test_user
):
    caller = await _job(db_session, test_user)
    findings = [{"title": f"finding {i}"} for i in range(25)]

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Continue", "config": {"source_id": "repo-1"}},
        {"findings": findings},
    )

    assert result.get("success") is True, result
    child = (await _other_rows(db_session, caller))[0]
    assert child.config["source_id"] == "repo-1"
    # The most recent twenty.
    assert child.config["inherited_data"]["parent_findings"] == findings[-20:]


async def test_delegate_withholds_the_findings_when_asked_to(db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Start clean", "share_findings": False},
        {"findings": [{"title": "private"}]},
    )

    assert result.get("success") is True, result
    child = (await _other_rows(db_session, caller))[0]
    assert "inherited_data" not in (child.config or {})


async def test_the_child_is_told_the_findings_its_parent_shared(db_session, test_user):
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    caller = await _job(db_session, test_user)
    await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Continue from what is known"},
        {"findings": [{"title": "ZEBRA-FINDING: ISB issued zero prefetches"}]},
    )
    child = (await _other_rows(db_session, caller))[0]

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(child, None, {})

    assert "ZEBRA-FINDING" in prompt


async def test_delegate_with_wait_returns_the_childs_results_once_it_finishes(
    db_session, test_user, monkeypatch
):
    caller = await _job(db_session, test_user)
    state = {}

    async def _child_finishes(seconds, *args, **kwargs):
        await db_session.execute(
            update(AgentJob)
            .where(AgentJob.parent_job_id == caller.id)
            .values(status="completed", results={"findings": [{"title": "done"}]})
        )

    monkeypatch.setattr(asyncio, "sleep", _child_finishes)

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Quick one", "wait": True},
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "completed"
    assert result["data"]["results"] == {"findings": [{"title": "done"}]}
    child_id = result["data"]["child_job_id"]
    assert state["delegated_subtask_results"][child_id] == result["data"]["results"]


async def test_delegate_with_wait_gives_up_after_a_minute_and_says_so(
    db_session, test_user, no_sleep
):
    caller = await _job(db_session, test_user)

    result = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "Slow one", "wait": True, "timeout_seconds": 3600},
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "pending"
    assert "wait_for_subtask" in result["data"]["note"]
    assert sum(no_sleep) <= 60


async def test_a_failed_delegation_leaves_the_session_usable(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user)
    state = {}

    with _inserts_fail():
        result = await _call(
            "delegate_subtask",
            db_session,
            caller,
            {"name": "Sub", "goal": "This insert fails"},
            state,
        )

    assert "success" not in result
    assert "Failed to create child job" in result["error"]
    assert queued == []
    assert not state.get("delegated_subtask_ids")
    # The executor goes on using this session and this job object.
    assert caller.name == "Calling job"
    assert await _other_rows(db_session, caller) == []
    retry = await _call(
        "delegate_subtask",
        db_session,
        caller,
        {"name": "Sub", "goal": "This one works"},
        state,
    )
    assert retry.get("success") is True, retry
    assert len(await _other_rows(db_session, caller)) == 1


# ---------------------------------------------------------------------------
# wait_for_subtask
# ---------------------------------------------------------------------------


async def _delegated(db, user, caller, **overrides):
    """A child the caller delegated, and the state that records it."""
    fields = {
        "name": "Child",
        "parent_job_id": caller.id,
        "root_job_id": caller.id,
        "chain_depth": 1,
        "status": "running",
    }
    fields.update(overrides)
    child = await _job(db, user, **fields)
    return child, {"delegated_subtask_ids": [str(child.id)]}


async def test_wait_reports_a_finished_child_and_its_results(
    db_session, test_user, no_sleep
):
    caller = await _job(db_session, test_user)
    child, state = await _delegated(
        db_session,
        test_user,
        caller,
        status="completed",
        progress=100,
        results={"findings": [{"title": "a"}, {"title": "b"}], "summary": "ok"},
    )

    result = await _call(
        "wait_for_subtask",
        db_session,
        caller,
        {"subtask_job_id": str(child.id)},
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "completed"
    assert result["data"]["progress"] == 100
    assert result["data"]["results"]["summary"] == "ok"
    assert result["data"]["findings_count"] == 2
    assert no_sleep == []
    final = state["delegated_subtask_final"][str(child.id)]
    assert final["status"] == "completed" and final["results"]["summary"] == "ok"


async def test_wait_reports_a_failed_child_as_failed(db_session, test_user, no_sleep):
    caller = await _job(db_session, test_user)
    child, state = await _delegated(
        db_session, test_user, caller, status="failed", error="tool unavailable"
    )

    result = await _call(
        "wait_for_subtask",
        db_session,
        caller,
        {"subtask_job_id": str(child.id)},
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "failed"
    assert result["data"]["findings_count"] == 0
    assert no_sleep == []


async def test_wait_polls_no_longer_than_two_minutes_for_a_child_still_running(
    db_session, test_user, no_sleep
):
    caller = await _job(db_session, test_user)
    child, state = await _delegated(db_session, test_user, caller)

    result = await _call(
        "wait_for_subtask",
        db_session,
        caller,
        {"subtask_job_id": str(child.id), "timeout_seconds": 86400},
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "running"
    assert 0 < sum(no_sleep) <= 120


async def test_wait_sees_the_child_finish_while_it_is_polling(
    db_session, test_user, monkeypatch
):
    caller = await _job(db_session, test_user)
    child, state = await _delegated(db_session, test_user, caller)

    async def _child_finishes(seconds, *args, **kwargs):
        await db_session.execute(
            update(AgentJob)
            .where(AgentJob.id == child.id)
            .values(status="completed", results={"findings": [{"title": "late"}]})
        )

    monkeypatch.setattr(asyncio, "sleep", _child_finishes)

    result = await _call(
        "wait_for_subtask",
        db_session,
        caller,
        {"subtask_job_id": str(child.id)},
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "completed"
    assert result["data"]["findings_count"] == 1


async def test_waiting_again_on_a_child_still_running_does_not_call_it_completed(
    db_session, test_user, no_sleep
):
    caller = await _job(db_session, test_user)
    child, state = await _delegated(
        db_session,
        test_user,
        caller,
        results={"findings": [{"title": "partial"}]},
    )
    params = {"subtask_job_id": str(child.id)}

    first = await _call("wait_for_subtask", db_session, caller, params, state)
    assert first["data"]["status"] == "running"

    second = await _call("wait_for_subtask", db_session, caller, params, state)

    assert second["data"]["status"] == "running"


@pytest.mark.parametrize("params", [{}, {"subtask_job_id": "  "}])
async def test_wait_refuses_a_call_that_names_no_subtask(db_session, test_user, params):
    caller = await _job(db_session, test_user)

    result = await _call("wait_for_subtask", db_session, caller, params)

    assert "success" not in result
    assert result["error"]


async def test_wait_refuses_a_job_this_job_did_not_delegate(
    db_session, test_user, admin_user, no_sleep
):
    caller = await _job(db_session, test_user)
    mine_but_unrelated = await _job(
        db_session, test_user, name="Unrelated", status="completed", results={"k": "v"}
    )
    theirs = await _job(
        db_session,
        admin_user,
        name="Another tenant",
        status="completed",
        results={"secret": "value"},
    )

    for target in (mine_but_unrelated, theirs, caller):
        result = await _call(
            "wait_for_subtask", db_session, caller, {"subtask_job_id": str(target.id)}
        )

        assert "success" not in result
        assert "not a delegated subtask" in result["error"]
        assert "secret" not in str(result)


async def test_wait_does_not_trust_a_recorded_id_that_is_not_its_own_child(
    db_session, test_user, admin_user, no_sleep
):
    """State is restored from a checkpoint; the row decides, not the list."""
    caller = await _job(db_session, test_user)
    theirs = await _job(
        db_session,
        admin_user,
        name="Another tenant",
        status="completed",
        results={"secret": "value"},
    )
    state = {"delegated_subtask_ids": [str(theirs.id), str(uuid4()), "not-a-uuid"]}

    for subtask_id in state["delegated_subtask_ids"]:
        result = await _call(
            "wait_for_subtask",
            db_session,
            caller,
            {"subtask_job_id": subtask_id},
            state,
        )

        assert "success" not in result
        assert result["error"]
        assert "secret" not in str(result)
    assert not state.get("delegated_subtask_results")


# ---------------------------------------------------------------------------
# share_findings
# ---------------------------------------------------------------------------


async def _family(db, user):
    """A parent with three children; the first is the one calling the tool."""
    parent = await _job(db, user, name="Parent")
    children = [
        await _job(
            db,
            user,
            name=name,
            parent_job_id=parent.id,
            root_job_id=parent.id,
            chain_depth=1,
        )
        for name in ("Caller", "Sibling A", "Sibling B")
    ]
    return parent, children


FINDING = {
    "title": "ISB never engaged",
    "content": "pfIdentified = 0 on every kernel",
    "category": "measurement",
}


@pytest.mark.parametrize("findings", [None, [], "a finding", {"title": "not a list"}])
async def test_share_refuses_a_call_with_no_findings(db_session, test_user, findings):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    params = {} if findings is None else {"findings": findings}

    result = await _call("share_findings", db_session, caller, params)

    assert "success" not in result
    assert "findings" in result["error"].lower()
    assert "shared_findings" not in await _stored(db_session, sibling_a)


async def test_share_is_refused_for_a_job_with_no_parent(db_session, test_user):
    caller = await _job(db_session, test_user)
    await _job(db_session, test_user, name="Unrelated")

    result = await _call("share_findings", db_session, caller, {"findings": [FINDING]})

    assert "success" not in result
    assert "no parent" in result["error"]


async def test_share_writes_the_findings_onto_every_sibling_and_nobody_else(
    db_session, test_user
):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)
    unrelated = await _job(db_session, test_user, name="Unrelated")

    result = await _call("share_findings", db_session, caller, {"findings": [FINDING]})

    assert result.get("success") is True, result
    assert result["data"] == {"siblings_updated": 2, "findings_shared": 1}
    for sibling in (sibling_a, sibling_b):
        shared = (await _stored(db_session, sibling))["shared_findings"]
        assert len(shared) == 1
        assert shared[0]["from_job_id"] == str(caller.id)
        assert shared[0]["title"] == FINDING["title"]
        assert shared[0]["content"] == FINDING["content"]
        assert shared[0]["category"] == FINDING["category"]
        assert shared[0]["shared_at"]
    for bystander in (caller, parent, unrelated):
        assert "shared_findings" not in await _stored(db_session, bystander)


async def test_share_keeps_what_a_sibling_already_recorded(db_session, test_user):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    sibling_a.results = {
        "findings": [{"title": "its own"}],
        "shared_findings": [{"title": "earlier"}],
    }
    await db_session.commit()

    result = await _call("share_findings", db_session, caller, {"findings": [FINDING]})

    assert result.get("success") is True, result
    stored = await _stored(db_session, sibling_a)
    assert stored["findings"] == [{"title": "its own"}]
    assert [f["title"] for f in stored["shared_findings"]] == [
        "earlier",
        FINDING["title"],
    ]


async def test_share_can_be_addressed_to_one_sibling(db_session, test_user):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)

    result = await _call(
        "share_findings",
        db_session,
        caller,
        {"findings": [FINDING], "target_job_ids": [str(sibling_b.id)]},
    )

    assert result.get("success") is True, result
    assert result["data"]["siblings_updated"] == 1
    assert "shared_findings" not in await _stored(db_session, sibling_a)
    assert len((await _stored(db_session, sibling_b))["shared_findings"]) == 1


async def test_share_addressed_to_a_job_that_is_not_a_sibling_reaches_nobody(
    db_session, test_user, admin_user
):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)
    unrelated = await _job(db_session, test_user, name="Unrelated")
    theirs = await _job(db_session, admin_user, name="Another tenant")

    result = await _call(
        "share_findings",
        db_session,
        caller,
        {
            "findings": [FINDING],
            "target_job_ids": [
                str(unrelated.id),
                str(theirs.id),
                str(parent.id),
                str(caller.id),
                str(uuid4()),
            ],
        },
    )

    assert result.get("success") is True, result
    assert result["data"]["siblings_updated"] == 0
    for job in (unrelated, theirs, parent, caller, sibling_a, sibling_b):
        assert "shared_findings" not in await _stored(db_session, job)


async def test_share_addressed_only_to_unreadable_ids_is_not_sent_to_everyone(
    db_session, test_user
):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)

    result = await _call(
        "share_findings",
        db_session,
        caller,
        {"findings": [FINDING], "target_job_ids": ["Sibling B"]},
    )

    assert result.get("data", {}).get("siblings_updated", 0) == 0
    assert "shared_findings" not in await _stored(db_session, sibling_a)
    assert "shared_findings" not in await _stored(db_session, sibling_b)


async def test_share_does_not_write_into_another_users_job(
    db_session, test_user, admin_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    theirs = await _job(
        db_session,
        admin_user,
        name="Another tenant under the same parent",
        parent_job_id=parent.id,
    )

    result = await _call("share_findings", db_session, caller, {"findings": [FINDING]})

    assert result.get("success") is True, result
    assert "shared_findings" not in await _stored(db_session, theirs)
    assert result["data"]["siblings_updated"] == 2


async def test_share_sends_at_most_ten_findings_and_trims_each(db_session, test_user):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    findings = [
        {"title": f"{i}-" + "t" * 500, "content": "c" * 5000, "category": "k" * 500}
        for i in range(15)
    ]

    result = await _call("share_findings", db_session, caller, {"findings": findings})

    assert result.get("success") is True, result
    assert result["data"]["findings_shared"] == 10
    shared = (await _stored(db_session, sibling_a))["shared_findings"]
    assert len(shared) == 10
    assert shared[0]["title"].startswith("0-")
    assert all(len(f["title"]) == 200 for f in shared)
    assert all(len(f["content"]) == 1000 for f in shared)
    assert all(len(f["category"]) == 100 for f in shared)


async def test_a_sibling_keeps_only_the_newest_fifty_shared_findings(
    db_session, test_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)

    for batch in range(6):
        result = await _call(
            "share_findings",
            db_session,
            caller,
            {"findings": [{"title": f"{batch}-{i}"} for i in range(10)]},
        )
        assert result.get("success") is True, result

    shared = (await _stored(db_session, sibling_a))["shared_findings"]
    assert len(shared) == 50
    assert shared[0]["title"] == "1-0"
    assert shared[-1]["title"] == "5-9"


async def test_a_sibling_can_read_the_findings_shared_with_it(db_session, test_user):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    await _call("share_findings", db_session, caller, {"findings": [FINDING]})
    await db_session.commit()
    await db_session.refresh(sibling_a)

    result = await _call("read_agent_messages", db_session, sibling_a, {})

    assert result.get("success") is True, result
    assert result["data"]["shared_findings_count"] == 1
    assert FINDING["content"] in str(result["data"])


# ---------------------------------------------------------------------------
# request_review
# ---------------------------------------------------------------------------


async def test_a_human_review_leaves_a_checkpoint_and_spawns_nothing(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user, iteration=4)
    state = {}

    result = await _call(
        "request_review",
        db_session,
        caller,
        {
            "review_type": "human",
            "content_to_review": "Draft: ISB is inert on L2",
            "review_criteria": ["accuracy", "completeness"],
        },
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["action"] == "paused_for_human_review"
    checkpoint = state["approval_checkpoint_pending"]
    assert checkpoint["type"] == "review_request"
    assert checkpoint["content_to_review"] == "Draft: ISB is inert on L2"
    assert checkpoint["review_criteria"] == ["accuracy", "completeness"]
    assert result["data"]["checkpoint"] == checkpoint
    assert state["review_requests"][0]["type"] == "human"
    assert state["review_requests"][0]["iteration"] == 4
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_a_peer_review_creates_a_reviewer_job_and_queues_it(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user, name="Prefetcher study")
    state = {}

    result = await _call(
        "request_review",
        db_session,
        caller,
        {
            "content_to_review": "Stride is 2.1x over no prefetcher",
            "review_criteria": ["is the baseline fair", 7, "is the spread reported"],
        },
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["action"] == "peer_review_spawned"
    await db_session.commit()
    reviewers = await _other_rows(db_session, caller)
    assert [str(row.id) for row in reviewers] == [result["data"]["review_job_id"]]
    reviewer = reviewers[0]
    assert reviewer.user_id == test_user.id
    assert reviewer.parent_job_id == caller.id
    assert reviewer.root_job_id == caller.id
    assert reviewer.chain_depth == 1
    assert reviewer.status == "pending"
    assert reviewer.job_type == "analysis"
    assert reviewer.name
    assert "Stride is 2.1x over no prefetcher" in reviewer.goal
    assert "is the baseline fair" in reviewer.goal
    assert "is the spread reported" in reviewer.goal
    assert queued == [((str(reviewer.id), str(test_user.id)), {})]
    assert state["review_requests"][0]["type"] == "peer_agent"
    assert "approval_checkpoint_pending" not in state


async def test_the_requester_can_collect_the_peer_reviewers_verdict(
    db_session, test_user, no_sleep
):
    caller = await _job(db_session, test_user)
    state = {}
    spawned = await _call(
        "request_review",
        db_session,
        caller,
        {"content_to_review": "Stride is 2.1x over no prefetcher"},
        state,
    )
    review_job_id = spawned["data"]["review_job_id"]
    await db_session.execute(
        update(AgentJob)
        .where(AgentJob.parent_job_id == caller.id)
        .values(status="completed", results={"summary": "baseline is unfair"})
    )
    await db_session.commit()

    result = await _call(
        "wait_for_subtask",
        db_session,
        caller,
        {"subtask_job_id": review_job_id},
        state,
    )

    assert result.get("success") is True, result
    assert result["data"]["status"] == "completed"
    assert result["data"]["results"] == {"summary": "baseline is unfair"}


@pytest.mark.parametrize("review_type", ["human", "peer_agent"])
async def test_request_review_refuses_a_request_with_nothing_to_review(
    db_session, test_user, queued, review_type
):
    caller = await _job(db_session, test_user)
    state = {}

    result = await _call(
        "request_review", db_session, caller, {"review_type": review_type}, state
    )

    assert "success" not in result
    assert "content_to_review" in result["error"]
    assert not state.get("approval_checkpoint_pending")
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_a_peer_review_is_refused_at_the_maximum_depth(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user, chain_depth=3)

    result = await _call(
        "request_review", db_session, caller, {"content_to_review": "Deep draft"}
    )

    assert "success" not in result
    assert "depth" in result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_a_human_review_is_still_available_at_the_maximum_depth(
    db_session, test_user
):
    caller = await _job(db_session, test_user, chain_depth=3)
    state = {}

    result = await _call(
        "request_review",
        db_session,
        caller,
        {"review_type": "human", "content_to_review": "Deep draft"},
        state,
    )

    assert result.get("success") is True, result
    assert state["approval_checkpoint_pending"]["content_to_review"] == "Deep draft"


async def test_a_peer_review_counts_against_the_parents_child_budget(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user)
    state = {}
    for index in range(5):
        delegated = await _call(
            "delegate_subtask",
            db_session,
            caller,
            {"name": f"Sub {index}", "goal": f"Part {index}"},
            state,
        )
        assert delegated.get("success") is True, delegated

    result = await _call(
        "request_review", db_session, caller, {"content_to_review": "A sixth"}, state
    )

    assert "success" not in result
    assert len(await _other_rows(db_session, caller)) == 5
    assert len(queued) == 5


def test_request_review_offers_no_reviewer_job_id():
    """It was declared as "specific sibling job to request review from" and
    never read: the named sibling was told nothing and a new reviewer was
    spawned instead. To reach a particular job, send it a message."""
    from app.services.agent_tools import get_tool_by_name

    tool = get_tool_by_name("request_review")
    assert "reviewer_job_id" not in tool["parameters"]["properties"]


async def test_review_keeps_only_the_latest_twenty_requests_and_ten_criteria(
    db_session, test_user
):
    caller = await _job(db_session, test_user)
    state = {}

    for index in range(22):
        await _call(
            "request_review",
            db_session,
            caller,
            {
                "review_type": "human",
                "content_to_review": f"draft {index}",
                "review_criteria": [f"criterion {i}" for i in range(15)],
            },
            state,
        )

    assert len(state["review_requests"]) == 20
    assert state["review_requests"][0]["content"] == "draft 2"
    assert len(state["review_requests"][-1]["criteria"]) == 10
    assert len(state["approval_checkpoint_pending"]["review_criteria"]) == 10


async def test_a_failed_peer_review_spawn_leaves_the_session_usable(
    db_session, test_user, queued
):
    caller = await _job(db_session, test_user)
    state = {}

    with _inserts_fail():
        result = await _call(
            "request_review", db_session, caller, {"content_to_review": "Draft"}, state
        )

    assert "success" not in result
    assert "Failed to spawn peer review" in result["error"]
    assert queued == []
    assert not state.get("delegated_subtask_ids")
    assert caller.name == "Calling job"
    assert await _other_rows(db_session, caller) == []
    retry = await _call(
        "request_review", db_session, caller, {"content_to_review": "Draft"}, state
    )
    assert retry.get("success") is True, retry
