"""`create_handoff`, `get_sibling_status` and `broadcast_to_siblings`, called
through the real handlers.

The handlers are run against the in-memory database and judged on the
`agent_jobs` rows they leave behind and on what the job at the other end then
sees: a handoff is only a handoff if a runnable child exists, was queued, and
is shown its contract; a broadcast is only delivered if a sibling can read it.

The Celery `delay` is the one edge replaced; what was queued is recorded and
checked against the task's real signature.
"""

import inspect
from contextlib import contextmanager
from uuid import uuid4

import pytest
from sqlalchemy import event, select

from app.models.agent_job import AgentJob
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_collaboration_provider,
    build_autonomous_output_state_provider,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.tasks.agent_job_tasks import execute_agent_job_task

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def executor():
    """The real executor: the handoff asks it for the job's source scope."""
    return AutonomousAgentExecutor()


def _ctx(db, job, state=None):
    return AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(job.user_id),
        job=job,
        state={} if state is None else state,
    )


async def _call(executor, tool_name, db, job, params, state=None):
    provider = build_autonomous_output_state_provider(executor)
    return await provider._handlers[tool_name](params, _ctx(db, job, state))


async def _read_messages(executor, db, job):
    """The recipient's side: the tool a job reads its inbox with."""
    provider = build_autonomous_collaboration_provider(executor)
    return await provider._handlers["read_agent_messages"]({}, _ctx(db, job))


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


HANDOFF = {
    "goal": "Re-profile the kernel at function granularity",
    "context": "The current profile is per-module and too coarse to mine",
    "expected_outputs": ["hot_functions", "summary"],
}


# ---------------------------------------------------------------------------
# create_handoff: refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, names",
    [
        ({"expected_outputs": ["summary"]}, "goal"),
        ({"goal": "   ", "expected_outputs": ["summary"]}, "goal"),
        ({"goal": "Re-profile"}, "expected_outputs"),
        ({"goal": "Re-profile", "expected_outputs": []}, "expected_outputs"),
        ({"goal": "Re-profile", "expected_outputs": "summary"}, "expected_outputs"),
    ],
)
async def test_handoff_refuses_and_names_what_is_missing(
    executor, db_session, test_user, queued, params, names
):
    caller = await _job(db_session, test_user)

    result = await _call(executor, "create_handoff", db_session, caller, params)

    assert "success" not in result
    assert names in result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_handoff_is_refused_at_the_maximum_depth(
    executor, db_session, test_user, queued
):
    caller = await _job(db_session, test_user, chain_depth=3)

    result = await _call(executor, "create_handoff", db_session, caller, HANDOFF)

    assert "success" not in result
    assert "depth" in result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_a_parent_may_hand_off_five_times_and_no_more(
    executor, db_session, test_user, queued
):
    caller = await _job(db_session, test_user)
    state = {}

    for _ in range(5):
        result = await _call(
            executor, "create_handoff", db_session, caller, HANDOFF, state
        )
        assert result.get("success") is True, result
    sixth = await _call(executor, "create_handoff", db_session, caller, HANDOFF, state)

    assert "success" not in sixth
    assert "5" in sixth["error"]
    assert len(await _other_rows(db_session, caller)) == 5
    assert len(queued) == 5
    assert len(state["delegated_subtask_ids"]) == 5


async def test_a_pipeline_stage_that_may_revisit_is_steered_to_the_rerun_tool(
    executor, db_session, test_user, queued
):
    caller = await _job(
        db_session, test_user, config={"may_revisit": ["profile", "scan"]}
    )

    result = await _call(executor, "create_handoff", db_session, caller, HANDOFF)

    assert "success" not in result
    assert "request_stage_rerun" in result["error"]
    assert "profile" in result["error"] and "scan" in result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


# ---------------------------------------------------------------------------
# create_handoff: what is created
# ---------------------------------------------------------------------------


async def test_handoff_creates_a_runnable_child_and_queues_it(
    executor, db_session, test_user, queued
):
    caller = await _job(db_session, test_user, name="Mining stage")
    state = {}

    result = await _call(executor, "create_handoff", db_session, caller, HANDOFF, state)

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
    assert child.goal == HANDOFF["goal"]
    assert child.job_type == "research"
    assert child.name == result["data"]["child_name"]
    assert "Mining stage" in child.name
    assert child.max_iterations == 10
    assert child.config["handoff_contract"] == {
        "from_job_id": str(caller.id),
        "from_job_name": "Mining stage",
        "context": HANDOFF["context"],
        "expected_outputs": HANDOFF["expected_outputs"],
    }
    assert result["data"]["expected_outputs"] == HANDOFF["expected_outputs"]
    assert result["data"]["job_type"] == "research"
    assert result["data"]["max_iterations"] == 10
    # The parent can wait on it, and it counts against the child budget.
    assert state["delegated_subtask_ids"] == [str(child.id)]
    # And a worker was asked to run it, for the same owner.
    assert queued == [((str(child.id), str(test_user.id)), {})]


async def test_the_child_is_shown_its_contract(executor, db_session, test_user):
    caller = await _job(db_session, test_user)
    await _call(executor, "create_handoff", db_session, caller, HANDOFF)
    child = (await _other_rows(db_session, caller))[0]

    prompt = executor._build_thinking_prompt_stable(child, None, {})

    assert "HANDOFF CONTRACT" in prompt
    assert HANDOFF["context"] in prompt
    assert "hot_functions" in prompt and "summary" in prompt


@pytest.mark.parametrize(
    "requested, stored",
    [
        (None, "research"),
        ("analysis", "analysis"),
        ("synthesis", "synthesis"),
        ("custom", "custom"),
        ("coding", "research"),
        ("  analysis  ", "analysis"),
    ],
)
async def test_handoff_runs_the_child_under_a_job_type_it_offers(
    executor, db_session, test_user, requested, stored
):
    caller = await _job(db_session, test_user)
    params = dict(HANDOFF)
    if requested is not None:
        params["job_type"] = requested

    result = await _call(executor, "create_handoff", db_session, caller, params)

    assert result.get("success") is True, result
    assert result["data"]["job_type"] == stored
    assert (await _other_rows(db_session, caller))[0].job_type == stored


async def test_handoff_caps_the_childs_budget(executor, db_session, test_user):
    caller = await _job(
        db_session,
        test_user,
        max_tool_calls=40,
        max_llm_calls=25,
        max_runtime_minutes=12,
    )

    result = await _call(
        executor,
        "create_handoff",
        db_session,
        caller,
        {**HANDOFF, "max_iterations": 500},
    )

    assert result.get("success") is True, result
    assert result["data"]["max_iterations"] == 20
    child = (await _other_rows(db_session, caller))[0]
    assert child.max_iterations == 20
    # Never more than the parent itself was allowed.
    assert child.max_tool_calls == 40
    assert child.max_llm_calls == 25
    assert child.max_runtime_minutes == 12


async def test_handoff_never_creates_a_child_with_a_negative_budget(
    executor, db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _call(
        executor,
        "create_handoff",
        db_session,
        caller,
        {**HANDOFF, "max_iterations": -5},
    )

    children = await _other_rows(db_session, caller)
    assert "error" in result or (
        children[0].max_iterations >= 1
        and children[0].max_tool_calls >= 1
        and children[0].max_llm_calls >= 1
    )


async def test_handoff_refuses_a_max_iterations_that_is_not_a_number(
    executor, db_session, test_user, queued
):
    caller = await _job(db_session, test_user)

    result = await _call(
        executor,
        "create_handoff",
        db_session,
        caller,
        {**HANDOFF, "max_iterations": "a few"},
    )

    assert "success" not in result
    assert result["error"]
    assert await _other_rows(db_session, caller) == []
    assert queued == []


async def test_handoff_trims_an_oversized_contract(executor, db_session, test_user):
    caller = await _job(db_session, test_user)

    result = await _call(
        executor,
        "create_handoff",
        db_session,
        caller,
        {
            "goal": "g" * 5000,
            "context": "c" * 5000,
            "expected_outputs": [f"{i}-" + "o" * 500 for i in range(15)],
        },
    )

    assert result.get("success") is True, result
    child = (await _other_rows(db_session, caller))[0]
    assert child.goal == "g" * 2000
    assert len(child.name) <= 200
    contract = child.config["handoff_contract"]
    assert contract["context"] == "c" * 2000
    assert len(contract["expected_outputs"]) == 10
    assert all(len(output) == 200 for output in contract["expected_outputs"])
    assert result["data"]["expected_outputs"] == contract["expected_outputs"]


async def test_handoff_passes_the_parents_latest_findings_to_the_child(
    executor, db_session, test_user
):
    caller = await _job(db_session, test_user)
    findings = [{"title": f"finding {i}"} for i in range(25)]

    result = await _call(
        executor, "create_handoff", db_session, caller, HANDOFF, {"findings": findings}
    )

    assert result.get("success") is True, result
    assert result["data"]["findings_shared"] is True
    child = (await _other_rows(db_session, caller))[0]
    # Under the key the child's prompt reads.
    assert child.config["inherited_data"]["parent_findings"] == findings[-20:]


async def test_handoff_withholds_the_findings_when_asked_to(
    executor, db_session, test_user
):
    caller = await _job(db_session, test_user)

    result = await _call(
        executor,
        "create_handoff",
        db_session,
        caller,
        {**HANDOFF, "share_findings": False},
        {"findings": [{"title": "private"}]},
    )

    assert result.get("success") is True, result
    assert result["data"]["findings_shared"] is False
    child = (await _other_rows(db_session, caller))[0]
    assert "inherited_data" not in child.config


async def test_the_child_is_told_the_findings_its_parent_shared(
    executor, db_session, test_user
):
    caller = await _job(db_session, test_user)
    await _call(
        executor,
        "create_handoff",
        db_session,
        caller,
        HANDOFF,
        {"findings": [{"title": "ZEBRA-FINDING: profile is per-module"}]},
    )
    child = (await _other_rows(db_session, caller))[0]

    prompt = executor._build_thinking_prompt_stable(child, None, {})

    assert "ZEBRA-FINDING" in prompt


async def test_handoff_keeps_the_child_in_the_parents_source_scope(
    executor, db_session, test_user
):
    scoped = await _job(db_session, test_user, config={"source_id": "repo-42"})
    unscoped = await _job(db_session, test_user, name="Unscoped")

    await _call(executor, "create_handoff", db_session, scoped, HANDOFF)
    await _call(executor, "create_handoff", db_session, unscoped, HANDOFF)

    children = {
        row.parent_job_id: row
        for row in await _other_rows(db_session, scoped, unscoped)
    }
    assert children[scoped.id].config["default_source_id"] == "repo-42"
    assert "default_source_id" not in children[unscoped.id].config


async def test_a_grandchild_handoff_records_its_depth_and_the_root(
    executor, db_session, test_user
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

    result = await _call(executor, "create_handoff", db_session, caller, HANDOFF)

    assert result.get("success") is True, result
    child = (await _other_rows(db_session, root, caller))[0]
    assert child.chain_depth == 3
    assert child.parent_job_id == caller.id
    assert child.root_job_id == root.id


async def test_a_failed_handoff_leaves_the_session_usable(
    executor, db_session, test_user, queued
):
    caller = await _job(db_session, test_user)
    state = {}

    with _inserts_fail():
        result = await _call(
            executor, "create_handoff", db_session, caller, HANDOFF, state
        )

    assert "success" not in result
    assert "Failed to create handoff" in result["error"]
    assert queued == []
    assert not state.get("delegated_subtask_ids")
    # The executor goes on using this session and this job object.
    assert caller.name == "Calling job"
    assert await _other_rows(db_session, caller) == []
    retry = await _call(executor, "create_handoff", db_session, caller, HANDOFF, state)
    assert retry.get("success") is True, retry
    assert len(await _other_rows(db_session, caller)) == 1


# ---------------------------------------------------------------------------
# get_sibling_status
# ---------------------------------------------------------------------------


async def test_sibling_status_is_refused_for_a_job_with_no_parent(
    executor, db_session, test_user
):
    caller = await _job(db_session, test_user)
    await _job(db_session, test_user, name="Unrelated")

    result = await _call(executor, "get_sibling_status", db_session, caller, {})

    assert "success" not in result
    assert "no parent" in result["error"]


async def test_sibling_status_lists_the_siblings_and_nobody_else(
    executor, db_session, test_user, admin_user
):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)
    sibling_a.status = "completed"
    sibling_a.iteration = 7
    sibling_a.max_iterations = 12
    sibling_a.job_type = "analysis"
    await db_session.commit()
    await _job(db_session, test_user, name="Unrelated")
    await _job(db_session, test_user, name="Nephew", parent_job_id=sibling_a.id)
    await _job(db_session, admin_user, name="Another tenant", parent_job_id=parent.id)

    result = await _call(executor, "get_sibling_status", db_session, caller, {})

    assert result.get("success") is True, result
    assert result["data"]["count"] == 2
    by_id = {entry["job_id"]: entry for entry in result["data"]["siblings"]}
    assert set(by_id) == {str(sibling_a.id), str(sibling_b.id)}
    assert by_id[str(sibling_a.id)] == {
        "job_id": str(sibling_a.id),
        "name": "Sibling A",
        "job_type": "analysis",
        "status": "completed",
        "iteration": 7,
        "max_iterations": 12,
    }
    assert by_id[str(sibling_b.id)]["status"] == "running"


async def test_an_only_child_has_no_siblings(executor, db_session, test_user):
    parent = await _job(db_session, test_user, name="Parent")
    caller = await _job(db_session, test_user, parent_job_id=parent.id)

    result = await _call(executor, "get_sibling_status", db_session, caller, {})

    assert result.get("success") is True, result
    assert result["data"] == {"siblings": [], "count": 0}


async def test_sibling_findings_are_left_out_unless_asked_for(
    executor, db_session, test_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    sibling_a.results = {"findings": [{"title": "Stride is 2.1x"}]}
    await db_session.commit()

    result = await _call(executor, "get_sibling_status", db_session, caller, {})

    for entry in result["data"]["siblings"]:
        assert "findings_count" not in entry
        assert "finding_titles" not in entry


async def test_sibling_status_can_include_finding_titles(
    executor, db_session, test_user
):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)
    sibling_a.results = {
        "findings": [{"title": f"{i}-" + "t" * 300} for i in range(14)] + ["not a dict"]
    }
    await db_session.commit()

    result = await _call(
        executor, "get_sibling_status", db_session, caller, {"include_findings": True}
    )

    assert result.get("success") is True, result
    by_id = {entry["job_id"]: entry for entry in result["data"]["siblings"]}
    with_findings = by_id[str(sibling_a.id)]
    assert with_findings["findings_count"] == 15
    assert len(with_findings["finding_titles"]) == 10
    assert with_findings["finding_titles"][0].startswith("0-")
    assert all(len(title) == 100 for title in with_findings["finding_titles"])
    # A sibling with nothing recorded is still listed.
    assert by_id[str(sibling_b.id)]["status"] == "running"
    assert not by_id[str(sibling_b.id)].get("finding_titles")


async def test_sibling_status_changes_nothing(executor, db_session, test_user):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)

    await _call(
        executor, "get_sibling_status", db_session, caller, {"include_findings": True}
    )

    assert await _stored(db_session, sibling_a) == {}
    assert await _other_rows(db_session, parent, caller, sibling_a, _b) == []


# ---------------------------------------------------------------------------
# broadcast_to_siblings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"message": "   "}, {"category": "status"}])
async def test_broadcast_refuses_a_call_with_no_message(
    executor, db_session, test_user, params
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)

    result = await _call(executor, "broadcast_to_siblings", db_session, caller, params)

    assert "success" not in result
    assert "message" in result["error"]
    assert "agent_messages" not in await _stored(db_session, sibling_a)


async def test_broadcast_is_refused_for_a_job_with_no_parent(
    executor, db_session, test_user
):
    caller = await _job(db_session, test_user)
    unrelated = await _job(db_session, test_user, name="Unrelated")

    result = await _call(
        executor, "broadcast_to_siblings", db_session, caller, {"message": "hello"}
    )

    assert "success" not in result
    assert "no parent" in result["error"]
    assert "agent_messages" not in await _stored(db_session, unrelated)


async def test_broadcast_reaches_every_sibling_and_nobody_else(
    executor, db_session, test_user, admin_user
):
    parent, (caller, sibling_a, sibling_b) = await _family(db_session, test_user)
    unrelated = await _job(db_session, test_user, name="Unrelated")
    nephew = await _job(db_session, test_user, name="Nephew", parent_job_id=caller.id)
    theirs = await _job(
        db_session, admin_user, name="Another tenant", parent_job_id=parent.id
    )

    result = await _call(
        executor,
        "broadcast_to_siblings",
        db_session,
        caller,
        {"message": "  the gem5 image is stale, rebuild first  "},
    )

    assert result.get("success") is True, result
    assert result["data"]["recipients"] == 2
    for sibling in (sibling_a, sibling_b):
        inbox = (await _stored(db_session, sibling))["agent_messages"]
        assert len(inbox) == 1
        assert inbox[0]["from_job_id"] == str(caller.id)
        assert inbox[0]["from_job_name"] == "Caller"
        assert inbox[0]["message"] == "the gem5 image is stale, rebuild first"
        assert inbox[0]["category"] == "broadcast"
        assert inbox[0]["broadcast"] is True
        assert inbox[0]["sent_at"]
    for bystander in (caller, parent, unrelated, nephew, theirs):
        assert "agent_messages" not in await _stored(db_session, bystander)


async def test_a_sibling_reads_the_broadcast_in_its_inbox(
    executor, db_session, test_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    await _call(
        executor,
        "broadcast_to_siblings",
        db_session,
        caller,
        {"message": "arXiv is answering 406", "category": "outage"},
    )
    await db_session.commit()
    await db_session.refresh(sibling_a)

    result = await _read_messages(executor, db_session, sibling_a)

    assert result.get("success") is True, result
    assert result["data"]["total"] == 1
    assert result["data"]["messages"][0]["message"] == "arXiv is answering 406"
    assert result["data"]["messages"][0]["category"] == "outage"
    assert result["data"]["messages"][0]["from_job_id"] == str(caller.id)


async def test_broadcast_with_no_siblings_delivers_to_nobody(
    executor, db_session, test_user
):
    parent = await _job(db_session, test_user, name="Parent")
    caller = await _job(db_session, test_user, parent_job_id=parent.id)

    result = await _call(
        executor, "broadcast_to_siblings", db_session, caller, {"message": "anyone?"}
    )

    assert result.get("success") is True, result
    assert result["data"]["recipients"] == 0
    assert "agent_messages" not in await _stored(db_session, parent)


async def test_broadcast_keeps_what_a_sibling_already_recorded(
    executor, db_session, test_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    sibling_a.results = {
        "findings": [{"title": "its own"}],
        "agent_messages": [{"from_job_id": "earlier", "message": "first"}],
    }
    await db_session.commit()

    result = await _call(
        executor, "broadcast_to_siblings", db_session, caller, {"message": "second"}
    )

    assert result.get("success") is True, result
    stored = await _stored(db_session, sibling_a)
    assert stored["findings"] == [{"title": "its own"}]
    assert [m["message"] for m in stored["agent_messages"]] == ["first", "second"]


async def test_broadcast_trims_a_long_message_and_category(
    executor, db_session, test_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)

    result = await _call(
        executor,
        "broadcast_to_siblings",
        db_session,
        caller,
        {"message": "m" * 5000, "category": "c" * 500},
    )

    assert result.get("success") is True, result
    entry = (await _stored(db_session, sibling_a))["agent_messages"][0]
    assert entry["message"] == "m" * 2000
    assert entry["category"] == "c" * 100


async def test_a_siblings_inbox_keeps_only_the_newest_hundred(
    executor, db_session, test_user
):
    parent, (caller, sibling_a, _b) = await _family(db_session, test_user)
    sibling_a.results = {
        "agent_messages": [
            {"from_job_id": "old", "message": f"old-{i}"} for i in range(99)
        ]
    }
    await db_session.commit()

    for text in ("new-0", "new-1", "new-2"):
        result = await _call(
            executor, "broadcast_to_siblings", db_session, caller, {"message": text}
        )
        assert result.get("success") is True, result

    inbox = (await _stored(db_session, sibling_a))["agent_messages"]
    assert len(inbox) == 100
    assert inbox[0]["message"] == "old-2"
    assert [m["message"] for m in inbox[-3:]] == ["new-0", "new-1", "new-2"]


async def test_a_missing_job_id_is_not_mistaken_for_a_sibling(
    executor, db_session, test_user
):
    """Two parentless jobs share parent_job_id NULL; that is not a family."""
    caller = await _job(db_session, test_user, parent_job_id=None)
    other_root = await _job(db_session, test_user, name="Another root")
    assert other_root.parent_job_id is None and uuid4() != other_root.id

    status = await _call(executor, "get_sibling_status", db_session, caller, {})
    broadcast = await _call(
        executor, "broadcast_to_siblings", db_session, caller, {"message": "hello"}
    )

    assert "success" not in status
    assert "success" not in broadcast
    assert "agent_messages" not in await _stored(db_session, other_root)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


class TestCollaborationSchemas:
    """Tests for collaboration tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "create_handoff" in names
        assert "get_sibling_status" in names
        assert "broadcast_to_siblings" in names

    def test_create_handoff_requires_goal_and_outputs(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_handoff")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "goal" in required
        assert "expected_outputs" in required

    def test_create_handoff_has_job_type_enum(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_handoff")
        job_type_prop = tool["parameters"]["properties"]["job_type"]
        assert "enum" in job_type_prop
        assert "research" in job_type_prop["enum"]

    def test_get_sibling_status_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_sibling_status")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_broadcast_requires_message(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("broadcast_to_siblings")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "message" in required


class TestCollaborationRegistry:
    """Tests for collaboration tool registry classification."""

    def test_create_handoff_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("create_handoff")
        assert meta is not None
        assert meta.effects == "write"

    def test_create_handoff_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("create_handoff")
        assert meta is not None
        assert meta.cost_tier == "medium"

    def test_get_sibling_status_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_sibling_status")
        assert meta is not None
        assert meta.effects == "read"

    def test_broadcast_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("broadcast_to_siblings")
        assert meta is not None
        assert meta.effects == "write"

    def test_get_sibling_status_is_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_sibling_status")
        assert meta is not None
        assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in [
            "create_handoff",
            "get_sibling_status",
            "broadcast_to_siblings",
        ]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
