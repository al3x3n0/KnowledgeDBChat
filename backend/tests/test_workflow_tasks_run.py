"""The workflow Celery tasks, run for real.

`app/tasks/workflow_tasks.py` was missing from the worker's `include` list
until 2735dbf and its scheduled trigger was never in the beat schedule, so none
of these bodies had ever executed. Here they run against the in-memory test
database: the task opens its own session through `create_celery_session`,
which is pointed at the test engine, and the sync task functions are called
through `.run(...)` on a worker thread, where `run_async` builds its own event
loop exactly as it does in a Celery child.

Only edges that leave the process are replaced -- the engine's Redis progress
feed and the Celery `delay` -- and each replacement binds its arguments against
the real callee, so a call the real thing would refuse fails here too.
"""

import asyncio
import inspect
import pathlib
import re
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.workflow import (
    UserTool,
    Workflow,
    WorkflowEdge,
    WorkflowExecution,
    WorkflowNode,
    WorkflowNodeExecution,
)
from app.services.workflow_engine import WorkflowEngine
from app.tasks import workflow_tasks
from tests.conftest import test_engine

pytestmark = pytest.mark.unit

APP_DIR = pathlib.Path(workflow_tasks.__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def task_session(monkeypatch):
    """The task's own session, bound to the engine the test's rows live in.

    Same options as the real `create_celery_session` (expire_on_commit=False),
    because the engine's behaviour between commits depends on it.
    """
    opened = []

    def _factory():
        opened.append(1)
        return async_sessionmaker(
            test_engine, class_=AsyncSession, expire_on_commit=False
        )

    assert inspect.signature(workflow_tasks.create_celery_session).parameters == {}
    monkeypatch.setattr(workflow_tasks, "create_celery_session", _factory)
    return opened


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    """The engine's progress feed is its one external edge."""
    inspect.signature(WorkflowEngine._get_redis).bind(object())

    async def _none(self):
        return None

    monkeypatch.setattr(WorkflowEngine, "_get_redis", _none)


@pytest.fixture(autouse=True)
def queued(monkeypatch):
    """Record what is handed to the Celery workflow task instead of sending."""
    calls = []
    real_run = workflow_tasks.execute_workflow_task.run

    def _delay(*args, **kwargs):
        bound = inspect.signature(real_run).bind(*args, **kwargs)
        calls.append(bound.arguments["execution_id"])

    monkeypatch.setattr(workflow_tasks.execute_workflow_task, "delay", _delay)
    return calls


class _Clock(datetime):
    """`datetime` with a `utcnow` the test decides."""

    frozen: datetime = None

    @classmethod
    def utcnow(cls):
        return cls.frozen


@pytest.fixture
def clock(monkeypatch):
    def _set(moment):
        _Clock.frozen = moment

    monkeypatch.setattr(workflow_tasks, "datetime", _Clock)
    yield _set
    _Clock.frozen = None


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


async def _workflow(db, user, name="Nightly report", nodes=None, edges=None, **kw):
    """A stored workflow. `nodes` is a list of WorkflowNode kwargs."""
    workflow = Workflow(user_id=user.id, name=name, **kw)
    db.add(workflow)
    await db.flush()
    nodes = nodes or [
        {"node_id": "start", "node_type": "start"},
        {"node_id": "end", "node_type": "end"},
    ]
    for spec in nodes:
        db.add(WorkflowNode(workflow_id=workflow.id, **{"config": {}, **spec}))
    ids = [spec["node_id"] for spec in nodes]
    for source, target in edges or list(zip(ids, ids[1:])):
        db.add(
            WorkflowEdge(
                workflow_id=workflow.id, source_node_id=source, target_node_id=target
            )
        )
    await db.commit()
    await db.refresh(workflow)
    return workflow


async def _transform_tool(db, user, name, config):
    tool = UserTool(user_id=user.id, name=name, tool_type="transform", config=config)
    db.add(tool)
    await db.commit()
    await db.refresh(tool)
    return tool


async def _execution(db, user, workflow, **fields):
    values = {
        "workflow_id": workflow.id,
        "user_id": user.id,
        "trigger_type": "manual",
        "trigger_data": {},
        "status": "pending",
        "progress": 0,
        "context": {},
    }
    values.update(fields)
    execution = WorkflowExecution(**values)
    db.add(execution)
    await db.commit()
    await db.refresh(execution)
    return execution


async def _executions(db, workflow=None):
    query = select(WorkflowExecution).execution_options(populate_existing=True)
    if workflow is not None:
        query = query.where(WorkflowExecution.workflow_id == workflow.id)
    return (await db.execute(query)).scalars().all()


async def _node_runs(db, execution):
    query = (
        select(WorkflowNodeExecution)
        .where(WorkflowNodeExecution.execution_id == execution.id)
        .execution_options(populate_existing=True)
    )
    return (await db.execute(query)).scalars().all()


async def _run_task(task, *args):
    """Call a sync task body from a sync context, as a worker child would."""
    return await asyncio.to_thread(task.run, *args)


async def _greeting_workflow(db, user, mapping="{{who}}", **kw):
    """start -> greet (a real transform tool reading the context) -> end."""
    tool = await _transform_tool(
        db,
        user,
        f"greeter-{uuid4().hex[:6]}",
        {"transform_type": "jinja2", "template": '{"greeting": "hello {{ name }}"}'},
    )
    return await _workflow(
        db,
        user,
        nodes=[
            {"node_id": "start", "node_type": "start"},
            {
                "node_id": "greet",
                "node_type": "tool",
                "tool_id": tool.id,
                "config": {"input_mapping": {"name": mapping}},
            },
            {"node_id": "end", "node_type": "end"},
        ],
        **kw,
    )


# ---------------------------------------------------------------------------
# execute_workflow_task
# ---------------------------------------------------------------------------


async def test_pending_execution_runs_to_completed_with_outputs_stored(
    db_session, test_user
):
    workflow = await _greeting_workflow(db_session, test_user)
    execution = await _execution(
        db_session, test_user, workflow, context={"who": "Ada"}
    )

    await _run_task(workflow_tasks.execute_workflow_task, str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == "completed", stored.error
    assert stored.progress == 100
    assert stored.error is None
    assert stored.started_at is not None and stored.completed_at is not None
    # The context in the database, not the engine's in-memory copy.
    assert stored.context["who"] == "Ada"
    assert stored.context["greet"] == {"greeting": "hello Ada"}
    assert stored.context["start"] == {"started": True}
    assert stored.context["end"] == {"ended": True}
    runs = {r.node_id: r for r in await _node_runs(db_session, stored)}
    assert set(runs) == {"start", "greet", "end"}
    assert all(r.status == "completed" for r in runs.values())
    assert runs["greet"].output_data == {"greeting": "hello Ada"}


async def test_async_body_runs_the_same_execution(db_session, test_user):
    workflow = await _greeting_workflow(db_session, test_user)
    execution = await _execution(
        db_session, test_user, workflow, context={"who": "Grace"}
    )

    await workflow_tasks._execute_workflow_async(str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == "completed"
    assert stored.context["greet"] == {"greeting": "hello Grace"}


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "running"])
async def test_non_pending_execution_is_skipped(db_session, test_user, status):
    workflow = await _greeting_workflow(db_session, test_user)
    execution = await _execution(
        db_session, test_user, workflow, status=status, context={"who": "Ada"}
    )

    await _run_task(workflow_tasks.execute_workflow_task, str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == status
    assert "greet" not in stored.context
    assert await _node_runs(db_session, stored) == []


async def test_missing_execution_raises(db_session, test_user):
    missing = str(uuid4())

    with pytest.raises(ValueError, match=f"Execution {missing} not found"):
        await _run_task(workflow_tasks.execute_workflow_task, missing)

    assert await _executions(db_session) == []


async def test_failing_node_leaves_execution_failed_with_its_error(
    db_session, test_user
):
    tool = await _transform_tool(
        db_session, test_user, "js", {"transform_type": "javascript", "template": "x"}
    )
    workflow = await _workflow(
        db_session,
        test_user,
        nodes=[
            {"node_id": "start", "node_type": "start"},
            {"node_id": "broken", "node_type": "tool", "tool_id": tool.id},
            {"node_id": "end", "node_type": "end"},
        ],
    )
    execution = await _execution(db_session, test_user, workflow)

    # The task re-raises so Celery records the failure too.
    with pytest.raises(Exception, match="JavaScript transforms are not yet supported"):
        await _run_task(workflow_tasks.execute_workflow_task, str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == "failed"
    assert "Node broken failed" in stored.error
    assert "JavaScript transforms are not yet supported" in stored.error
    assert stored.completed_at is not None
    runs = {r.node_id: r for r in await _node_runs(db_session, stored)}
    assert runs["start"].status == "completed"
    assert runs["broken"].status == "failed"
    assert "end" not in runs


async def test_failure_outside_the_engine_is_recorded_by_the_task(
    db_session, test_user, monkeypatch
):
    """A failure the engine never sees still leaves the row failed."""
    workflow = await _greeting_workflow(db_session, test_user)
    execution = await _execution(db_session, test_user, workflow)
    real = WorkflowEngine.execute_existing_execution

    async def _explode(self, execution):
        inspect.signature(real).bind(self, execution)
        raise RuntimeError("worker lost its database")

    monkeypatch.setattr(WorkflowEngine, "execute_existing_execution", _explode)

    with pytest.raises(RuntimeError, match="worker lost its database"):
        await _run_task(workflow_tasks.execute_workflow_task, str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == "failed"
    assert stored.error == "worker lost its database"
    assert stored.completed_at is not None


async def test_mark_failed_leaves_a_finished_execution_alone(db_session, test_user):
    workflow = await _workflow(db_session, test_user)
    execution = await _execution(db_session, test_user, workflow, status="completed")

    await workflow_tasks._mark_execution_failed(str(execution.id), "late error")

    [stored] = await _executions(db_session)
    assert stored.status == "completed"
    assert stored.error is None


async def test_queue_workflow_hands_the_task_an_execution_it_can_run(
    db_session, test_user, queued
):
    workflow = await _greeting_workflow(db_session, test_user)
    engine = WorkflowEngine(db_session, test_user)

    execution = await engine.queue_workflow(
        workflow.id, initial_context={"who": "Linus"}
    )

    assert queued == [str(execution.id)]
    await _run_task(workflow_tasks.execute_workflow_task, queued[0])
    [stored] = await _executions(db_session)
    assert stored.status == "completed"
    assert stored.context["greet"] == {"greeting": "hello Linus"}


async def test_documented_context_syntax_reaches_the_node(db_session, test_user):
    workflow = await _greeting_workflow(
        db_session, test_user, mapping="{{context.who}}"
    )
    execution = await _execution(
        db_session, test_user, workflow, context={"who": "Ada"}
    )

    await _run_task(workflow_tasks.execute_workflow_task, str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == "completed"
    assert stored.context["greet"] == {"greeting": "hello Ada"}


async def test_what_started_the_run_is_readable_as_context_trigger_data(
    db_session, test_user
):
    # The shipped templates read `{{context.trigger_data.topic}}`. Only a
    # sub-workflow's context carried trigger_data; a top-level run's did not.
    workflow = await _greeting_workflow(
        db_session, test_user, mapping="{{context.trigger_data.who}}"
    )
    execution = await _execution(
        db_session, test_user, workflow, trigger_data={"who": "Grace"}
    )

    await _run_task(workflow_tasks.execute_workflow_task, str(execution.id))

    [stored] = await _executions(db_session)
    assert stored.status == "completed"
    assert stored.context["greet"] == {"greeting": "hello Grace"}
    assert stored.context["trigger_data"] == {"who": "Grace"}


# ---------------------------------------------------------------------------
# trigger_scheduled_workflows
# ---------------------------------------------------------------------------


def _quiet_moment():
    """The real current time, away from a minute boundary.

    The dedup check compares the row's `created_at` -- the database's own
    clock -- against the cron tick computed from the task's clock, so a test
    of "a second call in the same minute" has to run the task at the real
    time. Away from second 0 because SQLite's CURRENT_TIMESTAMP has whole
    seconds and would equal the tick.
    """
    import time

    now = datetime.utcnow()
    while now.second < 2 or now.second > 55:
        time.sleep(0.5)
        now = datetime.utcnow()
    return now


def test_schedule_query_is_valid_for_the_postgres_json_column():
    """`trigger_config` is postgres JSON (not JSONB); `->>` exists for both."""
    column_type = Workflow.__table__.c.trigger_config.type
    assert type(column_type) is postgresql.JSON
    query = select(Workflow.id).where(
        Workflow.trigger_config["type"].astext == "schedule"
    )
    sql = str(query.compile(dialect=postgresql.dialect()))
    assert re.search(r"workflows\.trigger_config ->> %\(trigger_config_1\)s", sql), sql


async def test_schedule_query_evaluates_on_the_test_database(db_session, test_user):
    """Without this the scheduled-trigger tests would judge nothing."""
    await _workflow(
        db_session, test_user, name="s", trigger_config={"type": "schedule"}
    )
    await _workflow(db_session, test_user, name="m", trigger_config={"type": "manual"})
    await _workflow(db_session, test_user, name="n", trigger_config=None)

    rows = await db_session.execute(
        select(Workflow.name).where(
            Workflow.trigger_config["type"].astext == "schedule"
        )
    )
    assert rows.scalars().all() == ["s"]


async def test_tick_in_the_last_minute_fires_exactly_once(
    db_session, test_user, queued, clock
):
    now = _quiet_moment()
    clock(now)
    workflow = await _greeting_workflow(
        db_session,
        test_user,
        trigger_config={"type": "schedule", "schedule": "* * * * *"},
    )

    await workflow_tasks._trigger_scheduled_workflows_async()

    [execution] = await _executions(db_session, workflow)
    tick = now.replace(second=0, microsecond=0)
    assert execution.trigger_type == "schedule"
    assert execution.status == "pending"
    assert execution.user_id == test_user.id
    assert execution.trigger_data == {
        "schedule": "* * * * *",
        "scheduled_time": tick.isoformat(),
    }
    assert queued == [str(execution.id)]

    # The beat fires again in the same minute (or the task is redelivered).
    await workflow_tasks._trigger_scheduled_workflows_async()

    assert len(await _executions(db_session, workflow)) == 1
    assert queued == [str(execution.id)]


async def test_sync_task_fires_through_its_own_event_loop(
    db_session, test_user, queued, clock
):
    clock(_quiet_moment())
    workflow = await _workflow(
        db_session,
        test_user,
        trigger_config={"type": "schedule", "schedule": "* * * * *"},
    )

    await _run_task(workflow_tasks.trigger_scheduled_workflows)

    [execution] = await _executions(db_session, workflow)
    assert queued == [str(execution.id)]


async def test_daily_tick_fires_only_in_its_minute(
    db_session, test_user, queued, clock
):
    workflow = await _workflow(
        db_session,
        test_user,
        trigger_config={"type": "schedule", "schedule": "0 6 * * *"},
    )
    day = datetime(2031, 3, 4)

    clock(day.replace(hour=5, minute=59, second=59))
    await workflow_tasks._trigger_scheduled_workflows_async()
    assert await _executions(db_session) == []

    clock(day.replace(hour=6, minute=0, second=30))
    await workflow_tasks._trigger_scheduled_workflows_async()
    [execution] = await _executions(db_session)
    assert execution.workflow_id == workflow.id
    assert execution.trigger_data["scheduled_time"] == "2031-03-04T06:00:00"
    assert queued == [str(execution.id)]


@pytest.mark.parametrize("late", [timedelta(seconds=61), timedelta(hours=5)])
async def test_tick_more_than_a_minute_ago_is_not_replayed(
    db_session, test_user, queued, clock, late
):
    await _workflow(
        db_session,
        test_user,
        trigger_config={"type": "schedule", "schedule": "0 6 * * *"},
    )
    clock(datetime(2031, 3, 4, 6, 0) + late)

    await workflow_tasks._trigger_scheduled_workflows_async()

    assert await _executions(db_session) == []
    assert queued == []


async def test_inactive_and_non_schedule_workflows_are_ignored(
    db_session, test_user, queued, clock
):
    clock(datetime(2031, 3, 4, 6, 0, 10))
    every_minute = {"schedule": "* * * * *"}
    await _workflow(
        db_session,
        test_user,
        name="off",
        is_active=False,
        trigger_config={"type": "schedule", **every_minute},
    )
    for kind in ("manual", "event", "webhook"):
        await _workflow(
            db_session,
            test_user,
            name=kind,
            trigger_config={"type": kind, **every_minute},
        )
    await _workflow(db_session, test_user, name="nothing", trigger_config=None)
    await _workflow(
        db_session, test_user, name="no-cron", trigger_config={"type": "schedule"}
    )

    await workflow_tasks._trigger_scheduled_workflows_async()

    assert await _executions(db_session) == []
    assert queued == []


async def test_invalid_cron_on_one_workflow_does_not_stop_the_others(
    db_session, test_user, queued, clock
):
    clock(datetime(2031, 3, 4, 6, 0, 10))
    bad = await _workflow(
        db_session,
        test_user,
        name="bad",
        trigger_config={"type": "schedule", "schedule": "every tuesday"},
    )
    good = await _workflow(
        db_session,
        test_user,
        name="good",
        trigger_config={"type": "schedule", "schedule": "0 6 * * *"},
    )

    await workflow_tasks._trigger_scheduled_workflows_async()

    assert await _executions(db_session, bad) == []
    [execution] = await _executions(db_session, good)
    assert queued == [str(execution.id)]


async def test_a_manual_run_in_the_same_minute_does_not_suppress_the_schedule(
    db_session, test_user, queued, clock
):
    now = _quiet_moment()
    clock(now)
    workflow = await _workflow(
        db_session,
        test_user,
        trigger_config={"type": "schedule", "schedule": "* * * * *"},
    )
    await _execution(db_session, test_user, workflow, trigger_type="manual")

    await workflow_tasks._trigger_scheduled_workflows_async()

    kinds = sorted(e.trigger_type for e in await _executions(db_session, workflow))
    assert kinds == ["manual", "schedule"]
    assert len(queued) == 1


async def test_scheduled_execution_then_runs_through_the_task(
    db_session, test_user, queued, clock
):
    clock(_quiet_moment())
    workflow = await _greeting_workflow(
        db_session,
        test_user,
        trigger_config={"type": "schedule", "schedule": "* * * * *"},
    )

    await workflow_tasks._trigger_scheduled_workflows_async()
    clock(None)
    await _run_task(workflow_tasks.execute_workflow_task, queued[0])

    [execution] = await _executions(db_session, workflow)
    assert execution.status == "completed", execution.error
    assert execution.trigger_type == "schedule"
    assert set(execution.context) >= {"start", "greet", "end"}


# ---------------------------------------------------------------------------
# trigger_event_workflow
# ---------------------------------------------------------------------------


async def test_event_fires_the_users_matching_active_workflows(
    db_session, test_user, admin_user, queued
):
    on_upload = {"type": "event", "event": "document.uploaded"}
    mine = await _workflow(db_session, test_user, name="mine", trigger_config=on_upload)
    await _workflow(db_session, admin_user, name="theirs", trigger_config=on_upload)
    await _workflow(
        db_session, test_user, name="off", is_active=False, trigger_config=on_upload
    )
    await _workflow(
        db_session,
        test_user,
        name="other-event",
        trigger_config={"type": "event", "event": "document.deleted"},
    )
    await _workflow(
        db_session,
        test_user,
        name="schedule-named-like-event",
        trigger_config={"type": "schedule", "event": "document.uploaded"},
    )
    data = {"document_id": "d-1", "title": "Spec"}

    await workflow_tasks._trigger_event_workflow_async(
        "document.uploaded", data, str(test_user.id)
    )

    [execution] = await _executions(db_session)
    assert execution.workflow_id == mine.id
    assert execution.user_id == test_user.id
    assert execution.trigger_type == "event"
    assert execution.status == "pending"
    assert execution.trigger_data == {"event": "document.uploaded", "event_data": data}
    assert execution.context == {"event": data}
    assert queued == [str(execution.id)]


async def test_event_payload_reaches_the_workflows_nodes(db_session, test_user, queued):
    tool = await _transform_tool(
        db_session,
        test_user,
        "titler",
        {"transform_type": "jinja2", "template": '{"seen": "{{ title }}"}'},
    )
    workflow = await _workflow(
        db_session,
        test_user,
        trigger_config={"type": "event", "event": "document.uploaded"},
        nodes=[
            {"node_id": "start", "node_type": "start"},
            {
                "node_id": "read",
                "node_type": "tool",
                "tool_id": tool.id,
                "config": {"input_mapping": {"title": "{{event.title}}"}},
            },
            {"node_id": "end", "node_type": "end"},
        ],
    )

    await _run_task(
        workflow_tasks.trigger_event_workflow,
        "document.uploaded",
        {"title": "Spec"},
        str(test_user.id),
    )
    assert len(queued) == 1
    await _run_task(workflow_tasks.execute_workflow_task, queued[0])

    [execution] = await _executions(db_session, workflow)
    assert execution.status == "completed", execution.error
    assert execution.context["read"] == {"seen": "Spec"}


async def test_event_with_no_matching_workflow_creates_nothing(
    db_session, test_user, queued
):
    await _workflow(db_session, test_user, trigger_config={"type": "manual"})

    await workflow_tasks._trigger_event_workflow_async(
        "document.uploaded", {}, str(test_user.id)
    )

    assert await _executions(db_session) == []
    assert queued == []


def test_publish_workflow_event_sends_the_task_its_arguments(monkeypatch):
    sent = []
    real_run = workflow_tasks.trigger_event_workflow.run

    def _delay(*args, **kwargs):
        sent.append(inspect.signature(real_run).bind(*args, **kwargs).arguments)

    monkeypatch.setattr(workflow_tasks.trigger_event_workflow, "delay", _delay)

    workflow_tasks.publish_workflow_event("document.uploaded", {"a": 1}, "u-1")

    assert sent == [
        {"event_name": "document.uploaded", "event_data": {"a": 1}, "user_id": "u-1"}
    ]


def test_some_application_code_publishes_workflow_events():
    this_module = pathlib.Path(workflow_tasks.__file__).resolve()
    publishers = [
        path
        for path in APP_DIR.rglob("*.py")
        if path.resolve() != this_module
        and re.search(
            r"publish_workflow_event\s*\(|trigger_event_workflow\.(delay|apply_async)"
            r"|app\.tasks\.workflow_tasks\.trigger_event_workflow",
            path.read_text(encoding="utf-8"),
        )
    ]
    assert publishers, "nothing publishes a workflow event"
