"""`list_available_workflows`, `execute_workflow` and `get_workflow_status`.

Called through the real autonomous handlers against the in-memory database,
and judged on the rows involved: a listing is the caller's `workflows` rows, a
launch is a `workflow_executions` row the engine really ran, and a status is
what that row says. Only Redis (the engine's progress feed) and the Celery
`delay` are replaced.
"""

import inspect
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJob
from app.models.workflow import Workflow, WorkflowEdge, WorkflowExecution, WorkflowNode
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_workflow_provider,
)
from app.services.workflow_engine import WorkflowEngine
from app.tasks import workflow_tasks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    """The engine's progress feed is the one external edge it has."""

    async def _none(self):
        return None

    monkeypatch.setattr(WorkflowEngine, "_get_redis", _none)


@pytest.fixture
def queued(monkeypatch):
    """Record anything handed to the Celery workflow task instead of sending."""
    calls = []

    def _delay(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(workflow_tasks.execute_workflow_task, "delay", _delay)
    monkeypatch.setattr(
        workflow_tasks.execute_workflow_task,
        "apply_async",
        lambda args=(), kwargs=None, **_: calls.append((tuple(args), kwargs or {})),
    )
    return calls


def _handler(tool_name):
    provider = build_autonomous_workflow_provider(SimpleNamespace())
    return provider._handlers[tool_name]


async def _job(db, user):
    job = AgentJob(
        name="Calling job",
        goal="Run the nightly report workflow",
        job_type="research",
        user_id=user.id,
        status="running",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _call(tool_name, db, user, params):
    job = await _job(db, user)
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(user.id),
        job=job,
        state={},
    )
    return job, await _handler(tool_name)(params, ctx)


async def _workflow(db, user, name="Nightly report", nodes=("start", "end"), **fields):
    """A stored workflow whose nodes are joined in the order given."""
    workflow = Workflow(user_id=user.id, name=name, **fields)
    db.add(workflow)
    await db.flush()
    for node_type in nodes:
        db.add(
            WorkflowNode(
                workflow_id=workflow.id,
                node_id=node_type,
                node_type=node_type,
                config={},
            )
        )
    for source, target in zip(nodes, nodes[1:]):
        db.add(
            WorkflowEdge(
                workflow_id=workflow.id,
                source_node_id=source,
                target_node_id=target,
            )
        )
    await db.commit()
    await db.refresh(workflow)
    return workflow


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


async def _executions(db):
    query = select(WorkflowExecution).execution_options(populate_existing=True)
    return (await db.execute(query)).scalars().all()


def _refused(result):
    return isinstance(result, dict) and bool(result.get("error"))


class TestListAvailableWorkflows:
    async def test_lists_only_the_callers_active_workflows(
        self, db_session, test_user, admin_user
    ):
        mine = await _workflow(db_session, test_user, description="ETL")
        await _workflow(db_session, test_user, name="Retired", is_active=False)
        await _workflow(db_session, admin_user, name="Someone else's")

        _, result = await _call("list_available_workflows", db_session, test_user, {})

        assert result["success"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["workflows"] == [
            {
                "id": str(mine.id),
                "name": "Nightly report",
                "description": "ETL",
                "is_active": True,
                "node_count": 2,
            }
        ]

    async def test_inactive_filter_returns_only_the_callers_inactive(
        self, db_session, test_user, admin_user
    ):
        await _workflow(db_session, test_user)
        retired = await _workflow(
            db_session, test_user, name="Retired", is_active=False
        )
        await _workflow(db_session, admin_user, name="Theirs", is_active=False)

        _, result = await _call(
            "list_available_workflows", db_session, test_user, {"is_active": False}
        )

        assert [w["id"] for w in result["data"]["workflows"]] == [str(retired.id)]
        assert result["data"]["workflows"][0]["is_active"] is False

    async def test_node_count_and_missing_description(self, db_session, test_user):
        await _workflow(db_session, test_user, nodes=("start", "wait", "end"))
        await _workflow(db_session, test_user, name="Empty", nodes=())

        _, result = await _call("list_available_workflows", db_session, test_user, {})

        by_name = {w["name"]: w for w in result["data"]["workflows"]}
        assert by_name["Nightly report"]["node_count"] == 3
        assert by_name["Empty"]["node_count"] == 0
        assert by_name["Empty"]["description"] == ""

    async def test_nothing_to_list(self, db_session, test_user, admin_user):
        await _workflow(db_session, admin_user)

        _, result = await _call("list_available_workflows", db_session, test_user, {})

        assert result == {"success": True, "data": {"workflows": [], "count": 0}}

    async def test_listing_is_bounded(self, db_session, test_user):
        for index in range(23):
            await _workflow(db_session, test_user, name=f"wf {index}", nodes=())

        _, result = await _call("list_available_workflows", db_session, test_user, {})

        assert result["data"]["count"] == 20
        assert len(result["data"]["workflows"]) == 20


class TestExecuteWorkflow:
    async def test_requires_workflow_id(self, db_session, test_user):
        await _workflow(db_session, test_user)

        for params in ({}, {"workflow_id": "   "}):
            _, result = await _call("execute_workflow", db_session, test_user, params)
            assert _refused(result)
            assert "workflow_id" in result["error"]
        assert await _executions(db_session) == []

    async def test_creates_and_runs_an_execution_owned_by_the_caller(
        self, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user)

        job, result = await _call(
            "execute_workflow",
            db_session,
            test_user,
            {"workflow_id": str(workflow.id)},
        )

        assert result["success"] is True
        rows = await _executions(db_session)
        assert len(rows) == 1
        row = rows[0]
        assert result["data"]["execution_id"] == str(row.id)
        assert result["data"]["workflow_id"] == str(workflow.id)
        assert result["data"]["status"] == row.status
        assert row.workflow_id == workflow.id
        assert row.user_id == test_user.id
        assert row.trigger_type
        assert row.context is not None
        assert row.progress is not None
        # With nothing passed, the run says which job launched it.
        assert row.trigger_data == {"source_job_id": str(job.id)}

        # Launched, not run: the work is handed to the task that does it,
        # exactly once. Running the graph inline held the agent's turn on
        # every node.
        assert row.status == "pending"
        assert row.started_at is None
        (args, kwargs), *rest = queued
        assert rest == []
        task_fn = workflow_tasks.execute_workflow_task.run
        bound = inspect.signature(task_fn).bind(*args, **kwargs)
        assert bound.arguments["execution_id"] == str(row.id)

    async def test_trigger_data_and_inputs_are_passed_through(
        self, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user)

        _, result = await _call(
            "execute_workflow",
            db_session,
            test_user,
            {
                "workflow_id": str(workflow.id),
                "trigger_data": {"reason": "weekly"},
                "inputs": {"topic": "prefetchers", "limit": 3},
            },
        )

        assert result["success"] is True
        (row,) = await _executions(db_session)
        assert row.trigger_data == {"reason": "weekly"}
        assert row.context["topic"] == "prefetchers"
        assert row.context["limit"] == 3

    async def test_unknown_workflow_is_refused(self, db_session, test_user, queued):
        _, result = await _call(
            "execute_workflow", db_session, test_user, {"workflow_id": str(uuid4())}
        )

        assert _refused(result)
        assert "not found" in result["error"]
        assert await _executions(db_session) == []
        assert queued == []

    async def test_inactive_workflow_is_refused(self, db_session, test_user, queued):
        workflow = await _workflow(db_session, test_user, is_active=False)

        _, result = await _call(
            "execute_workflow",
            db_session,
            test_user,
            {"workflow_id": str(workflow.id)},
        )

        assert _refused(result)
        assert "not active" in result["error"]
        assert await _executions(db_session) == []
        assert queued == []

    async def test_another_users_workflow_is_refused(
        self, db_session, test_user, admin_user, queued
    ):
        theirs = await _workflow(db_session, admin_user)

        _, result = await _call(
            "execute_workflow", db_session, test_user, {"workflow_id": str(theirs.id)}
        )

        assert _refused(result)
        # Indistinguishable from an id that does not exist.
        assert "not found" in result["error"]
        assert await _executions(db_session) == []
        assert queued == []

    async def test_malformed_workflow_id_is_an_error_result(
        self, db_session, test_user, queued
    ):
        _, result = await _call(
            "execute_workflow", db_session, test_user, {"workflow_id": "not-a-uuid"}
        )

        assert _refused(result)
        assert await _executions(db_session) == []
        assert queued == []

    async def test_a_workflow_that_cannot_run_is_reported_and_recorded(
        self, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user, nodes=("end",))

        _, result = await _call(
            "execute_workflow",
            db_session,
            test_user,
            {"workflow_id": str(workflow.id)},
        )

        assert _refused(result)
        assert "start node" in result["error"]
        (row,) = await _executions(db_session)
        assert row.status == "failed"
        assert "start node" in row.error
        assert row.completed_at is not None

    async def test_a_failed_run_names_the_execution_it_left_behind(
        self, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user, nodes=("end",))

        _, result = await _call(
            "execute_workflow",
            db_session,
            test_user,
            {"workflow_id": str(workflow.id)},
        )

        (row,) = await _executions(db_session)
        assert str(row.id) in str(result)


class TestGetWorkflowStatus:
    async def test_requires_execution_id(self, db_session, test_user):
        for params in ({}, {"execution_id": "  "}):
            _, result = await _call(
                "get_workflow_status", db_session, test_user, params
            )
            assert _refused(result)
            assert "execution_id" in result["error"]

    async def test_reports_a_running_execution(self, db_session, test_user):
        workflow = await _workflow(db_session, test_user)
        started = datetime(2026, 1, 1, 9, 0, 0)
        execution = await _execution(
            db_session,
            test_user,
            workflow,
            status="running",
            progress=50,
            started_at=started,
        )

        _, result = await _call(
            "get_workflow_status",
            db_session,
            test_user,
            {"execution_id": str(execution.id)},
        )

        assert result["success"] is True
        data = result["data"]
        assert data["execution_id"] == str(execution.id)
        assert data["workflow_id"] == str(workflow.id)
        assert data["status"] == "running"
        assert data["progress"] == 50
        assert data["error"] is None
        assert data["started_at"].startswith("2026-01-01")
        assert data["completed_at"] is None

    async def test_reports_a_failed_execution_with_its_error(
        self, db_session, test_user
    ):
        workflow = await _workflow(db_session, test_user)
        execution = await _execution(
            db_session,
            test_user,
            workflow,
            status="failed",
            progress=33,
            error="Node 'step3' failed: timeout",
            started_at=datetime(2026, 1, 1, 9, 0, 0),
            completed_at=datetime(2026, 1, 1, 9, 2, 0),
        )

        _, result = await _call(
            "get_workflow_status",
            db_session,
            test_user,
            {"execution_id": str(execution.id)},
        )

        data = result["data"]
        assert data["status"] == "failed"
        assert data["progress"] == 33
        assert data["error"] == "Node 'step3' failed: timeout"
        assert data["completed_at"].startswith("2026-01-01")

    async def test_reports_the_output_of_a_completed_execution(
        self, db_session, test_user
    ):
        workflow = await _workflow(db_session, test_user)
        execution = await _execution(
            db_session,
            test_user,
            workflow,
            status="completed",
            progress=100,
            context={"summary": {"papers": 7}},
        )

        _, result = await _call(
            "get_workflow_status",
            db_session,
            test_user,
            {"execution_id": str(execution.id)},
        )

        assert result["data"]["status"] == "completed"
        assert result["data"]["progress"] == 100
        # What the workflow produced lives in the execution's context; a
        # status that omits it leaves the caller no way to read the result.
        assert {"summary": {"papers": 7}} in result["data"].values()

    async def test_status_of_an_execution_the_tool_launched(
        self, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user)
        _, launched = await _call(
            "execute_workflow",
            db_session,
            test_user,
            {"workflow_id": str(workflow.id)},
        )

        _, result = await _call(
            "get_workflow_status",
            db_session,
            test_user,
            {"execution_id": launched["data"]["execution_id"]},
        )

        (row,) = await _executions(db_session)
        assert result["data"]["execution_id"] == str(row.id)
        assert result["data"]["status"] == row.status == launched["data"]["status"]
        assert result["data"]["progress"] == row.progress

    async def test_another_users_execution_is_refused(
        self, db_session, test_user, admin_user
    ):
        theirs = await _workflow(db_session, admin_user)
        execution = await _execution(
            db_session,
            admin_user,
            theirs,
            status="failed",
            error="token sk-live-123 rejected",
        )

        _, result = await _call(
            "get_workflow_status",
            db_session,
            test_user,
            {"execution_id": str(execution.id)},
        )

        assert _refused(result)
        assert "sk-live-123" not in str(result)

    async def test_unknown_execution_is_refused(self, db_session, test_user):
        missing = str(uuid4())

        _, result = await _call(
            "get_workflow_status", db_session, test_user, {"execution_id": missing}
        )

        assert _refused(result)
        assert "not found" in result["error"]
        assert UUID(missing)

    async def test_malformed_execution_id_is_an_error_result(
        self, db_session, test_user
    ):
        _, result = await _call(
            "get_workflow_status",
            db_session,
            test_user,
            {"execution_id": "not-a-uuid"},
        )

        assert _refused(result)


class TestWorkflowToolSchemas:
    """Tests for workflow tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "list_available_workflows" in names
        assert "execute_workflow" in names
        assert "get_workflow_status" in names

    def test_execute_workflow_requires_workflow_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("execute_workflow")
        assert tool is not None
        assert "workflow_id" in tool["parameters"].get("required", [])

    def test_get_workflow_status_requires_execution_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_workflow_status")
        assert tool is not None
        assert "execution_id" in tool["parameters"].get("required", [])

    def test_list_workflows_has_no_required_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("list_available_workflows")
        assert tool is not None
        assert tool["parameters"].get("required", []) == []


class TestWorkflowToolRegistry:
    """Tests for workflow tool registry classification."""

    def test_execute_workflow_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("execute_workflow")
        assert meta is not None
        assert meta.effects == "write"

    def test_list_workflows_is_read_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("list_available_workflows")
        assert meta is not None
        assert meta.effects == "read"

    def test_get_status_is_read_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_workflow_status")
        assert meta is not None
        assert meta.effects == "read"

    def test_execute_workflow_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("execute_workflow")
        assert meta is not None
        assert meta.cost_tier == "medium"


class TestAsyncExecuteRoute:
    """POST /workflows/{id}/execute/async queues through the same path."""

    async def test_queues_a_pending_execution(
        self, client, auth_headers, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user)

        response = client.post(
            f"/api/v1/workflows/{workflow.id}/execute/async",
            json={"inputs": {"topic": "x"}},
            headers=auth_headers,
        )

        assert response.status_code == 202, response.text
        (row,) = await _executions(db_session)
        assert response.json()["execution_id"] == str(row.id)
        assert row.status == "pending"
        assert [args for args, _ in queued] == [(str(row.id),)]

    async def test_a_workflow_without_a_start_is_refused_and_not_queued(
        self, client, auth_headers, db_session, test_user, queued
    ):
        workflow = await _workflow(db_session, test_user, nodes=("end",))

        response = client.post(
            f"/api/v1/workflows/{workflow.id}/execute/async",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "start node" in response.json()["detail"]
        assert queued == []

    async def test_an_unknown_workflow_is_404(self, client, auth_headers, queued):
        response = client.post(
            f"/api/v1/workflows/{uuid4()}/execute/async", json={}, headers=auth_headers
        )
        assert response.status_code == 404
        assert queued == []
