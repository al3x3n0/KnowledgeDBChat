"""What a workflow's nodes produce is stored with the execution.

`execution.context` is a plain JSON column and the engine writes node outputs
into the dict in place. SQLAlchemy does not see a change made inside a JSON
value, so nothing told it to write the column: an execution could complete
with every node's output present in memory and none of it in the database,
where the status endpoint, the agent tool and a parent workflow read it.
"""

import pytest
from sqlalchemy import select

from app.models.workflow import Workflow, WorkflowEdge, WorkflowExecution, WorkflowNode
from app.services.workflow_engine import WorkflowEngine
from tests.conftest import TestSessionLocal

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def no_redis(monkeypatch):
    async def _none(self):
        return None

    monkeypatch.setattr(WorkflowEngine, "_get_redis", _none)


async def test_a_nodes_output_is_in_the_stored_execution(db_session, test_user):
    workflow = Workflow(user_id=test_user.id, name="Pause then finish")
    db_session.add(workflow)
    await db_session.flush()
    for node_id, node_type, config in (
        ("start", "start", {}),
        ("pause", "wait", {"wait_seconds": 0, "output_key": "paused"}),
        ("end", "end", {}),
    ):
        db_session.add(
            WorkflowNode(
                workflow_id=workflow.id,
                node_id=node_id,
                node_type=node_type,
                config=config,
            )
        )
    for source, target in (("start", "pause"), ("pause", "end")):
        db_session.add(
            WorkflowEdge(
                workflow_id=workflow.id, source_node_id=source, target_node_id=target
            )
        )
    await db_session.commit()

    execution = await WorkflowEngine(db_session, test_user).execute_workflow(
        workflow_id=workflow.id, initial_context={"given": 1}
    )
    assert execution.status == "completed"
    assert execution.context.get("paused") == {"waited": 0}

    # Read by a different session: what is in the database, not in memory.
    async with TestSessionLocal() as other:
        stored = (
            await other.execute(
                select(WorkflowExecution.context).where(
                    WorkflowExecution.id == execution.id
                )
            )
        ).scalar_one()

    assert stored.get("given") == 1
    assert stored.get("paused") == {"waited": 0}
