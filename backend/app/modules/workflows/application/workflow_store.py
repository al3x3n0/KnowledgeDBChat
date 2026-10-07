"""Reading and writing a user's workflows and their executions.

This was inline in ``api/endpoints/workflows.py``: the "load this user's
workflow with its graph" query appeared eight times, and create, update and
template import each rebuilt the nodes and edges by hand. Here there is one of
each, and the endpoint translates the results to HTTP.

**A reload after a write uses ``populate_existing``.** Sessions here do not
expire on commit, so an object loaded before the write keeps the relationship
collections it had: an update replaced the graph, reloaded it with
``selectinload``, and answered with the graph it had just replaced, because a
loader does not overwrite a collection that is already loaded. The workflow
editor saves through that endpoint and got its own edit back undone.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Optional, Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.workflow import Workflow, WorkflowEdge, WorkflowExecution, WorkflowNode

CANCELLABLE_STATUSES = ("pending", "running")


class WorkflowNotFound(LookupError):
    """No workflow of that id belongs to that user."""


class ExecutionNotFound(LookupError):
    """No execution of that id belongs to that user."""


class ExecutionNotCancellable(ValueError):
    """The execution has already finished."""


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Pydantic model or a template dict alike."""
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _add_graph(
    db: AsyncSession, workflow_id: UUID, nodes: Iterable[Any], edges: Iterable[Any]
) -> None:
    for node in nodes:
        db.add(
            WorkflowNode(
                workflow_id=workflow_id,
                node_id=_field(node, "node_id"),
                node_type=_field(node, "node_type"),
                tool_id=_field(node, "tool_id"),
                builtin_tool=_field(node, "builtin_tool"),
                config=_field(node, "config") or {},
                position_x=_field(node, "position_x", 0) or 0,
                position_y=_field(node, "position_y", 0) or 0,
            )
        )
    for edge in edges:
        db.add(
            WorkflowEdge(
                workflow_id=workflow_id,
                source_node_id=_field(edge, "source_node_id"),
                target_node_id=_field(edge, "target_node_id"),
                source_handle=_field(edge, "source_handle"),
                condition=_field(edge, "condition"),
            )
        )


async def get_owned_workflow(
    db: AsyncSession,
    workflow_id: UUID,
    user_id: UUID,
    *,
    with_graph: bool = False,
    with_tools: bool = False,
    fresh: bool = False,
) -> Workflow:
    """The user's workflow, or :class:`WorkflowNotFound`.

    ``with_graph`` loads nodes and edges (``with_tools`` also each node's
    custom tool); ``fresh`` re-reads collections already in the session.
    """
    query = select(Workflow).where(
        Workflow.id == workflow_id, Workflow.user_id == user_id
    )
    if with_graph or with_tools:
        nodes = selectinload(Workflow.nodes)
        if with_tools:
            nodes = nodes.selectinload(WorkflowNode.tool)
        query = query.options(nodes, selectinload(Workflow.edges))
    if fresh:
        query = query.execution_options(populate_existing=True)
    workflow = (await db.execute(query)).scalar_one_or_none()
    if workflow is None:
        raise WorkflowNotFound(str(workflow_id))
    return workflow


async def list_workflows(
    db: AsyncSession,
    user_id: UUID,
    *,
    is_active: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[Sequence[Workflow], int]:
    """A page of the user's workflows, newest change first, and the total."""
    query = select(Workflow).where(Workflow.user_id == user_id)
    if is_active is not None:
        query = query.where(Workflow.is_active == is_active)
    total = (
        await db.execute(select(func.count()).select_from(query.subquery()))
    ).scalar() or 0
    page = (
        await db.execute(
            query.options(
                selectinload(Workflow.nodes), selectinload(Workflow.executions)
            )
            .order_by(Workflow.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars()
    return page.all(), total


async def list_selectable_workflows(
    db: AsyncSession, user_id: UUID, *, exclude_id: Optional[UUID] = None
) -> Sequence[Workflow]:
    """Active workflows a sub-workflow node may call, by name."""
    query = select(Workflow).where(
        Workflow.user_id == user_id, Workflow.is_active.is_(True)
    )
    if exclude_id:
        query = query.where(Workflow.id != exclude_id)
    return (await db.execute(query.order_by(Workflow.name))).scalars().all()


async def create_workflow(db: AsyncSession, user_id: UUID, data: Any) -> Workflow:
    """Create a workflow with its nodes and edges; returns it loaded."""
    workflow = Workflow(
        user_id=user_id,
        name=data.name,
        description=data.description,
        is_active=data.is_active,
        trigger_config=data.trigger_config or {},
    )
    db.add(workflow)
    await db.flush()
    _add_graph(db, workflow.id, data.nodes, data.edges)
    await db.commit()
    return await get_owned_workflow(
        db, workflow.id, user_id, with_graph=True, fresh=True
    )


async def update_workflow(
    db: AsyncSession, workflow_id: UUID, user_id: UUID, data: Any
) -> Workflow:
    """Apply an update; a given ``nodes`` or ``edges`` list replaces the old.

    Returns the workflow as saved -- reloaded with ``fresh``, or it would
    carry the graph this update replaced.
    """
    workflow = await get_owned_workflow(db, workflow_id, user_id)
    for name in ("name", "description", "is_active", "trigger_config"):
        value = getattr(data, name)
        if value is not None:
            setattr(workflow, name, value)
    if data.nodes is not None:
        await db.execute(
            delete(WorkflowNode).where(WorkflowNode.workflow_id == workflow_id)
        )
        _add_graph(db, workflow_id, data.nodes, ())
    if data.edges is not None:
        await db.execute(
            delete(WorkflowEdge).where(WorkflowEdge.workflow_id == workflow_id)
        )
        _add_graph(db, workflow_id, (), data.edges)
    await db.commit()
    return await get_owned_workflow(
        db, workflow_id, user_id, with_graph=True, fresh=True
    )


async def delete_workflow(db: AsyncSession, workflow_id: UUID, user_id: UUID) -> None:
    workflow = await get_owned_workflow(db, workflow_id, user_id)
    await db.delete(workflow)
    await db.commit()


async def import_template(
    db: AsyncSession,
    user_id: UUID,
    template: Mapping[str, Any],
    *,
    name: Optional[str] = None,
) -> Workflow:
    """Copy a workflow template into the user's workflows.

    A name the user already has gets a numeric suffix rather than failing.
    """
    workflow_name = name or template["name"]
    taken = (
        await db.execute(
            select(Workflow.id).where(
                Workflow.user_id == user_id, Workflow.name == workflow_name
            )
        )
    ).first()
    if taken:
        workflow_name = f"{workflow_name} ({int(time.time()) % 10000})"

    workflow = Workflow(
        user_id=user_id,
        name=workflow_name,
        description=template.get("description"),
        is_active=True,
        trigger_config=template.get("trigger_config", {"type": "manual"}),
    )
    db.add(workflow)
    await db.flush()
    _add_graph(db, workflow.id, template.get("nodes", []), template.get("edges", []))
    await db.commit()
    return await get_owned_workflow(
        db, workflow.id, user_id, with_graph=True, fresh=True
    )


async def list_executions(
    db: AsyncSession,
    workflow_id: UUID,
    user_id: UUID,
    *,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[Workflow, Sequence[WorkflowExecution], int]:
    """The workflow, a page of its executions (newest first), and the total."""
    workflow = await get_owned_workflow(db, workflow_id, user_id)
    query = select(WorkflowExecution).where(
        WorkflowExecution.workflow_id == workflow_id
    )
    if status:
        query = query.where(WorkflowExecution.status == status)
    total = (
        await db.execute(select(func.count()).select_from(query.subquery()))
    ).scalar() or 0
    page = (
        await db.execute(
            query.order_by(WorkflowExecution.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars()
    return workflow, page.all(), total


async def get_execution(
    db: AsyncSession,
    execution_id: UUID,
    *,
    user_id: Optional[UUID] = None,
    with_nodes: bool = False,
    fresh: bool = False,
) -> WorkflowExecution:
    """An execution, the user's when ``user_id`` is given; else ExecutionNotFound."""
    query = select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
    if user_id is not None:
        query = query.where(WorkflowExecution.user_id == user_id)
    if with_nodes:
        query = query.options(selectinload(WorkflowExecution.node_executions))
    if fresh:
        query = query.execution_options(populate_existing=True)
    execution = (await db.execute(query)).scalar_one_or_none()
    if execution is None:
        raise ExecutionNotFound(str(execution_id))
    return execution


async def cancel_execution(
    db: AsyncSession, execution_id: UUID, user_id: UUID
) -> WorkflowExecution:
    execution = await get_execution(db, execution_id, user_id=user_id)
    if execution.status not in CANCELLABLE_STATUSES:
        raise ExecutionNotCancellable(execution.status)
    execution.status = "cancelled"
    execution.error = "Cancelled by user"
    await db.commit()
    return execution


__all__ = [
    "ExecutionNotCancellable",
    "ExecutionNotFound",
    "WorkflowNotFound",
    "cancel_execution",
    "create_workflow",
    "delete_workflow",
    "get_execution",
    "get_owned_workflow",
    "import_template",
    "list_executions",
    "list_selectable_workflows",
    "list_workflows",
    "update_workflow",
]
