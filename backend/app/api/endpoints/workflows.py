"""
API endpoints for workflows.

Provides:
- Workflow CRUD operations
- Workflow execution
- Execution history
- WebSocket for real-time execution updates
"""

from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from sqlalchemy.orm import selectinload
from loguru import logger
import json

from app.core.database import get_db
from app.models.user import User
from app.models.workflow import (
    Workflow, WorkflowNode, WorkflowEdge,
    WorkflowExecution, WorkflowNodeExecution
)
from app.services.auth_service import get_current_user
from app.services.workflow_engine import WorkflowEngine, WorkflowExecutionError
from app.schemas.workflow import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowListItem,
    WorkflowListResponse,
    WorkflowNodeCreate,
    WorkflowEdgeCreate,
    WorkflowExecutionCreate,
    WorkflowExecutionResponse,
    WorkflowExecutionListItem,
    WorkflowExecutionListResponse,
    ToolSchemaResponse,
    ToolSchemaListResponse,
    ToolParameterDetail,
    ContextVariable,
    ContextSchemaResponse,
    WorkflowValidationIssue,
    WorkflowValidationResponse,
    WorkflowSynthesisRequest,
    WorkflowSynthesisResponse,
    WorkflowSummary,
)
from app.services.workflow_synthesis_service import WorkflowSynthesisService

# Try to import redis for WebSocket pub/sub
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


router = APIRouter()


# =============================================================================
# Workflow CRUD Endpoints
# =============================================================================

@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all workflows for the current user."""
    try:
        query = select(Workflow).where(Workflow.user_id == current_user.id)

        if is_active is not None:
            query = query.where(Workflow.is_active == is_active)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results with node count
        query = query.options(
            selectinload(Workflow.nodes),
            selectinload(Workflow.executions)
        ).order_by(Workflow.updated_at.desc()).offset(offset).limit(limit)

        result = await db.execute(query)
        workflows = result.scalars().all()

        # Build response with counts
        items = []
        for wf in workflows:
            items.append(WorkflowListItem(
                id=wf.id,
                user_id=wf.user_id,
                name=wf.name,
                description=wf.description,
                is_active=wf.is_active,
                trigger_config=wf.trigger_config,
                created_at=wf.created_at,
                updated_at=wf.updated_at,
                node_count=len(wf.nodes),
                execution_count=len(wf.executions)
            ))

        return WorkflowListResponse(workflows=items, total=total)

    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list-for-selection", response_model=List[WorkflowSummary])
async def list_workflows_for_selection(
    exclude_id: Optional[UUID] = Query(None, description="Workflow ID to exclude (e.g., current workflow)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List workflows available for sub-workflow selection.

    Returns a simplified list (id, name, description, is_active) for use
    in dropdown selectors. Optionally excludes a workflow by ID to prevent
    self-references.
    """
    try:
        query = select(Workflow).where(
            Workflow.user_id == current_user.id,
            Workflow.is_active == True  # Only active workflows can be used as sub-workflows
        )

        # Exclude the specified workflow (prevents selecting itself)
        if exclude_id:
            query = query.where(Workflow.id != exclude_id)

        query = query.order_by(Workflow.name)

        result = await db.execute(query)
        workflows = result.scalars().all()

        return [
            WorkflowSummary(
                id=wf.id,
                name=wf.name,
                description=wf.description,
                is_active=wf.is_active
            )
            for wf in workflows
        ]

    except Exception as e:
        logger.error(f"Error listing workflows for selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    workflow_data: WorkflowCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new workflow with nodes and edges."""
    try:
        # Create workflow
        workflow = Workflow(
            user_id=current_user.id,
            name=workflow_data.name,
            description=workflow_data.description,
            is_active=workflow_data.is_active,
            trigger_config=workflow_data.trigger_config or {}
        )
        db.add(workflow)
        await db.flush()

        # Create nodes
        for node_data in workflow_data.nodes:
            node = WorkflowNode(
                workflow_id=workflow.id,
                node_id=node_data.node_id,
                node_type=node_data.node_type,
                tool_id=node_data.tool_id,
                builtin_tool=node_data.builtin_tool,
                config=node_data.config,
                position_x=node_data.position_x,
                position_y=node_data.position_y
            )
            db.add(node)

        # Create edges
        for edge_data in workflow_data.edges:
            edge = WorkflowEdge(
                workflow_id=workflow.id,
                source_node_id=edge_data.source_node_id,
                target_node_id=edge_data.target_node_id,
                source_handle=edge_data.source_handle,
                condition=edge_data.condition
            )
            db.add(edge)

        await db.commit()

        # Reload with relationships
        result = await db.execute(
            select(Workflow)
            .options(selectinload(Workflow.nodes), selectinload(Workflow.edges))
            .where(Workflow.id == workflow.id)
        )
        workflow = result.scalar_one()

        logger.info(f"Created workflow '{workflow.name}' for user {current_user.id}")
        return WorkflowResponse.model_validate(workflow)

    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize", response_model=WorkflowSynthesisResponse)
async def synthesize_workflow(
    synthesis_request: WorkflowSynthesisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a workflow draft from a natural language description."""
    service = WorkflowSynthesisService()
    try:
        workflow, warnings = await service.synthesize(
            description=synthesis_request.description,
            name=synthesis_request.name,
            trigger_config=synthesis_request.trigger_config,
            is_active=synthesis_request.is_active,
            user_id=current_user.id,
            db=db,
        )
        return WorkflowSynthesisResponse(workflow=workflow, warnings=warnings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Error synthesizing workflow: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific workflow with all nodes and edges."""
    result = await db.execute(
        select(Workflow)
        .options(
            selectinload(Workflow.nodes).selectinload(WorkflowNode.tool),
            selectinload(Workflow.edges)
        )
        .where(Workflow.id == workflow_id, Workflow.user_id == current_user.id)
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowResponse.model_validate(workflow)


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: UUID,
    workflow_data: WorkflowUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a workflow including its nodes and edges."""
    try:
        result = await db.execute(
            select(Workflow)
            .options(selectinload(Workflow.nodes), selectinload(Workflow.edges))
            .where(Workflow.id == workflow_id, Workflow.user_id == current_user.id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Update basic fields
        if workflow_data.name is not None:
            workflow.name = workflow_data.name
        if workflow_data.description is not None:
            workflow.description = workflow_data.description
        if workflow_data.is_active is not None:
            workflow.is_active = workflow_data.is_active
        if workflow_data.trigger_config is not None:
            workflow.trigger_config = workflow_data.trigger_config

        # Update nodes if provided
        if workflow_data.nodes is not None:
            # Delete existing nodes
            await db.execute(
                delete(WorkflowNode).where(WorkflowNode.workflow_id == workflow_id)
            )

            # Create new nodes
            for node_data in workflow_data.nodes:
                node = WorkflowNode(
                    workflow_id=workflow.id,
                    node_id=node_data.node_id,
                    node_type=node_data.node_type,
                    tool_id=node_data.tool_id,
                    builtin_tool=node_data.builtin_tool,
                    config=node_data.config,
                    position_x=node_data.position_x,
                    position_y=node_data.position_y
                )
                db.add(node)

        # Update edges if provided
        if workflow_data.edges is not None:
            # Delete existing edges
            await db.execute(
                delete(WorkflowEdge).where(WorkflowEdge.workflow_id == workflow_id)
            )

            # Create new edges
            for edge_data in workflow_data.edges:
                edge = WorkflowEdge(
                    workflow_id=workflow.id,
                    source_node_id=edge_data.source_node_id,
                    target_node_id=edge_data.target_node_id,
                    source_handle=edge_data.source_handle,
                    condition=edge_data.condition
                )
                db.add(edge)

        await db.commit()

        # Reload with relationships
        result = await db.execute(
            select(Workflow)
            .options(selectinload(Workflow.nodes), selectinload(Workflow.edges))
            .where(Workflow.id == workflow.id)
        )
        workflow = result.scalar_one()

        logger.info(f"Updated workflow '{workflow.name}'")
        return WorkflowResponse.model_validate(workflow)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating workflow: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a workflow and all its nodes, edges, and executions."""
    try:
        result = await db.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.user_id == current_user.id
            )
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        await db.delete(workflow)
        await db.commit()

        logger.info(f"Deleted workflow {workflow_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Workflow Execution Endpoints
# =============================================================================

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: UUID,
    execution_data: WorkflowExecutionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute a workflow synchronously.

    For long-running workflows, consider using the async execution endpoint.
    """
    try:
        engine = WorkflowEngine(db, current_user)

        execution = await engine.execute_workflow(
            workflow_id=workflow_id,
            trigger_type=execution_data.trigger_type,
            trigger_data=execution_data.trigger_data,
            initial_context=execution_data.inputs
        )

        # Reload with node executions
        result = await db.execute(
            select(WorkflowExecution)
            .options(selectinload(WorkflowExecution.node_executions))
            .where(WorkflowExecution.id == execution.id)
        )
        execution = result.scalar_one()

        return WorkflowExecutionResponse.model_validate(execution)

    except WorkflowExecutionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/execute/async", status_code=202)
async def execute_workflow_async(
    workflow_id: UUID,
    execution_data: WorkflowExecutionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Queue a workflow for asynchronous execution.

    Returns the execution ID. Use WebSocket or polling to monitor progress.
    """
    try:
        # Verify workflow exists
        result = await db.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.user_id == current_user.id
            )
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if not workflow.is_active:
            raise HTTPException(status_code=400, detail="Workflow is not active")

        # Create pending execution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            user_id=current_user.id,
            trigger_type=execution_data.trigger_type,
            trigger_data=execution_data.trigger_data,
            status="pending",
            progress=0,
            context=execution_data.inputs
        )
        db.add(execution)
        await db.commit()
        await db.refresh(execution)

        # Queue Celery task
        from app.tasks.workflow_tasks import execute_workflow_task
        execute_workflow_task.delay(str(execution.id))

        return {
            "execution_id": str(execution.id),
            "status": "pending",
            "message": "Workflow execution queued"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue workflow execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/executions", response_model=WorkflowExecutionListResponse)
async def list_workflow_executions(
    workflow_id: UUID,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List executions for a specific workflow."""
    try:
        # Verify workflow ownership
        result = await db.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.user_id == current_user.id
            )
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Build query
        query = select(WorkflowExecution).where(
            WorkflowExecution.workflow_id == workflow_id
        )

        if status:
            query = query.where(WorkflowExecution.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results
        query = query.order_by(WorkflowExecution.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        executions = result.scalars().all()

        items = [
            WorkflowExecutionListItem(
                id=e.id,
                workflow_id=e.workflow_id,
                workflow_name=workflow.name,
                trigger_type=e.trigger_type,
                status=e.status,
                progress=e.progress,
                error=e.error,
                created_at=e.created_at,
                started_at=e.started_at,
                completed_at=e.completed_at
            )
            for e in executions
        ]

        return WorkflowExecutionListResponse(executions=items, total=total)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_execution(
    execution_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get details of a specific workflow execution."""
    result = await db.execute(
        select(WorkflowExecution)
        .options(selectinload(WorkflowExecution.node_executions))
        .where(
            WorkflowExecution.id == execution_id,
            WorkflowExecution.user_id == current_user.id
        )
    )
    execution = result.scalar_one_or_none()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    return WorkflowExecutionResponse.model_validate(execution)


@router.post("/executions/{execution_id}/cancel", status_code=200)
async def cancel_execution(
    execution_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Cancel a running workflow execution."""
    result = await db.execute(
        select(WorkflowExecution).where(
            WorkflowExecution.id == execution_id,
            WorkflowExecution.user_id == current_user.id
        )
    )
    execution = result.scalar_one_or_none()

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    if execution.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution with status '{execution.status}'"
        )

    execution.status = "cancelled"
    execution.error = "Cancelled by user"
    await db.commit()

    return {"status": "cancelled", "message": "Execution cancelled"}


# =============================================================================
# WebSocket for Real-time Execution Updates
# =============================================================================

@router.websocket("/executions/{execution_id}/stream")
async def execution_stream(
    websocket: WebSocket,
    execution_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time execution updates.

    Subscribes to Redis pub/sub channel for the execution.
    """
    await websocket.accept()

    if not REDIS_AVAILABLE:
        await websocket.send_json({
            "type": "error",
            "message": "Real-time updates not available (Redis not configured)"
        })
        await websocket.close()
        return

    try:
        # Create Redis connection
        from app.core.config import settings
        redis = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

        pubsub = redis.pubsub()
        channel = f"workflow:{execution_id}"
        await pubsub.subscribe(channel)

        # Send initial status
        result = await db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
        )
        execution = result.scalar_one_or_none()

        if execution:
            await websocket.send_json({
                "type": "initial",
                "status": execution.status,
                "progress": execution.progress,
                "current_node_id": execution.current_node_id
            })

        # Listen for updates
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)

                    # Close on completion or error
                    if data.get("type") in ["complete", "error"]:
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await redis.close()
        except:
            pass
        try:
            await websocket.close()
        except:
            pass


# =============================================================================
# Schema Introspection Endpoints
# =============================================================================

def _flatten_json_schema(schema: dict) -> List[ToolParameterDetail]:
    """
    Flatten a JSON Schema into a list of parameter details for UI display.
    """
    parameters = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for name, prop_schema in properties.items():
        parameters.append(ToolParameterDetail(
            name=name,
            type=prop_schema.get("type", "any"),
            description=prop_schema.get("description"),
            required=name in required,
            default=prop_schema.get("default"),
            enum=prop_schema.get("enum")
        ))

    return parameters


@router.get("/tools/builtin", response_model=ToolSchemaListResponse)
async def list_builtin_tools(
    current_user: User = Depends(get_current_user)
):
    """
    List all built-in tools with their parameter schemas.

    This is useful for the workflow editor to display available tools
    and their input requirements.
    """
    from app.services.agent_tools import AGENT_TOOLS

    tools = []
    for tool in AGENT_TOOLS:
        schema = tool.get("parameters", {})
        tools.append(ToolSchemaResponse(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=schema,
            parameter_list=_flatten_json_schema(schema),
            tool_type="builtin"
        ))

    return ToolSchemaListResponse(tools=tools)


@router.get("/tools/builtin/{tool_name}", response_model=ToolSchemaResponse)
async def get_builtin_tool_schema(
    tool_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the parameter schema for a specific built-in tool.
    """
    from app.services.agent_tools import AGENT_TOOLS

    for tool in AGENT_TOOLS:
        if tool["name"] == tool_name:
            schema = tool.get("parameters", {})
            return ToolSchemaResponse(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=schema,
                parameter_list=_flatten_json_schema(schema),
                tool_type="builtin"
            )

    raise HTTPException(status_code=404, detail=f"Built-in tool '{tool_name}' not found")


@router.get("/{workflow_id}/context-schema", response_model=ContextSchemaResponse)
async def get_workflow_context_schema(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a workflow and return available context variables at each node.

    This helps the workflow editor provide autocomplete suggestions for
    input mappings by showing what context variables are available at
    each node based on the outputs of upstream nodes.
    """
    # Load workflow
    result = await db.execute(
        select(Workflow)
        .options(
            selectinload(Workflow.nodes).selectinload(WorkflowNode.tool),
            selectinload(Workflow.edges)
        )
        .where(Workflow.id == workflow_id, Workflow.user_id == current_user.id)
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Build a graph to analyze context flow
    from app.services.agent_tools import AGENT_TOOLS

    # Build adjacency list
    outgoing = {}  # node_id -> list of target_node_ids
    for edge in workflow.edges:
        if edge.source_node_id not in outgoing:
            outgoing[edge.source_node_id] = []
        outgoing[edge.source_node_id].append(edge.target_node_id)

    # Determine what each node outputs
    node_outputs = {}  # node_id -> list of ContextVariables
    node_map = {n.node_id: n for n in workflow.nodes}

    for node in workflow.nodes:
        output_key = node.config.get("output_key", node.node_id)

        if node.node_type in ("start", "end"):
            # Control nodes don't produce meaningful context
            node_outputs[node.node_id] = []
        elif node.node_type == "tool":
            # Tool nodes output based on tool schema
            variables = []

            # Default: add output_key as generic object
            variables.append(ContextVariable(
                path=f"context.{output_key}",
                type="object",
                from_node=node.node_id,
                description=f"Output from {node.builtin_tool or 'custom tool'}"
            ))

            # Try to infer output structure from tool
            if node.builtin_tool:
                for tool in AGENT_TOOLS:
                    if tool["name"] == node.builtin_tool:
                        # Add common output fields based on tool type
                        if node.builtin_tool == "search_documents":
                            variables.append(ContextVariable(
                                path=f"context.{output_key}.results",
                                type="array",
                                from_node=node.node_id,
                                description="Search results"
                            ))
                        elif node.builtin_tool == "get_document_details":
                            variables.append(ContextVariable(
                                path=f"context.{output_key}.title",
                                type="string",
                                from_node=node.node_id,
                                description="Document title"
                            ))
                            variables.append(ContextVariable(
                                path=f"context.{output_key}.content",
                                type="string",
                                from_node=node.node_id,
                                description="Document content"
                            ))
                        break

            node_outputs[node.node_id] = variables

        elif node.node_type == "condition":
            # Conditions output their result
            node_outputs[node.node_id] = [
                ContextVariable(
                    path=f"context.{output_key}.condition_result",
                    type="boolean",
                    from_node=node.node_id,
                    description="Result of condition evaluation"
                )
            ]

        elif node.node_type == "loop":
            # Loops provide loop context
            node_outputs[node.node_id] = [
                ContextVariable(
                    path="loop.item",
                    type="any",
                    from_node=node.node_id,
                    description="Current loop item"
                ),
                ContextVariable(
                    path="loop.index",
                    type="integer",
                    from_node=node.node_id,
                    description="Current loop index (0-based)"
                ),
                ContextVariable(
                    path="loop.total",
                    type="integer",
                    from_node=node.node_id,
                    description="Total number of items"
                ),
                ContextVariable(
                    path=f"context.{output_key}.results",
                    type="array",
                    from_node=node.node_id,
                    description="Results from all loop iterations"
                )
            ]

        elif node.node_type == "parallel":
            node_outputs[node.node_id] = [
                ContextVariable(
                    path=f"context.{output_key}.parallel_results",
                    type="array",
                    from_node=node.node_id,
                    description="Results from parallel branches"
                )
            ]

        else:
            node_outputs[node.node_id] = []

    # Compute available context at each node using topological traversal
    def get_ancestors(node_id: str, visited: set = None) -> set:
        """Get all ancestor nodes (nodes that execute before this one)."""
        if visited is None:
            visited = set()
        if node_id in visited:
            return set()

        ancestors = set()
        for edge in workflow.edges:
            if edge.target_node_id == node_id:
                ancestors.add(edge.source_node_id)
                ancestors.update(get_ancestors(edge.source_node_id, visited | {node_id}))
        return ancestors

    result_nodes = {}
    for node in workflow.nodes:
        ancestors = get_ancestors(node.node_id)
        available = []

        # Collect outputs from all ancestors
        for ancestor_id in ancestors:
            available.extend(node_outputs.get(ancestor_id, []))

        # Add trigger_data as always available
        available.insert(0, ContextVariable(
            path="context.trigger_data",
            type="object",
            from_node="_trigger",
            description="Data from workflow trigger"
        ))

        result_nodes[node.node_id] = available

    return ContextSchemaResponse(nodes=result_nodes)


@router.post("/{workflow_id}/validate", response_model=WorkflowValidationResponse)
async def validate_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Validate a workflow and return any issues found.

    Checks:
    - Required input parameters are mapped
    - Output key collisions
    - Graph structure (exactly one start node, etc.)
    - Unreachable nodes
    """
    # Load workflow
    result = await db.execute(
        select(Workflow)
        .options(
            selectinload(Workflow.nodes).selectinload(WorkflowNode.tool),
            selectinload(Workflow.edges)
        )
        .where(Workflow.id == workflow_id, Workflow.user_id == current_user.id)
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    issues = []

    # Check for exactly one start node
    start_nodes = [n for n in workflow.nodes if n.node_type == "start"]
    if len(start_nodes) == 0:
        issues.append(WorkflowValidationIssue(
            severity="error",
            message="Workflow must have a start node"
        ))
    elif len(start_nodes) > 1:
        issues.append(WorkflowValidationIssue(
            severity="error",
            message=f"Workflow has {len(start_nodes)} start nodes, should have exactly one"
        ))

    # Check for end nodes
    end_nodes = [n for n in workflow.nodes if n.node_type == "end"]
    if len(end_nodes) == 0:
        issues.append(WorkflowValidationIssue(
            severity="warning",
            message="Workflow has no end node"
        ))

    # Check output key collisions
    output_keys = {}
    for node in workflow.nodes:
        if node.node_type in ("start", "end"):
            continue
        output_key = node.config.get("output_key", node.node_id)
        if output_key in output_keys:
            output_keys[output_key].append(node.node_id)
        else:
            output_keys[output_key] = [node.node_id]

    for key, node_ids in output_keys.items():
        if len(node_ids) > 1:
            issues.append(WorkflowValidationIssue(
                severity="warning",
                node_id=node_ids[0],
                field="output_key",
                message=f"Output key '{key}' is used by multiple nodes: {', '.join(node_ids)}"
            ))

    # Check tool nodes have valid tools
    from app.services.agent_tools import AGENT_TOOLS
    builtin_names = {t["name"] for t in AGENT_TOOLS}

    for node in workflow.nodes:
        if node.node_type == "tool":
            if not node.tool_id and not node.builtin_tool:
                issues.append(WorkflowValidationIssue(
                    severity="error",
                    node_id=node.node_id,
                    message="Tool node has no tool configured"
                ))
            elif node.builtin_tool and node.builtin_tool not in builtin_names:
                issues.append(WorkflowValidationIssue(
                    severity="error",
                    node_id=node.node_id,
                    field="builtin_tool",
                    message=f"Unknown built-in tool: {node.builtin_tool}"
                ))

    # Check for unreachable nodes
    if start_nodes:
        reachable = set()
        to_visit = [start_nodes[0].node_id]

        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            reachable.add(current)

            for edge in workflow.edges:
                if edge.source_node_id == current:
                    to_visit.append(edge.target_node_id)

        for node in workflow.nodes:
            if node.node_id not in reachable:
                issues.append(WorkflowValidationIssue(
                    severity="warning",
                    node_id=node.node_id,
                    message=f"Node '{node.node_id}' is not reachable from the start node"
                ))

    return WorkflowValidationResponse(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues
    )
