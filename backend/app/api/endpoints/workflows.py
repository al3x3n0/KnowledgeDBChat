"""
API endpoints for workflows.

Provides:
- Workflow CRUD operations
- Workflow execution
- Execution history
- WebSocket for real-time execution updates
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.modules.workflows.application import workflow_store
from app.schemas.workflow import (
    ContextSchemaResponse,
    ContextVariable,
    ToolParameterDetail,
    ToolSchemaListResponse,
    ToolSchemaResponse,
    WorkflowCreate,
    WorkflowExecutionCreate,
    WorkflowExecutionListItem,
    WorkflowExecutionListResponse,
    WorkflowExecutionResponse,
    WorkflowListItem,
    WorkflowListResponse,
    WorkflowResponse,
    WorkflowSummary,
    WorkflowSynthesisRequest,
    WorkflowSynthesisResponse,
    WorkflowUpdate,
    WorkflowValidationIssue,
    WorkflowValidationResponse,
)
from app.services.auth_service import get_current_user
from app.services.workflow_engine import WorkflowEngine, WorkflowExecutionError
from app.services.workflow_synthesis_service import WorkflowSynthesisService

# Try to import redis for WebSocket pub/sub
try:
    import redis.asyncio  # noqa: F401 - only to learn whether it is installed

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
    db: AsyncSession = Depends(get_db),
):
    """List all workflows for the current user."""
    try:
        workflows, total = await workflow_store.list_workflows(
            db, current_user.id, is_active=is_active, limit=limit, offset=offset
        )
        items = [
            WorkflowListItem(
                id=wf.id,
                user_id=wf.user_id,
                name=wf.name,
                description=wf.description,
                is_active=wf.is_active,
                trigger_config=wf.trigger_config,
                created_at=wf.created_at,
                updated_at=wf.updated_at,
                node_count=len(wf.nodes),
                execution_count=len(wf.executions),
                origin_plugin_slug=wf.origin_plugin_slug,
            )
            for wf in workflows
        ]
        return WorkflowListResponse(workflows=items, total=total)
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list-for-selection", response_model=List[WorkflowSummary])
async def list_workflows_for_selection(
    exclude_id: Optional[UUID] = Query(
        None, description="Workflow ID to exclude (e.g., current workflow)"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List workflows available for sub-workflow selection.

    Returns a simplified list (id, name, description, is_active) for use
    in dropdown selectors. Optionally excludes a workflow by ID to prevent
    self-references.
    """
    try:
        workflows = await workflow_store.list_selectable_workflows(
            db, current_user.id, exclude_id=exclude_id
        )
        return [
            WorkflowSummary(
                id=wf.id,
                name=wf.name,
                description=wf.description,
                is_active=wf.is_active,
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
    db: AsyncSession = Depends(get_db),
):
    """Create a new workflow with nodes and edges."""
    try:
        workflow = await workflow_store.create_workflow(
            db, current_user.id, workflow_data
        )
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
        bundle = await service.synthesize_bundle(
            description=synthesis_request.description,
            name=synthesis_request.name,
            trigger_config=synthesis_request.trigger_config,
            is_active=synthesis_request.is_active,
            user_id=current_user.id,
            db=db,
            synthesize_custom_tools=bool(synthesis_request.synthesize_custom_tools),
            preferred_tool_type=synthesis_request.preferred_tool_type,
            expose_workflow_as_tool=bool(synthesis_request.expose_workflow_as_tool),
            workflow_tool_name=synthesis_request.workflow_tool_name,
        )
        return WorkflowSynthesisResponse(
            workflow=bundle.workflow,
            warnings=bundle.warnings,
            custom_tools=bundle.custom_tools,
            workflow_tool=bundle.workflow_tool,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Error synthesizing workflow: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific workflow with all nodes and edges."""
    try:
        workflow = await workflow_store.get_owned_workflow(
            db, workflow_id, current_user.id, with_tools=True
        )
    except workflow_store.WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowResponse.model_validate(workflow)


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: UUID,
    workflow_data: WorkflowUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a workflow including its nodes and edges."""
    try:
        workflow = await workflow_store.update_workflow(
            db, workflow_id, current_user.id, workflow_data
        )
        logger.info(f"Updated workflow '{workflow.name}'")
        return WorkflowResponse.model_validate(workflow)
    except workflow_store.WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        logger.error(f"Error updating workflow: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workflow and all its nodes, edges, and executions."""
    try:
        await workflow_store.delete_workflow(db, workflow_id, current_user.id)
        logger.info(f"Deleted workflow {workflow_id}")
    except workflow_store.WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
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
    db: AsyncSession = Depends(get_db),
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
            initial_context=execution_data.inputs,
        )

        # Reload with node executions, overwriting what the engine left in the
        # session -- a plain reload keeps a collection it already loaded.
        execution = await workflow_store.get_execution(
            db, execution.id, with_nodes=True, fresh=True
        )

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
    db: AsyncSession = Depends(get_db),
):
    """
    Queue a workflow for asynchronous execution.

    Returns the execution ID. Use WebSocket or polling to monitor progress.
    """
    try:
        try:
            execution = await WorkflowEngine(db, current_user).queue_workflow(
                workflow_id=workflow_id,
                trigger_type=execution_data.trigger_type,
                trigger_data=execution_data.trigger_data,
                initial_context=execution_data.inputs,
            )
        except WorkflowExecutionError as e:
            status_code = 404 if "not found" in str(e) else 400
            detail = "Workflow not found" if status_code == 404 else str(e)
            raise HTTPException(status_code=status_code, detail=detail)

        return {
            "execution_id": str(execution.id),
            "status": "pending",
            "message": "Workflow execution queued",
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
    db: AsyncSession = Depends(get_db),
):
    """List executions for a specific workflow."""
    try:
        workflow, executions, total = await workflow_store.list_executions(
            db,
            workflow_id,
            current_user.id,
            status=status,
            limit=limit,
            offset=offset,
        )
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
                completed_at=e.completed_at,
            )
            for e in executions
        ]
        return WorkflowExecutionListResponse(executions=items, total=total)
    except workflow_store.WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        logger.error(f"Error listing executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_execution(
    execution_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific workflow execution."""
    try:
        execution = await workflow_store.get_execution(
            db, execution_id, user_id=current_user.id, with_nodes=True
        )
    except workflow_store.ExecutionNotFound:
        raise HTTPException(status_code=404, detail="Execution not found")
    return WorkflowExecutionResponse.model_validate(execution)


@router.post("/executions/{execution_id}/cancel", status_code=200)
async def cancel_execution(
    execution_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running workflow execution."""
    try:
        await workflow_store.cancel_execution(db, execution_id, current_user.id)
    except workflow_store.ExecutionNotFound:
        raise HTTPException(status_code=404, detail="Execution not found")
    except workflow_store.ExecutionNotCancellable as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution with status '{e}'",
        )
    return {"status": "cancelled", "message": "Execution cancelled"}


# =============================================================================
# WebSocket for Real-time Execution Updates
# =============================================================================


@router.websocket("/executions/{execution_id}/stream")
async def execution_stream(
    websocket: WebSocket,
    execution_id: UUID,
):
    """
    WebSocket endpoint for real-time execution updates.

    Subscribes to Redis pub/sub channel for the execution.
    """
    await websocket.accept()

    # Who is asking, and whose execution this is, before anything is sent:
    # the stream carries node outputs as the workflow runs.
    from app.core.database import AsyncSessionLocal as _Session
    from app.utils.websocket_auth import authorize_owner

    async with _Session() as _db:
        try:
            _owner = (await workflow_store.get_execution(_db, execution_id)).user_id
        except workflow_store.ExecutionNotFound:
            _owner = None
    if await authorize_owner(websocket, _owner, what="Execution") is None:
        return

    if not REDIS_AVAILABLE:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Real-time updates not available (Redis not configured)",
            }
        )
        await websocket.close()
        return

    from app.utils.websocket_progress import forward_progress

    async with _Session() as _db:
        try:
            execution = await workflow_store.get_execution(_db, execution_id)
        except workflow_store.ExecutionNotFound:
            execution = None

    await forward_progress(
        websocket,
        f"workflow:{execution_id}",
        initial=(
            {
                "type": "initial",
                "status": execution.status,
                "progress": execution.progress,
                "current_node_id": execution.current_node_id,
            }
            if execution
            else None
        ),
        is_terminal=lambda m: m.get("type") in ("complete", "error"),
    )


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
        parameters.append(
            ToolParameterDetail(
                name=name,
                type=prop_schema.get("type", "any"),
                description=prop_schema.get("description"),
                required=name in required,
                default=prop_schema.get("default"),
                enum=prop_schema.get("enum"),
            )
        )

    return parameters


@router.get("/tools/builtin", response_model=ToolSchemaListResponse)
async def list_builtin_tools(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
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
        try:
            from app.services.tool_policy_engine import evaluate_tool_policy

            decision = await evaluate_tool_policy(
                db=db,
                tool_name=str(tool.get("name") or ""),
                user=current_user,
            )
            if not decision.allowed:
                continue
        except Exception:
            continue

        tools.append(
            ToolSchemaResponse(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=schema,
                parameter_list=_flatten_json_schema(schema),
                tool_type="builtin",
            )
        )

    return ToolSchemaListResponse(tools=tools)


@router.get("/tools/builtin/{tool_name}", response_model=ToolSchemaResponse)
async def get_builtin_tool_schema(
    tool_name: str, current_user: User = Depends(get_current_user)
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
                tool_type="builtin",
            )

    raise HTTPException(
        status_code=404, detail=f"Built-in tool '{tool_name}' not found"
    )


@router.get("/{workflow_id}/context-schema", response_model=ContextSchemaResponse)
async def get_workflow_context_schema(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze a workflow and return available context variables at each node.

    This helps the workflow editor provide autocomplete suggestions for
    input mappings by showing what context variables are available at
    each node based on the outputs of upstream nodes.
    """
    # Load workflow
    try:
        workflow = await workflow_store.get_owned_workflow(
            db, workflow_id, current_user.id, with_tools=True
        )
    except workflow_store.WorkflowNotFound:
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

    for node in workflow.nodes:
        output_key = node.config.get("output_key", node.node_id)

        if node.node_type in ("start", "end"):
            # Control nodes don't produce meaningful context
            node_outputs[node.node_id] = []
        elif node.node_type == "tool":
            # Tool nodes output based on tool schema
            variables = []

            # Default: add output_key as generic object
            variables.append(
                ContextVariable(
                    path=f"context.{output_key}",
                    type="object",
                    from_node=node.node_id,
                    description=f"Output from {node.builtin_tool or 'custom tool'}",
                )
            )

            # Try to infer output structure from tool
            if node.builtin_tool:
                for tool in AGENT_TOOLS:
                    if tool["name"] == node.builtin_tool:
                        # Add common output fields based on tool type
                        if node.builtin_tool == "search_documents":
                            variables.append(
                                ContextVariable(
                                    path=f"context.{output_key}.results",
                                    type="array",
                                    from_node=node.node_id,
                                    description="Search results",
                                )
                            )
                        elif node.builtin_tool == "get_document_details":
                            variables.append(
                                ContextVariable(
                                    path=f"context.{output_key}.title",
                                    type="string",
                                    from_node=node.node_id,
                                    description="Document title",
                                )
                            )
                            variables.append(
                                ContextVariable(
                                    path=f"context.{output_key}.content",
                                    type="string",
                                    from_node=node.node_id,
                                    description="Document content",
                                )
                            )
                        break

            node_outputs[node.node_id] = variables

        elif node.node_type == "condition":
            # Conditions output their result
            node_outputs[node.node_id] = [
                ContextVariable(
                    path=f"context.{output_key}.condition_result",
                    type="boolean",
                    from_node=node.node_id,
                    description="Result of condition evaluation",
                )
            ]

        elif node.node_type == "loop":
            # Loops provide loop context
            node_outputs[node.node_id] = [
                ContextVariable(
                    path="loop.item",
                    type="any",
                    from_node=node.node_id,
                    description="Current loop item",
                ),
                ContextVariable(
                    path="loop.index",
                    type="integer",
                    from_node=node.node_id,
                    description="Current loop index (0-based)",
                ),
                ContextVariable(
                    path="loop.total",
                    type="integer",
                    from_node=node.node_id,
                    description="Total number of items",
                ),
                ContextVariable(
                    path=f"context.{output_key}.results",
                    type="array",
                    from_node=node.node_id,
                    description="Results from all loop iterations",
                ),
            ]

        elif node.node_type == "parallel":
            node_outputs[node.node_id] = [
                ContextVariable(
                    path=f"context.{output_key}.parallel_results",
                    type="array",
                    from_node=node.node_id,
                    description="Results from parallel branches",
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
                ancestors.update(
                    get_ancestors(edge.source_node_id, visited | {node_id})
                )
        return ancestors

    result_nodes = {}
    for node in workflow.nodes:
        ancestors = get_ancestors(node.node_id)
        available = []

        # Collect outputs from all ancestors
        for ancestor_id in ancestors:
            available.extend(node_outputs.get(ancestor_id, []))

        # Add trigger_data as always available
        available.insert(
            0,
            ContextVariable(
                path="context.trigger_data",
                type="object",
                from_node="_trigger",
                description="Data from workflow trigger",
            ),
        )

        result_nodes[node.node_id] = available

    return ContextSchemaResponse(nodes=result_nodes)


@router.post("/{workflow_id}/validate", response_model=WorkflowValidationResponse)
async def validate_workflow(
    workflow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
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
    try:
        workflow = await workflow_store.get_owned_workflow(
            db, workflow_id, current_user.id, with_tools=True
        )
    except workflow_store.WorkflowNotFound:
        raise HTTPException(status_code=404, detail="Workflow not found")

    issues = []

    # Check for exactly one start node
    start_nodes = [n for n in workflow.nodes if n.node_type == "start"]
    if len(start_nodes) == 0:
        issues.append(
            WorkflowValidationIssue(
                severity="error", message="Workflow must have a start node"
            )
        )
    elif len(start_nodes) > 1:
        issues.append(
            WorkflowValidationIssue(
                severity="error",
                message=f"Workflow has {len(start_nodes)} start nodes, should have exactly one",
            )
        )

    # Check for end nodes
    end_nodes = [n for n in workflow.nodes if n.node_type == "end"]
    if len(end_nodes) == 0:
        issues.append(
            WorkflowValidationIssue(
                severity="warning", message="Workflow has no end node"
            )
        )

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
            issues.append(
                WorkflowValidationIssue(
                    severity="warning",
                    node_id=node_ids[0],
                    field="output_key",
                    message=f"Output key '{key}' is used by multiple nodes: {', '.join(node_ids)}",
                )
            )

    # Check tool nodes have valid tools
    from app.services.agent_tools import AGENT_TOOLS

    builtin_names = {t["name"] for t in AGENT_TOOLS}

    for node in workflow.nodes:
        if node.node_type == "tool":
            if not node.tool_id and not node.builtin_tool:
                issues.append(
                    WorkflowValidationIssue(
                        severity="error",
                        node_id=node.node_id,
                        message="Tool node has no tool configured",
                    )
                )
            elif node.builtin_tool and node.builtin_tool not in builtin_names:
                issues.append(
                    WorkflowValidationIssue(
                        severity="error",
                        node_id=node.node_id,
                        field="builtin_tool",
                        message=f"Unknown built-in tool: {node.builtin_tool}",
                    )
                )

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
                issues.append(
                    WorkflowValidationIssue(
                        severity="warning",
                        node_id=node.node_id,
                        message=f"Node '{node.node_id}' is not reachable from the start node",
                    )
                )

    return WorkflowValidationResponse(
        valid=not any(i.severity == "error" for i in issues), issues=issues
    )


# =============================================================================
# Workflow Template Endpoints
# =============================================================================


@router.get("/templates", tags=["workflow-templates"])
async def list_workflow_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_user),
):
    """
    List all available workflow templates.

    Templates are pre-built workflows for common automation tasks:
    - reporting: Weekly digest, automated reports
    - research: arXiv paper pipeline, literature review
    - analysis: Document analysis, batch processing
    - productivity: Meeting notes, email drafting
    - maintenance: Health checks, batch summarization
    """
    from app.services.workflow_templates import (
        get_template_summary,
        get_templates_by_category,
        list_template_categories,
    )

    if category:
        templates = get_templates_by_category(category)
        summary = [
            {
                "template_id": t["template_id"],
                "name": t["name"],
                "description": t["description"],
                "category": t.get("category", "other"),
                "trigger_type": t.get("trigger_config", {}).get("type", "manual"),
                "node_count": len(t.get("nodes", [])),
            }
            for t in templates
        ]
    else:
        summary = get_template_summary()

    return {
        "templates": summary,
        "categories": list_template_categories(),
        "total": len(summary),
    }


@router.get("/templates/{template_id}", tags=["workflow-templates"])
async def get_workflow_template(
    template_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get details of a specific workflow template."""
    from app.services.workflow_templates import get_template_by_id

    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return template


@router.post("/templates/{template_id}/import", tags=["workflow-templates"])
async def import_workflow_template(
    template_id: str,
    name_override: Optional[str] = Query(
        None, description="Custom name for the workflow"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Import a workflow template as a new workflow for the current user.

    This creates a copy of the template as a personal workflow that
    can be customized and executed.
    """
    from app.services.workflow_templates import get_template_by_id

    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    workflow = await workflow_store.import_template(
        db, current_user.id, template, name=name_override
    )

    return {
        "message": f"Template '{template['name']}' imported successfully",
        "workflow": WorkflowResponse.model_validate(workflow),
        "template_id": template_id,
    }
