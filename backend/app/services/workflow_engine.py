"""
Workflow Execution Engine.

Executes workflow graphs with support for:
- Linear execution
- Conditional branching
- Parallel execution
- Loops
- Real-time progress updates via Redis pub/sub
"""

import json
import asyncio
import time
import re
import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from uuid import UUID
from collections import defaultdict

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.workflow import (
    Workflow, WorkflowNode, WorkflowEdge,
    WorkflowExecution, WorkflowNodeExecution, UserTool
)
from app.models.user import User
from app.services.custom_tool_service import CustomToolService, ToolExecutionError
from app.services.agent_tools import AGENT_TOOLS
from app.core.config import settings

# Try to import redis for pub/sub
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class ResolvedValue:
    """
    Wrapper for resolved context values.

    Distinguishes between:
    - value=None, exists=True -> the path exists but value is null
    - value=None, exists=False -> the path doesn't exist
    - error is set -> syntax or resolution error occurred
    """
    value: Any
    exists: bool = True
    path: Optional[str] = None
    error: Optional[str] = None


class InputResolutionError(Exception):
    """Raised when input resolution fails."""
    def __init__(self, message: str, errors: Dict[str, str]):
        super().__init__(message)
        self.errors = errors


class WorkflowExecutionError(Exception):
    """Raised when workflow execution fails."""
    pass


class WorkflowCancelledError(Exception):
    """Raised when workflow execution is cancelled."""
    pass


class WorkflowEngine:
    """
    Engine for executing workflow graphs.

    Handles:
    - Graph traversal (topological execution)
    - Node execution dispatch
    - Context management
    - Error handling and recovery
    - Real-time progress updates
    """

    def __init__(self, db: AsyncSession, user: User):
        self.db = db
        self.user = user
        self.custom_tool_service = CustomToolService()
        self._redis_client: Optional[Any] = None

    async def _get_redis(self):
        """Get Redis client for pub/sub."""
        if not REDIS_AVAILABLE:
            return None
        if self._redis_client is None:
            try:
                self._redis_client = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Could not connect to Redis for workflow updates: {e}")
        return self._redis_client

    async def _publish_progress(
        self,
        execution_id: UUID,
        message_type: str,
        node_id: Optional[str] = None,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Publish progress update via Redis pub/sub."""
        redis = await self._get_redis()
        if redis is None:
            return

        try:
            message = {
                "type": message_type,
                "execution_id": str(execution_id),
                "node_id": node_id,
                "status": status,
                "progress": progress,
                "output": output,
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            }
            channel = f"workflow:{execution_id}"
            await redis.publish(channel, json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to publish workflow progress: {e}")

    async def execute_workflow(
        self,
        workflow_id: UUID,
        trigger_type: str = "manual",
        trigger_data: Optional[Dict[str, Any]] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Execute a workflow.

        Args:
            workflow_id: The workflow to execute
            trigger_type: How the workflow was triggered
            trigger_data: Data from the trigger
            initial_context: Initial context variables

        Returns:
            The WorkflowExecution record
        """
        # Load workflow with nodes and edges
        result = await self.db.execute(
            select(Workflow)
            .options(
                selectinload(Workflow.nodes).selectinload(WorkflowNode.tool),
                selectinload(Workflow.edges)
            )
            .where(Workflow.id == workflow_id, Workflow.user_id == self.user.id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise WorkflowExecutionError(f"Workflow {workflow_id} not found")

        if not workflow.is_active:
            raise WorkflowExecutionError("Workflow is not active")

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            user_id=self.user.id,
            trigger_type=trigger_type,
            trigger_data=trigger_data or {},
            status="pending",
            progress=0,
            context=initial_context or {}
        )
        self.db.add(execution)
        await self.db.commit()
        await self.db.refresh(execution)

        try:
            await self._run_workflow(
                workflow=workflow,
                execution=execution
            )
        except WorkflowCancelledError:
            return execution
        except Exception:
            raise

        return execution

    async def execute_existing_execution(
        self,
        execution: WorkflowExecution
    ) -> WorkflowExecution:
        """
        Execute an existing WorkflowExecution without creating a new record.
        """
        await self.db.refresh(execution)

        if execution.status == "cancelled":
            return execution

        if execution.status not in ["pending", "running"]:
            logger.warning(
                f"Execution {execution.id} is not runnable (status: {execution.status})"
            )
            return execution

        # Load workflow with nodes and edges
        result = await self.db.execute(
            select(Workflow)
            .options(
                selectinload(Workflow.nodes).selectinload(WorkflowNode.tool),
                selectinload(Workflow.edges)
            )
            .where(Workflow.id == execution.workflow_id, Workflow.user_id == self.user.id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise WorkflowExecutionError(f"Workflow {execution.workflow_id} not found")

        try:
            await self._run_workflow(
                workflow=workflow,
                execution=execution
            )
        except WorkflowCancelledError:
            return execution
        except Exception:
            raise

        return execution

    async def _run_workflow(
        self,
        workflow: Workflow,
        execution: WorkflowExecution
    ) -> None:
        """
        Execute a workflow using an existing execution record.
        """
        try:
            # Build execution graph
            graph = self._build_graph(workflow.nodes, workflow.edges)

            # Validate graph
            start_nodes = [n for n in workflow.nodes if n.node_type == "start"]
            if len(start_nodes) != 1:
                raise WorkflowExecutionError("Workflow must have exactly one start node")

            # Detect output key collisions
            collision_warnings = self._detect_output_key_collisions(workflow.nodes)
            if collision_warnings:
                logger.warning(f"Output key collisions detected: {collision_warnings}")
                execution.context["_warnings"] = execution.context.get("_warnings", [])
                execution.context["_warnings"].extend(collision_warnings)
                await self.db.commit()

            await self._raise_if_cancelled(execution)

            # Start execution
            execution.status = "running"
            execution.started_at = execution.started_at or datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "progress",
                status="running", progress=0
            )

            # Execute from start node
            start_node = start_nodes[0]
            await self._execute_node_chain(
                workflow, execution, graph, start_node.node_id
            )

            # Mark as completed
            execution.status = "completed"
            execution.progress = 100
            execution.completed_at = datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "complete",
                status="completed", progress=100
            )

        except WorkflowCancelledError as exc:
            execution.status = "cancelled"
            execution.error = str(exc)
            execution.completed_at = datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "cancelled",
                status="cancelled", error=str(exc)
            )
            raise

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = "failed"
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "error",
                status="failed", error=str(e)
            )
            raise

    async def _raise_if_cancelled(self, execution: WorkflowExecution) -> None:
        """Raise if execution is cancelled."""
        await self.db.refresh(execution)
        if execution.status == "cancelled":
            raise WorkflowCancelledError("Cancelled by user")

    def _detect_output_key_collisions(
        self,
        nodes: List[WorkflowNode]
    ) -> List[str]:
        """
        Detect potential output key collisions.

        Checks if multiple nodes use the same output_key, which could
        cause one node's output to overwrite another's.

        Returns:
            List of warning messages for detected collisions.
        """
        output_keys: Dict[str, List[str]] = {}  # output_key -> list of node_ids
        warnings = []

        for node in nodes:
            # Skip control flow nodes that don't produce meaningful outputs
            if node.node_type in ("start", "end"):
                continue

            # Get output_key from config, default to node_id
            output_key = node.config.get("output_key", node.node_id)

            if output_key in output_keys:
                output_keys[output_key].append(node.node_id)
            else:
                output_keys[output_key] = [node.node_id]

        # Generate warnings for collisions
        for key, node_ids in output_keys.items():
            if len(node_ids) > 1:
                warnings.append(
                    f"Output key '{key}' is used by multiple nodes: {', '.join(node_ids)}. "
                    f"Later executions may overwrite earlier outputs."
                )

        return warnings

    def _build_graph(
        self,
        nodes: List[WorkflowNode],
        edges: List[WorkflowEdge]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build an adjacency graph from nodes and edges.

        Returns:
            Dict mapping node_id to:
            - node: The WorkflowNode object
            - outgoing: List of (target_node_id, edge) tuples
            - incoming: List of source_node_ids
        """
        graph = {}

        # Initialize all nodes
        for node in nodes:
            graph[node.node_id] = {
                "node": node,
                "outgoing": [],
                "incoming": []
            }

        # Add edges
        for edge in edges:
            if edge.source_node_id in graph and edge.target_node_id in graph:
                graph[edge.source_node_id]["outgoing"].append(
                    (edge.target_node_id, edge)
                )
                graph[edge.target_node_id]["incoming"].append(
                    edge.source_node_id
                )

        return graph

    async def _execute_node_chain(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        graph: Dict[str, Dict[str, Any]],
        start_node_id: str,
        loop_context: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        written_keys: Optional[Set[str]] = None
    ):
        """
        Execute a chain of nodes starting from start_node_id.

        Handles branching, parallel execution, and loops.
        """
        current_node_id = start_node_id
        visited: Set[str] = set()
        context = execution.context if context is None else context
        written_keys = set() if written_keys is None else written_keys

        while current_node_id:
            await self._raise_if_cancelled(execution)
            if current_node_id in visited:
                # Prevent infinite loops (unless it's a loop node)
                node_info = graph.get(current_node_id)
                if node_info and node_info["node"].node_type != "loop":
                    logger.warning(f"Cycle detected at node {current_node_id}, breaking")
                    break

            visited.add(current_node_id)
            node_info = graph.get(current_node_id)

            if not node_info:
                logger.warning(f"Node {current_node_id} not found in graph")
                break

            node = node_info["node"]

            # Update current node
            execution.current_node_id = current_node_id
            await self.db.commit()

            # Execute the node
            next_node_id = await self._execute_single_node(
                workflow, execution, graph, node, loop_context, context, written_keys
            )

            # Calculate progress
            total_nodes = len(graph)
            completed = len(visited)
            execution.progress = min(int((completed / total_nodes) * 100), 99)
            await self.db.commit()

            # Move to next node
            current_node_id = next_node_id

        return written_keys

    async def _execute_single_node(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        graph: Dict[str, Dict[str, Any]],
        node: WorkflowNode,
        loop_context: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        written_keys: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Execute a single node and return the next node ID.

        Returns:
            The next node_id to execute, or None if chain ends
        """
        node_info = graph[node.node_id]
        context = execution.context if context is None else context
        written_keys = set() if written_keys is None else written_keys

        # Create node execution record
        node_execution = WorkflowNodeExecution(
            execution_id=execution.id,
            node_id=node.node_id,
            status="running",
            started_at=datetime.utcnow()
        )
        self.db.add(node_execution)
        await self.db.commit()

        await self._publish_progress(
            execution.id, "node_start",
            node_id=node.node_id, status="running"
        )

        start_time = time.time()
        output = None

        try:
            await self._raise_if_cancelled(execution)
            # Prepare input data by resolving context references
            input_data, resolution_details = self._resolve_inputs(
                node.config, context, loop_context, node_id=node.node_id
            )
            node_execution.input_data = input_data

            # Store resolution metadata for debugging
            resolution_errors = {
                k: v.error for k, v in resolution_details.items() if v.error
            }
            missing_paths = {
                k: v.path for k, v in resolution_details.items() if not v.exists and not v.error
            }
            if resolution_errors or missing_paths:
                node_execution.input_data = {
                    **input_data,
                    "_resolution_issues": {
                        "errors": resolution_errors,
                        "missing_paths": missing_paths
                    }
                }

            # Execute based on node type
            if node.node_type == "start":
                output = {"started": True}

            elif node.node_type == "end":
                output = {"ended": True}

            elif node.node_type == "tool":
                output = await self._execute_tool_node(node, input_data, execution)
                await self._raise_if_cancelled(execution)

            elif node.node_type == "condition":
                output = await self._execute_condition_node(
                    node, input_data, execution, graph, node_info, context
                )

            elif node.node_type == "parallel":
                output = await self._execute_parallel_node(
                    workflow, execution, graph, node, input_data, context
                )

            elif node.node_type == "loop":
                output = await self._execute_loop_node(
                    workflow, execution, graph, node, input_data, context
                )

            elif node.node_type == "wait":
                wait_seconds = node.config.get("wait_seconds", 0)
                if wait_seconds > 0:
                    remaining = min(wait_seconds, 300)
                    while remaining > 0:
                        await self._raise_if_cancelled(execution)
                        step = min(1, remaining)
                        await asyncio.sleep(step)
                        remaining -= step
                output = {"waited": wait_seconds}

            elif node.node_type == "switch":
                output = await self._execute_switch_node(
                    node, input_data, execution, graph, node_info, context
                )

            elif node.node_type == "subworkflow":
                output = await self._execute_subworkflow_node(
                    node, input_data, execution, context
                )

            else:
                raise WorkflowExecutionError(f"Unknown node type: {node.node_type}")

            # Store output in context
            output_key = node.config.get("output_key", node.node_id)
            context[output_key] = self._trim_for_context(output)
            written_keys.add(output_key)

            # Update node execution
            node_execution.status = "completed"
            node_execution.output_data = output
            node_execution.execution_time_ms = int((time.time() - start_time) * 1000)
            node_execution.completed_at = datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "node_complete",
                node_id=node.node_id, status="completed", output=output
            )

            # Determine next node
            if node.node_type == "end":
                return None

            if node.node_type in ("condition", "switch"):
                # Condition/Switch nodes determine next node based on result
                return output.get("_next_node")

            # Get first outgoing edge (for linear flow)
            if node_info["outgoing"]:
                return node_info["outgoing"][0][0]

            return None

        except WorkflowCancelledError:
            node_execution.status = "cancelled"
            node_execution.output_data = output
            node_execution.execution_time_ms = int((time.time() - start_time) * 1000)
            node_execution.completed_at = datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "node_cancelled",
                node_id=node.node_id, status="cancelled"
            )
            raise

        except Exception as e:
            logger.error(f"Node {node.node_id} execution failed: {e}")

            node_execution.status = "failed"
            node_execution.error = str(e)
            node_execution.execution_time_ms = int((time.time() - start_time) * 1000)
            node_execution.completed_at = datetime.utcnow()
            await self.db.commit()

            await self._publish_progress(
                execution.id, "node_error",
                node_id=node.node_id, status="failed", error=str(e)
            )

            raise WorkflowExecutionError(f"Node {node.node_id} failed: {e}")

    def _resolve_inputs(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
        loop_context: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, ResolvedValue]]:
        """
        Resolve input mappings from config using context.

        Supports expressions like:
        - {{context.step1.output}}
        - {{loop.item}}
        - {{loop.index}}

        Returns:
            Tuple of (resolved_values, resolution_details)
            - resolved_values: Dict mapping input keys to their resolved values
            - resolution_details: Dict mapping input keys to ResolvedValue with metadata
        """
        input_mapping = config.get("input_mapping", {})
        resolved = {}
        resolution_details = {}

        # Merge loop context into main context for resolution
        full_context = {**context}
        if loop_context:
            full_context["loop"] = loop_context

        for key, value in input_mapping.items():
            result = self._resolve_value(value, full_context)
            resolution_details[key] = result

            if result.error:
                logger.warning(
                    f"Input resolution error for '{key}' in node '{node_id}': {result.error}"
                )
            elif not result.exists:
                logger.warning(
                    f"Input path '{result.path}' not found for '{key}' in node '{node_id}'"
                )

            # Use the resolved value (may be None)
            resolved[key] = result.value

        return resolved, resolution_details

    def _validate_template_syntax(self, expr: str) -> Optional[str]:
        """
        Validate template expression syntax.

        Returns error message if invalid, None if valid.
        """
        # Check for balanced braces
        if expr.count("{{") != expr.count("}}"):
            return "Unbalanced braces in template expression"

        # Extract the path
        if expr.startswith("{{") and expr.endswith("}}"):
            path = expr[2:-2].strip()
            if not path:
                return "Empty path in template expression"

            # Validate path format (alphanumeric, dots, underscores, digits)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)*$', path):
                # Allow array indexing like items.0.name
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+|\.\d+)*$', path):
                    return f"Invalid path format: '{path}'"

        return None

    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> ResolvedValue:
        """
        Resolve a single value, handling template expressions.

        Returns ResolvedValue with metadata about the resolution.
        """
        if not isinstance(value, str):
            return ResolvedValue(value=value, exists=True)

        # Check for template expression
        if value.startswith("{{") and value.endswith("}}"):
            # Validate syntax first
            syntax_error = self._validate_template_syntax(value)
            if syntax_error:
                return ResolvedValue(
                    value=None,
                    exists=False,
                    path=value,
                    error=syntax_error
                )

            path = value[2:-2].strip()
            resolved_val, exists = self._get_nested_value(context, path)

            return ResolvedValue(
                value=resolved_val,
                exists=exists,
                path=path,
                error=None
            )

        # Plain string value
        return ResolvedValue(value=value, exists=True)

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Tuple[Any, bool]:
        """
        Get a nested value from a dict using dot notation.

        Returns:
            Tuple of (value, exists)
            - value: The resolved value or None
            - exists: True if the path exists (even if value is None)
        """
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None, False
            else:
                return None, False

        return current, True

    def _trim_for_context(self, value: Any) -> Any:
        """
        Trim large values before storing them in execution context.

        Keeps outputs bounded while preserving the shape of small values.
        """
        max_depth = 4
        max_items = 50
        max_str = 2000

        def trim(val: Any, depth: int) -> Any:
            if depth > max_depth:
                return {"_truncated": True}

            if isinstance(val, dict):
                trimmed = {}
                for idx, (k, v) in enumerate(val.items()):
                    if idx >= max_items:
                        trimmed["_truncated"] = True
                        break
                    trimmed[k] = trim(v, depth + 1)
                return trimmed

            if isinstance(val, list):
                items = [trim(item, depth + 1) for item in val[:max_items]]
                if len(val) > max_items:
                    items.append({"_truncated": True})
                return items

            if isinstance(val, tuple):
                items = [trim(item, depth + 1) for item in val[:max_items]]
                if len(val) > max_items:
                    items.append({"_truncated": True})
                return tuple(items)

            if isinstance(val, str) and len(val) > max_str:
                return val[:max_str] + "...[truncated]"

            return val

        return trim(value, 0)

    def _get_tool_schema(self, node: WorkflowNode) -> Optional[Dict[str, Any]]:
        """
        Get the parameter schema for a tool node.

        Returns the JSON Schema for the tool's parameters, or None if not available.
        """
        if node.tool_id and node.tool:
            # Custom user tool
            return node.tool.parameters_schema

        elif node.builtin_tool:
            # Built-in agent tool - look up in AGENT_TOOLS
            for tool in AGENT_TOOLS:
                if tool["name"] == node.builtin_tool:
                    return tool.get("parameters")
            return None

        return None

    def _validate_inputs_against_schema(
        self,
        inputs: Dict[str, Any],
        schema: Dict[str, Any],
        node_id: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate inputs against a JSON Schema.

        Returns:
            Tuple of (coerced_inputs, errors)
            - coerced_inputs: Inputs with type coercion applied where possible
            - errors: List of validation error messages
        """
        errors = []
        coerced = dict(inputs)

        if not schema:
            return coerced, errors

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for field in required:
            if field not in inputs or inputs[field] is None:
                errors.append(f"Missing required field: '{field}'")

        # Validate and coerce types
        for field, value in inputs.items():
            if field.startswith("_"):
                # Skip internal fields like _resolution_issues
                continue

            if field not in properties:
                # Unknown field - skip but warn
                logger.debug(f"Unknown input field '{field}' for node '{node_id}'")
                continue

            field_schema = properties[field]
            expected_type = field_schema.get("type")

            if value is None:
                # None is acceptable for optional fields
                continue

            # Type coercion
            coerced[field], type_error = self._coerce_type(value, expected_type, field)
            if type_error:
                errors.append(type_error)

        return coerced, errors

    def _coerce_type(
        self,
        value: Any,
        expected_type: str,
        field_name: str
    ) -> Tuple[Any, Optional[str]]:
        """
        Attempt to coerce a value to the expected type.

        Returns:
            Tuple of (coerced_value, error_message)
        """
        if expected_type is None:
            return value, None

        try:
            if expected_type == "string":
                if not isinstance(value, str):
                    return str(value), None
                return value, None

            elif expected_type == "integer":
                if isinstance(value, int) and not isinstance(value, bool):
                    return value, None
                if isinstance(value, float) and value.is_integer():
                    return int(value), None
                if isinstance(value, str) and value.isdigit():
                    return int(value), None
                return value, f"Field '{field_name}' expects integer, got {type(value).__name__}"

            elif expected_type == "number":
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value), None
                if isinstance(value, str):
                    try:
                        return float(value), None
                    except ValueError:
                        pass
                return value, f"Field '{field_name}' expects number, got {type(value).__name__}"

            elif expected_type == "boolean":
                if isinstance(value, bool):
                    return value, None
                if isinstance(value, str):
                    if value.lower() in ("true", "1", "yes"):
                        return True, None
                    if value.lower() in ("false", "0", "no"):
                        return False, None
                return value, f"Field '{field_name}' expects boolean, got {type(value).__name__}"

            elif expected_type == "array":
                if isinstance(value, (list, tuple)):
                    return list(value), None
                return value, f"Field '{field_name}' expects array, got {type(value).__name__}"

            elif expected_type == "object":
                if isinstance(value, dict):
                    return value, None
                return value, f"Field '{field_name}' expects object, got {type(value).__name__}"

            else:
                # Unknown type - pass through
                return value, None

        except Exception as e:
            return value, f"Type coercion failed for '{field_name}': {e}"

    async def _execute_tool_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a tool node (custom or built-in)."""
        # Get tool schema and validate inputs
        schema = self._get_tool_schema(node)
        if schema:
            validated_inputs, validation_errors = self._validate_inputs_against_schema(
                input_data, schema, node.node_id
            )

            if validation_errors:
                logger.warning(
                    f"Input validation issues for node '{node.node_id}': {validation_errors}"
                )
                # Store validation errors in execution context for debugging
                if "_validation_warnings" not in execution.context:
                    execution.context["_validation_warnings"] = {}
                execution.context["_validation_warnings"][node.node_id] = validation_errors

            # Use validated (potentially coerced) inputs
            input_data = validated_inputs
        if node.tool_id and node.tool:
            # Custom user tool
            result = await self.custom_tool_service.execute_tool(
                tool=node.tool,
                inputs=input_data,
                user=self.user,
                db=self.db
            )
            return result.get("output", {})

        elif node.builtin_tool:
            # Built-in agent tool
            return await self._execute_builtin_tool(node.builtin_tool, input_data)

        else:
            raise WorkflowExecutionError(
                f"Tool node {node.node_id} has no tool configured"
            )

    async def _execute_builtin_tool(
        self,
        tool_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a built-in agent tool.

        This bridges workflow execution to the agent tools system.
        """
        # Import agent service to use its tool handlers
        from app.services.agent_service import AgentService

        agent_service = AgentService()

        try:
            result = await agent_service.execute_tool(
                tool_name=tool_name,
                tool_input=inputs,
                user_id=self.user.id,
                db=self.db
            )
            return result
        except Exception as e:
            raise ToolExecutionError(f"Built-in tool '{tool_name}' failed: {e}")

    async def _execute_condition_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: WorkflowExecution,
        graph: Dict[str, Dict[str, Any]],
        node_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a condition node and determine the next branch.

        Config:
            condition: Expression to evaluate
            true_branch: node_id for true case
            false_branch: node_id for false case
        """
        condition = node.config.get("condition", {})

        # Evaluate condition
        result = self._evaluate_condition(condition, context)

        # Find the matching outgoing edge
        for target_node_id, edge in node_info["outgoing"]:
            edge_handle = edge.source_handle
            if result and edge_handle == "true":
                return {"condition_result": True, "_next_node": target_node_id}
            elif not result and edge_handle == "false":
                return {"condition_result": False, "_next_node": target_node_id}

        # Default: use first outgoing if no match
        if node_info["outgoing"]:
            return {"condition_result": result, "_next_node": node_info["outgoing"][0][0]}

        return {"condition_result": result, "_next_node": None}

    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a condition expression.

        Supports:
        - type: "equals", "not_equals", "greater_than", "less_than", "contains", "expression"
        - left: Value or path
        - right: Value or path
        """
        cond_type = condition.get("type", "expression")
        left = condition.get("left")
        right = condition.get("right")

        # Resolve values (extract .value from ResolvedValue)
        left_resolved = self._resolve_value(left, context) if left else ResolvedValue(value=None)
        right_resolved = self._resolve_value(right, context) if right else ResolvedValue(value=None)

        left_val = left_resolved.value
        right_val = right_resolved.value

        # Log warnings for resolution issues
        if left_resolved.error:
            logger.warning(f"Left condition value resolution error: {left_resolved.error}")
        if right_resolved.error:
            logger.warning(f"Right condition value resolution error: {right_resolved.error}")

        try:
            if cond_type == "equals":
                return left_val == right_val
            elif cond_type == "not_equals":
                return left_val != right_val
            elif cond_type == "greater_than":
                return float(left_val) > float(right_val)
            elif cond_type == "less_than":
                return float(left_val) < float(right_val)
            elif cond_type == "contains":
                return str(right_val) in str(left_val)
            elif cond_type == "truthy":
                return bool(left_val)
            elif cond_type == "expression":
                # Simple expression evaluation
                expression = condition.get("expression", "")
                # Basic safety: only allow simple expressions
                if any(kw in expression for kw in ["import", "exec", "eval", "__"]):
                    return False
                return bool(left_val)
            else:
                return bool(left_val)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False

    async def _execute_parallel_node(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        graph: Dict[str, Dict[str, Any]],
        node: WorkflowNode,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute parallel branches and wait for all to complete.

        All outgoing edges from a parallel node are executed concurrently.
        """
        node_info = graph[node.node_id]
        branches = node_info["outgoing"]

        if not branches:
            return {"parallel_results": []}

        async def execute_branch(target_node_id: str) -> Dict[str, Any]:
            """Execute a single branch."""
            try:
                # Use isolated context per branch
                branch_context = copy.deepcopy(context)
                branch_written: Set[str] = set()

                await self._execute_node_chain(
                    workflow,
                    execution,
                    graph,
                    target_node_id,
                    context=branch_context,
                    written_keys=branch_written
                )

                return {
                    "branch": target_node_id,
                    "status": "completed",
                    "context": branch_context,
                    "written_keys": list(branch_written)
                }
            except Exception as e:
                return {"branch": target_node_id, "status": "failed", "error": str(e)}

        # Execute all branches concurrently
        tasks = [execute_branch(target_id) for target_id, _ in branches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        branch_results = []
        branch_outputs = {}
        merge_conflicts = []
        for result in results:
            if isinstance(result, Exception):
                branch_results.append({"status": "failed", "error": str(result)})
            else:
                branch_results.append({
                    "branch": result.get("branch"),
                    "status": result.get("status"),
                    "error": result.get("error")
                })
                if result.get("status") == "completed":
                    branch_context = result.get("context", context)
                    written_keys = result.get("written_keys", [])
                    branch_outputs[result["branch"]] = {
                        key: self._trim_for_context(branch_context.get(key))
                        for key in written_keys
                    }

                    for key in written_keys:
                        if key in context and context.get(key) != branch_context.get(key):
                            merge_conflicts.append({
                                "key": key,
                                "existing": self._trim_for_context(context.get(key)),
                                "incoming": self._trim_for_context(branch_context.get(key)),
                                "branch": result["branch"]
                            })
                            continue
                        context[key] = branch_context.get(key)

        return {
            "parallel_results": branch_results,
            "branch_outputs": branch_outputs,
            "merge_conflicts": merge_conflicts
        }

    async def _execute_loop_node(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        graph: Dict[str, Dict[str, Any]],
        node: WorkflowNode,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a loop node - iterate over a collection.

        Config:
            loop_source: Path to collection in context
            loop_body_node: First node of the loop body
            max_iterations: Safety limit
        """
        loop_source = node.config.get("loop_source", "")
        max_iterations = node.config.get("max_iterations", 100)

        # Get the items to iterate over
        items_resolved = self._resolve_value(loop_source, context)
        items = items_resolved.value

        # Log warnings for resolution issues
        if items_resolved.error:
            logger.warning(f"Loop source resolution error: {items_resolved.error}")
        elif not items_resolved.exists:
            logger.warning(f"Loop source path not found: {items_resolved.path}")

        if not isinstance(items, (list, tuple)):
            items = [items] if items else []

        # Limit iterations
        items = items[:max_iterations]

        node_info = graph[node.node_id]
        loop_results = []

        for index, item in enumerate(items):
            await self._raise_if_cancelled(execution)
            loop_context = {
                "item": item,
                "index": index,
                "total": len(items),
                "is_first": index == 0,
                "is_last": index == len(items) - 1
            }

            # Execute the loop body (first outgoing node)
            if node_info["outgoing"]:
                body_node_id = node_info["outgoing"][0][0]
                await self._execute_node_chain(
                    workflow,
                    execution,
                    graph,
                    body_node_id,
                    loop_context=loop_context,
                    context=context
                )

            loop_results.append({
                "index": index,
                "item": item,
                "completed": True
            })

        return {
            "loop_completed": True,
            "iterations": len(items),
            "results": loop_results
        }

    async def _execute_switch_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: WorkflowExecution,
        graph: Dict[str, Dict[str, Any]],
        node_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a switch node - multi-way branch based on value matching.

        Config:
            switch_expression: Expression to evaluate (e.g., "{{context.status}}")
            cases: List of {value, label} objects for matching
            default_label: Label for default case
        """
        switch_expr = node.config.get("switch_expression", "")
        cases = node.config.get("cases", [])

        # Resolve the switch expression
        resolved = self._resolve_value(switch_expr, context)
        switch_value = resolved.value

        if resolved.error:
            logger.warning(f"Switch expression resolution error: {resolved.error}")
        elif not resolved.exists:
            logger.warning(f"Switch expression path not found: {resolved.path}")

        # Find matching case
        matched_handle = "default"
        matched_label = node.config.get("default_label", "Default")

        for idx, case in enumerate(cases):
            case_value = case.get("value")
            # Support both exact match and type-coerced match
            if switch_value == case_value or str(switch_value) == str(case_value):
                matched_handle = f"case_{idx}"
                matched_label = case.get("label", f"Case {idx}")
                break

        # Find outgoing edge with matching handle
        for target_node_id, edge in node_info["outgoing"]:
            if edge.source_handle == matched_handle:
                return {
                    "switch_value": switch_value,
                    "matched_case": matched_handle,
                    "matched_label": matched_label,
                    "_next_node": target_node_id
                }

        # Fallback to default edge
        for target_node_id, edge in node_info["outgoing"]:
            if edge.source_handle == "default":
                return {
                    "switch_value": switch_value,
                    "matched_case": "default",
                    "matched_label": matched_label,
                    "_next_node": target_node_id
                }

        # No matching edge found
        logger.warning(f"No matching edge found for switch value '{switch_value}'")
        return {
            "switch_value": switch_value,
            "matched_case": None,
            "matched_label": None,
            "_next_node": None
        }

    async def _execute_subworkflow_node(
        self,
        node: WorkflowNode,
        input_data: Dict[str, Any],
        execution: WorkflowExecution,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a sub-workflow as a node.

        Config:
            workflow_id: UUID of the child workflow to execute
            timeout_seconds: Maximum execution time (default: 300)
            on_error: "fail" or "continue" (default: "fail")
        """
        workflow_id = node.config.get("workflow_id")
        timeout = node.config.get("timeout_seconds", 300)
        on_error = node.config.get("on_error", "fail")

        if not workflow_id:
            raise WorkflowExecutionError("Sub-workflow node missing workflow_id")

        # Check recursion depth to prevent infinite loops
        depth = await self._check_recursion_depth(execution)
        if depth > 5:
            raise WorkflowExecutionError(
                f"Maximum sub-workflow depth exceeded (depth={depth}). "
                "This may indicate a circular workflow reference."
            )

        # Load child workflow
        result = await self.db.execute(
            select(Workflow)
            .options(
                selectinload(Workflow.nodes).selectinload(WorkflowNode.tool),
                selectinload(Workflow.edges)
            )
            .where(Workflow.id == UUID(workflow_id))
        )
        child_workflow = result.scalar_one_or_none()

        if not child_workflow:
            raise WorkflowExecutionError(f"Sub-workflow {workflow_id} not found")

        if not child_workflow.is_active:
            raise WorkflowExecutionError(f"Sub-workflow {workflow_id} is not active")

        # Create child execution with parent reference
        child_execution = WorkflowExecution(
            workflow_id=child_workflow.id,
            user_id=execution.user_id,
            trigger_type="subworkflow",
            trigger_data={"parent_execution_id": str(execution.id), "parent_node_id": node.node_id},
            status="pending",
            context={"trigger_data": input_data},
            parent_execution_id=execution.id
        )
        self.db.add(child_execution)
        await self.db.commit()
        await self.db.refresh(child_execution)

        logger.info(
            f"Starting sub-workflow {workflow_id} (execution {child_execution.id}) "
            f"from parent {execution.id} at depth {depth + 1}"
        )

        try:
            # Execute child workflow with timeout
            await asyncio.wait_for(
                self._run_workflow(child_workflow, child_execution),
                timeout=timeout
            )

            # Refresh to get final state
            await self.db.refresh(child_execution)

            return {
                "subworkflow_id": str(workflow_id),
                "subworkflow_name": child_workflow.name,
                "execution_id": str(child_execution.id),
                "status": child_execution.status,
                "output": self._trim_for_context(child_execution.context)
            }

        except asyncio.TimeoutError:
            child_execution.status = "failed"
            child_execution.error = f"Timeout after {timeout} seconds"
            child_execution.completed_at = datetime.utcnow()
            await self.db.commit()

            logger.warning(f"Sub-workflow {workflow_id} timed out after {timeout}s")

            if on_error == "fail":
                raise WorkflowExecutionError(
                    f"Sub-workflow '{child_workflow.name}' timed out after {timeout} seconds"
                )

            return {
                "subworkflow_id": str(workflow_id),
                "subworkflow_name": child_workflow.name,
                "execution_id": str(child_execution.id),
                "status": "timeout",
                "error": f"Timeout after {timeout} seconds"
            }

        except WorkflowCancelledError:
            # Propagate cancellation
            raise

        except Exception as e:
            logger.error(f"Sub-workflow {workflow_id} failed: {e}")

            # Update child execution status
            child_execution.status = "failed"
            child_execution.error = str(e)
            child_execution.completed_at = datetime.utcnow()
            await self.db.commit()

            if on_error == "fail":
                raise WorkflowExecutionError(
                    f"Sub-workflow '{child_workflow.name}' failed: {e}"
                )

            return {
                "subworkflow_id": str(workflow_id),
                "subworkflow_name": child_workflow.name,
                "execution_id": str(child_execution.id),
                "status": "failed",
                "error": str(e)
            }

    async def _check_recursion_depth(self, execution: WorkflowExecution) -> int:
        """
        Count parent execution chain to prevent infinite recursion.

        Returns:
            Current depth (0 for top-level execution)
        """
        depth = 0
        current_id = getattr(execution, 'parent_execution_id', None)

        while current_id:
            depth += 1
            if depth > 10:  # Safety limit
                break

            result = await self.db.execute(
                select(WorkflowExecution.parent_execution_id)
                .where(WorkflowExecution.id == current_id)
            )
            row = result.first()
            if not row:
                break
            current_id = row[0]

        return depth


async def execute_workflow_async(
    db: AsyncSession,
    user: User,
    workflow_id: UUID,
    trigger_type: str = "manual",
    trigger_data: Optional[Dict[str, Any]] = None,
    initial_context: Optional[Dict[str, Any]] = None
) -> WorkflowExecution:
    """
    Convenience function to execute a workflow.

    Can be called from Celery tasks or directly.
    """
    engine = WorkflowEngine(db, user)
    return await engine.execute_workflow(
        workflow_id=workflow_id,
        trigger_type=trigger_type,
        trigger_data=trigger_data,
        initial_context=initial_context
    )
