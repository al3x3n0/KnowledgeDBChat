"""
Pydantic schemas for workflows and custom tools.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field, validator


# =============================================================================
# Tool Type Configurations
# =============================================================================

class WebhookToolConfig(BaseModel):
    """Configuration for webhook-type tools."""
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    url: str = Field(..., description="URL template with {{input.field}} placeholders")
    headers: Dict[str, str] = Field(default_factory=dict)
    body_template: Optional[str] = Field(None, description="Request body template (Jinja2)")
    response_path: Optional[str] = Field(None, description="JSONPath to extract from response")
    timeout_seconds: int = Field(30, ge=1, le=300)


class TransformToolConfig(BaseModel):
    """Configuration for transform-type tools."""
    transform_type: Literal["jinja2", "jsonpath", "javascript"] = "jinja2"
    template: str = Field(..., description="Transformation template or expression")


class PythonToolConfig(BaseModel):
    """Configuration for Python script tools."""
    code: str = Field(..., description="Python code to execute")
    timeout_seconds: int = Field(10, ge=1, le=60)
    allowed_imports: List[str] = Field(
        default_factory=lambda: ["json", "re", "datetime", "math", "collections"]
    )


class LLMPromptToolConfig(BaseModel):
    """Configuration for LLM prompt tools."""
    system_prompt: Optional[str] = Field(None, description="System prompt for the LLM")
    user_prompt: str = Field(..., description="User prompt template with {{input.field}} placeholders")
    output_format: Literal["text", "json"] = "text"
    model_override: Optional[str] = Field(None, description="Override user's default model")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)


class DockerContainerToolConfig(BaseModel):
    """Configuration for Docker container tools."""
    image: str = Field(..., description="Docker image name")
    command: Optional[List[str]] = Field(None, description="Command to run in the container")
    entrypoint: Optional[List[str]] = Field(None, description="Override entrypoint")
    input_mode: Literal["stdin", "file", "both"] = Field("stdin", description="Input method")
    output_mode: Literal["stdout", "file", "both"] = Field("stdout", description="Output method")
    input_file_path: Optional[str] = Field("/workspace/input.txt", description="Input file path inside container")
    output_file_path: Optional[str] = Field("/workspace/output.txt", description="Output file path inside container")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Execution timeout")
    memory_limit: str = Field("512m", description="Memory limit (e.g., '512m', '1g')")
    cpu_limit: float = Field(1.0, ge=0.1, le=8.0, description="CPU limit")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_dir: str = Field("/workspace", description="Working directory inside container")
    network_enabled: bool = Field(False, description="Enable network access")
    user: Optional[str] = Field(None, description="User to run as")


# =============================================================================
# User Tool Schemas
# =============================================================================

class UserToolBase(BaseModel):
    """Base schema for user tools."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    tool_type: Literal["webhook", "transform", "python", "llm_prompt", "docker_container"]
    parameters_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool input parameters"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific configuration"
    )
    is_enabled: bool = True


class UserToolCreate(UserToolBase):
    """Schema for creating a user tool."""
    pass


class UserToolUpdate(BaseModel):
    """Schema for updating a user tool."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    tool_type: Optional[Literal["webhook", "transform", "python", "llm_prompt", "docker_container"]] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None


class UserToolResponse(UserToolBase):
    """Response schema for user tools."""
    id: UUID
    user_id: UUID
    version: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserToolListResponse(BaseModel):
    """Response schema for listing tools."""
    tools: List[UserToolResponse]
    total: int


class UserToolTestRequest(BaseModel):
    """Request schema for testing a tool."""
    inputs: Dict[str, Any] = Field(default_factory=dict)


class UserToolTestResponse(BaseModel):
    """Response schema for tool test execution."""
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: int


# =============================================================================
# Workflow Node Schemas
# =============================================================================

class WorkflowNodeBase(BaseModel):
    """Base schema for workflow nodes."""
    node_id: str = Field(..., min_length=1, max_length=50)
    node_type: Literal["start", "end", "tool", "condition", "parallel", "loop", "wait", "switch", "subworkflow"]
    tool_id: Optional[UUID] = None
    builtin_tool: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    position_x: int = 0
    position_y: int = 0


class WorkflowNodeCreate(WorkflowNodeBase):
    """Schema for creating a workflow node."""
    pass


class WorkflowNodeUpdate(BaseModel):
    """Schema for updating a workflow node."""
    node_type: Optional[Literal["start", "end", "tool", "condition", "parallel", "loop", "wait", "switch", "subworkflow"]] = None
    tool_id: Optional[UUID] = None
    builtin_tool: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    position_x: Optional[int] = None
    position_y: Optional[int] = None


class WorkflowNodeResponse(WorkflowNodeBase):
    """Response schema for workflow nodes."""
    id: UUID
    workflow_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Workflow Edge Schemas
# =============================================================================

class WorkflowEdgeBase(BaseModel):
    """Base schema for workflow edges."""
    source_node_id: str = Field(..., min_length=1, max_length=50)
    target_node_id: str = Field(..., min_length=1, max_length=50)
    source_handle: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None


class WorkflowEdgeCreate(WorkflowEdgeBase):
    """Schema for creating a workflow edge."""
    pass


class WorkflowEdgeResponse(WorkflowEdgeBase):
    """Response schema for workflow edges."""
    id: UUID
    workflow_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Workflow Schemas
# =============================================================================

class TriggerConfig(BaseModel):
    """Configuration for workflow triggers."""
    type: Literal["manual", "schedule", "event", "webhook"] = "manual"
    schedule: Optional[str] = Field(None, description="Cron expression for scheduled triggers")
    event: Optional[str] = Field(None, description="Event name for event triggers")
    webhook_secret: Optional[str] = Field(None, description="Secret for webhook authentication")


class WorkflowBase(BaseModel):
    """Base schema for workflows."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: bool = True
    trigger_config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowCreate(WorkflowBase):
    """Schema for creating a workflow."""
    nodes: List[WorkflowNodeCreate] = Field(default_factory=list)
    edges: List[WorkflowEdgeCreate] = Field(default_factory=list)


class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None
    trigger_config: Optional[Dict[str, Any]] = None
    nodes: Optional[List[WorkflowNodeCreate]] = None
    edges: Optional[List[WorkflowEdgeCreate]] = None


class WorkflowSynthesisRequest(BaseModel):
    """Request schema for workflow synthesis."""
    description: str = Field(..., min_length=1)
    name: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = True
    trigger_config: Optional[Dict[str, Any]] = None


class WorkflowSynthesisResponse(BaseModel):
    """Response schema for workflow synthesis."""
    workflow: WorkflowCreate
    warnings: List[str] = Field(default_factory=list)


class WorkflowResponse(WorkflowBase):
    """Response schema for workflows."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    nodes: List[WorkflowNodeResponse] = Field(default_factory=list)
    edges: List[WorkflowEdgeResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class WorkflowListItem(BaseModel):
    """List item schema for workflows (without nodes/edges)."""
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str]
    is_active: bool
    trigger_config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    node_count: int = 0
    execution_count: int = 0

    class Config:
        from_attributes = True


class WorkflowListResponse(BaseModel):
    """Response schema for listing workflows."""
    workflows: List[WorkflowListItem]
    total: int


class WorkflowSummary(BaseModel):
    """Summary schema for workflow selection (sub-workflow picker)."""
    id: UUID
    name: str
    description: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True


# =============================================================================
# Workflow Execution Schemas
# =============================================================================

class WorkflowNodeExecutionResponse(BaseModel):
    """Response schema for node execution details."""
    id: UUID
    node_id: str
    status: str
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time_ms: Optional[int]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class WorkflowExecutionBase(BaseModel):
    """Base schema for workflow executions."""
    trigger_type: Literal["manual", "schedule", "event", "webhook"] = "manual"
    trigger_data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionCreate(WorkflowExecutionBase):
    """Schema for starting a workflow execution."""
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Initial context inputs")


class WorkflowExecutionResponse(BaseModel):
    """Response schema for workflow executions."""
    id: UUID
    workflow_id: UUID
    user_id: UUID
    parent_execution_id: Optional[UUID] = None  # For sub-workflow tracking
    trigger_type: str
    trigger_data: Optional[Dict[str, Any]]
    status: str
    progress: int
    current_node_id: Optional[str]
    context: Dict[str, Any]
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    node_executions: List[WorkflowNodeExecutionResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class WorkflowExecutionListItem(BaseModel):
    """List item schema for executions (without node details)."""
    id: UUID
    workflow_id: UUID
    workflow_name: Optional[str] = None
    trigger_type: str
    status: str
    progress: int
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class WorkflowExecutionListResponse(BaseModel):
    """Response schema for listing executions."""
    executions: List[WorkflowExecutionListItem]
    total: int


# =============================================================================
# WebSocket Messages
# =============================================================================

class WorkflowProgressMessage(BaseModel):
    """WebSocket message for execution progress updates."""
    type: Literal["progress", "node_start", "node_complete", "node_error", "complete", "error"]
    execution_id: UUID
    node_id: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[int] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# =============================================================================
# Schema Introspection Schemas
# =============================================================================

class ToolParameterDetail(BaseModel):
    """Detailed information about a tool parameter."""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


class ToolSchemaResponse(BaseModel):
    """Response schema for tool schema introspection."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full JSON Schema for parameters"
    )
    parameter_list: List[ToolParameterDetail] = Field(
        default_factory=list,
        description="Flattened list of parameters for UI display"
    )
    tool_type: Literal["builtin", "custom"] = "builtin"


class ToolSchemaListResponse(BaseModel):
    """Response schema for listing all tool schemas."""
    tools: List[ToolSchemaResponse]


class ContextVariable(BaseModel):
    """A context variable available for input mapping."""
    path: str = Field(..., description="Full path to access this variable (e.g., 'step1.results.items')")
    type: str = Field(..., description="JSON Schema type (string, array, object, etc.)")
    from_node: str = Field(..., description="Node ID that produces this output")
    description: Optional[str] = None


class NodeContextSchema(BaseModel):
    """Context variables available at a specific node."""
    node_id: str
    available_variables: List[ContextVariable] = Field(default_factory=list)


class ContextSchemaResponse(BaseModel):
    """Response schema for workflow context flow analysis."""
    nodes: Dict[str, List[ContextVariable]] = Field(
        default_factory=dict,
        description="Map of node_id to list of available context variables"
    )


class WorkflowValidationIssue(BaseModel):
    """A single validation issue."""
    severity: Literal["error", "warning", "info"] = "warning"
    node_id: Optional[str] = None
    field: Optional[str] = None
    message: str


class WorkflowValidationResponse(BaseModel):
    """Response schema for workflow validation."""
    valid: bool
    issues: List[WorkflowValidationIssue] = Field(default_factory=list)
