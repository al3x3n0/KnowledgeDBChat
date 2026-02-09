"""
Pydantic schemas for autonomous agent jobs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AgentJobCreate(BaseModel):
    """Request schema for creating an agent job."""

    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    job_type: str = Field("custom", description="Type of job: research, monitor, analysis, synthesis, knowledge_expansion, custom")
    goal: str = Field(..., min_length=1, description="The goal for the agent to achieve")
    goal_criteria: Optional[Dict[str, Any]] = Field(None, description="Structured success criteria")
    config: Optional[Dict[str, Any]] = Field(None, description="Job-specific configuration")
    agent_definition_id: Optional[UUID] = Field(None, description="ID of agent definition to use")

    # Resource limits (optional overrides)
    max_iterations: Optional[int] = Field(None, ge=1, le=1000, description="Maximum iterations")
    max_tool_calls: Optional[int] = Field(None, ge=1, le=5000, description="Maximum tool calls")
    max_llm_calls: Optional[int] = Field(None, ge=1, le=2000, description="Maximum LLM calls")
    max_runtime_minutes: Optional[int] = Field(None, ge=1, le=480, description="Maximum runtime in minutes")

    # Scheduling
    schedule_type: Optional[str] = Field(None, description="Scheduling type: once, recurring, continuous")
    schedule_cron: Optional[str] = Field(None, description="Cron expression for recurring jobs")
    start_immediately: bool = Field(True, description="Start job immediately after creation")

    # Job chaining
    chain_config: Optional[Dict[str, Any]] = Field(None, description="Chain configuration for triggering child jobs")
    # Structure:
    # {
    #   "trigger_condition": "on_complete",  # on_complete, on_fail, on_any_end, on_progress, on_findings
    #   "progress_threshold": 50,            # For on_progress trigger
    #   "findings_threshold": 10,            # For on_findings trigger
    #   "inherit_results": true,             # Pass parent results to child
    #   "inherit_config": false,             # Inherit parent config
    #   "child_jobs": [
    #     {
    #       "name": "Follow-up Analysis",
    #       "job_type": "analysis",
    #       "goal": "Analyze the research findings",
    #       "config": {...}
    #     }
    #   ]
    # }
    parent_job_id: Optional[UUID] = Field(None, description="ID of parent job (for manually chained jobs)")


class AgentJobFromTemplate(BaseModel):
    """Request schema for creating a job from a template."""

    template_id: UUID = Field(..., description="ID of the template to use")
    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    goal: Optional[str] = Field(None, description="Override the default goal")
    config: Optional[Dict[str, Any]] = Field(None, description="Override template config")
    start_immediately: bool = Field(True, description="Start job immediately")
    chain_config: Optional[Dict[str, Any]] = Field(None, description="Optional chain configuration (triggers child jobs)")


class AgentJobUpdate(BaseModel):
    """Request schema for updating an agent job."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    goal: Optional[str] = None
    goal_criteria: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = Field(None, ge=1, le=1000)
    max_tool_calls: Optional[int] = Field(None, ge=1, le=5000)
    max_llm_calls: Optional[int] = Field(None, ge=1, le=2000)
    max_runtime_minutes: Optional[int] = Field(None, ge=1, le=480)
    schedule_type: Optional[str] = None
    schedule_cron: Optional[str] = None


class AgentJobLogEntry(BaseModel):
    """Schema for a job log entry."""

    iteration: int
    phase: str
    timestamp: str
    thought: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class AgentJobResponse(BaseModel):
    """Response schema for an agent job."""

    id: UUID
    name: str
    description: Optional[str]
    job_type: str
    goal: str
    goal_criteria: Optional[Dict[str, Any]]
    config: Optional[Dict[str, Any]]

    # Agent assignment
    agent_definition_id: Optional[UUID]
    agent_definition_name: Optional[str] = None

    # Ownership
    user_id: UUID

    # Status and progress
    status: str
    progress: int
    current_phase: Optional[str]
    phase_details: Optional[str]

    # Execution tracking
    iteration: int
    max_iterations: int

    # Resource limits
    max_tool_calls: int
    max_llm_calls: int
    max_runtime_minutes: int

    # Usage tracking
    tool_calls_used: int
    llm_calls_used: int
    tokens_used: int

    # Error tracking
    error: Optional[str]
    error_count: int

    # Scheduling
    schedule_type: Optional[str]
    schedule_cron: Optional[str]
    next_run_at: Optional[datetime]

    # Results
    results: Optional[Dict[str, Any]]
    output_artifacts: Optional[List[Dict[str, Any]]]

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    last_activity_at: Optional[datetime]

    # Celery task tracking
    celery_task_id: Optional[str]

    # Job chaining
    parent_job_id: Optional[UUID] = None
    root_job_id: Optional[UUID] = None
    chain_depth: int = 0
    chain_triggered: bool = False
    chain_config: Optional[Dict[str, Any]] = None
    swarm_summary: Optional[Dict[str, Any]] = None
    goal_contract_summary: Optional[Dict[str, Any]] = None
    approval_checkpoint: Optional[Dict[str, Any]] = None
    executive_digest: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class AgentJobListResponse(BaseModel):
    """Response schema for listing agent jobs."""

    jobs: List[AgentJobResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class AgentJobDetailResponse(AgentJobResponse):
    """Detailed response schema including execution log."""

    execution_log: Optional[List[Dict[str, Any]]]


class AgentJobTemplateResponse(BaseModel):
    """Response schema for an agent job template."""

    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    category: Optional[str]
    job_type: str
    default_goal: Optional[str]
    default_config: Optional[Dict[str, Any]]
    default_chain_config: Optional[Dict[str, Any]] = None
    agent_definition_id: Optional[UUID]

    # Resource defaults
    default_max_iterations: int
    default_max_tool_calls: int
    default_max_llm_calls: int
    default_max_runtime_minutes: int

    # Visibility
    is_system: bool
    is_active: bool
    owner_user_id: Optional[UUID]

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentJobTemplateListResponse(BaseModel):
    """Response schema for listing job templates."""

    templates: List[AgentJobTemplateResponse]
    total: int


class AgentJobProgressUpdate(BaseModel):
    """WebSocket message for job progress updates."""

    type: str = "progress"
    job_id: str
    progress: int
    phase: str
    status: str
    iteration: int
    phase_details: Optional[str]
    error: Optional[str]
    timestamp: str


class AgentJobActionRequest(BaseModel):
    """Request schema for job actions (pause, resume, cancel)."""

    action: str = Field(..., description="Action to perform: pause, resume, cancel")


class AgentJobFeedbackCreate(BaseModel):
    """Create human feedback that can tune future autonomous behavior."""

    rating: int = Field(..., ge=1, le=5, description="User rating for this output/checkpoint")
    feedback: Optional[str] = Field(None, max_length=4000, description="Optional feedback text")
    target_type: str = Field("job", description="Target: job, checkpoint, finding, action, or tool")
    target_id: Optional[str] = Field(None, max_length=200, description="Optional target identifier")
    scope: str = Field("user", description="Learning scope: user, customer, or team")
    team_key: Optional[str] = Field(None, max_length=120, description="Team key when scope=team")
    preferred_tools: Optional[List[str]] = Field(default_factory=list, description="Tools to favor in future runs")
    discouraged_tools: Optional[List[str]] = Field(default_factory=list, description="Tools to avoid in future runs")
    checkpoint: Optional[str] = Field(None, max_length=200, description="Checkpoint label if feedback is checkpoint-specific")


class AgentJobFeedbackResponse(BaseModel):
    """Stored human feedback item."""

    id: UUID
    job_id: Optional[UUID] = None
    rating: int
    feedback: Optional[str]
    target_type: str
    target_id: Optional[str]
    scope: str
    preferred_tools: List[str] = Field(default_factory=list)
    discouraged_tools: List[str] = Field(default_factory=list)
    checkpoint: Optional[str]
    created_at: Optional[datetime]


class AgentJobFeedbackListResponse(BaseModel):
    """Paginated feedback list for a job or user scope."""

    items: List[AgentJobFeedbackResponse]
    total: int


class AgentJobStatsResponse(BaseModel):
    """Response schema for job statistics."""

    total_jobs: int
    running_jobs: int
    pending_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_iterations: int
    total_tool_calls: int
    total_llm_calls: int
    avg_completion_time_minutes: Optional[float]
    success_rate: Optional[float]


class AgentJobCheckpointResponse(BaseModel):
    """Response schema for a job checkpoint."""

    id: UUID
    job_id: UUID
    iteration: int
    phase: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# Chain Definition schemas
class ChainStepConfig(BaseModel):
    """Configuration for a single step in a job chain."""

    step_name: str = Field(..., description="Name of this step")
    template_id: Optional[UUID] = Field(None, description="Optional template to use")
    job_type: str = Field("custom", description="Job type for this step")
    goal_template: str = Field(..., description="Goal template with {variable} placeholders")
    config: Optional[Dict[str, Any]] = Field(None, description="Step-specific configuration")
    trigger_condition: str = Field("on_complete", description="When to trigger next step")
    trigger_thresholds: Optional[Dict[str, int]] = Field(None, description="Thresholds for progress/findings triggers")


class AgentJobChainDefinitionCreate(BaseModel):
    """Request schema for creating a chain definition."""

    name: str = Field(..., min_length=1, max_length=100, description="Unique chain name")
    display_name: str = Field(..., min_length=1, max_length=200, description="Display name")
    description: Optional[str] = Field(None, description="Chain description")
    chain_steps: List[ChainStepConfig] = Field(..., min_length=1, description="Ordered list of chain steps")
    default_settings: Optional[Dict[str, Any]] = Field(None, description="Default settings for all jobs")


class AgentJobChainDefinitionUpdate(BaseModel):
    """Request schema for updating a chain definition."""

    display_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    chain_steps: Optional[List[ChainStepConfig]] = Field(None, min_length=1)
    default_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class AgentJobChainDefinitionResponse(BaseModel):
    """Response schema for a chain definition."""

    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    chain_steps: List[Dict[str, Any]]
    default_settings: Optional[Dict[str, Any]]
    owner_user_id: Optional[UUID]
    is_system: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentJobChainDefinitionListResponse(BaseModel):
    """Response schema for listing chain definitions."""

    chains: List[AgentJobChainDefinitionResponse]
    total: int


class AgentJobFromChainCreate(BaseModel):
    """Request schema for creating a job chain from a definition."""

    chain_definition_id: UUID = Field(..., description="ID of the chain definition to use")
    name_prefix: str = Field(..., min_length=1, max_length=150, description="Prefix for job names")
    variables: Dict[str, str] = Field(default_factory=dict, description="Variables to substitute in goal templates")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Override chain default settings")
    start_immediately: bool = Field(True, description="Start first job immediately")


class AgentJobSaveAsChainRequest(BaseModel):
    """Request schema for saving an executed job chain as a reusable chain definition (playbook)."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=100, description="Unique chain name (optional)")
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=200, description="Display name (optional)")
    description: Optional[str] = Field(default=None, max_length=2000, description="Description (optional)")


class AgentJobChainStatusResponse(BaseModel):
    """Response schema for chain status."""

    root_job_id: UUID
    chain_definition_id: Optional[UUID]
    total_steps: int
    completed_steps: int
    current_step: int
    overall_progress: int
    status: str  # pending, running, completed, failed, partially_completed
    jobs: List[AgentJobResponse]
