"""
Autonomous Agent Job models.

Tracks autonomous agent executions that run independently in the background,
working toward defined goals without requiring continuous user interaction.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.core.database import Base


class AgentJobStatus(str, enum.Enum):
    """Status of an autonomous agent job."""
    PENDING = "pending"          # Job created, waiting to start
    RUNNING = "running"          # Currently executing
    PAUSED = "paused"           # Paused by user or system
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"           # Failed with error
    CANCELLED = "cancelled"      # Cancelled by user


class AgentJobType(str, enum.Enum):
    """Type of autonomous agent job."""
    RESEARCH = "research"                    # Research a topic, find papers, synthesize
    MONITOR = "monitor"                      # Monitor for changes/updates
    ANALYSIS = "analysis"                    # Analyze documents/data
    SYNTHESIS = "synthesis"                  # Synthesize information from sources
    KNOWLEDGE_EXPANSION = "knowledge_expansion"  # Expand knowledge base
    DATA_ANALYSIS = "data_analysis"          # Data analysis, ETL, visualization
    CUSTOM = "custom"                        # Custom goal-driven task


class ChainTriggerCondition(str, enum.Enum):
    """Conditions for triggering chained jobs."""
    ON_COMPLETE = "on_complete"              # Trigger when parent completes successfully
    ON_FAIL = "on_fail"                      # Trigger when parent fails
    ON_ANY_END = "on_any_end"                # Trigger on any completion (success or failure)
    ON_PROGRESS = "on_progress"              # Trigger when parent reaches progress threshold
    ON_FINDINGS = "on_findings"              # Trigger when parent finds certain number of findings


class AgentJob(Base):
    """
    Autonomous agent job that runs independently.

    An AgentJob represents a goal-driven task that an agent works on
    autonomously, potentially over multiple execution cycles.
    """

    __tablename__ = "agent_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Job identification
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Job type and configuration
    job_type = Column(String(50), nullable=False, default=AgentJobType.CUSTOM.value)

    # Goal definition - what the agent should accomplish
    goal = Column(Text, nullable=False)  # Natural language goal
    goal_criteria = Column(JSON, nullable=True)  # Structured success criteria

    # Configuration
    config = Column(JSON, nullable=True)  # Job-specific configuration
    # Example config for research job:
    # {
    #   "topics": ["machine learning", "transformers"],
    #   "sources": ["arxiv", "documents"],
    #   "max_papers": 50,
    #   "synthesis_format": "report",
    #   "depth": "comprehensive"
    # }

    # Agent assignment
    agent_definition_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_definitions.id", ondelete="SET NULL"),
        nullable=True
    )
    agent_definition = relationship("AgentDefinition")

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    user = relationship("User", backref="agent_jobs")

    # Status and progress
    status = Column(String(20), nullable=False, default=AgentJobStatus.PENDING.value)
    progress = Column(Integer, default=0)  # 0-100
    current_phase = Column(String(100), nullable=True)  # Current execution phase
    phase_details = Column(Text, nullable=True)  # Details about current phase

    # Execution tracking
    iteration = Column(Integer, default=0)  # Current iteration of the autonomous loop
    max_iterations = Column(Integer, default=100)  # Maximum iterations before stopping

    # Execution log - stores the agent's thought process and actions
    execution_log = Column(JSON, nullable=True)
    # Structure: [
    #   {"iteration": 1, "phase": "planning", "thought": "...", "action": "...", "result": "...", "timestamp": "..."},
    #   ...
    # ]

    # Results and outputs
    results = Column(JSON, nullable=True)  # Structured results
    # Example results for research job:
    # {
    #   "papers_found": 45,
    #   "papers_analyzed": 30,
    #   "key_findings": [...],
    #   "synthesis_document_id": "...",
    #   "knowledge_graph_nodes_added": 150
    # }

    output_artifacts = Column(JSON, nullable=True)  # References to created artifacts
    # Example: [
    #   {"type": "document", "id": "...", "title": "Research Synthesis"},
    #   {"type": "presentation", "id": "...", "title": "Research Summary"}
    # ]

    # Error tracking
    error = Column(Text, nullable=True)
    error_count = Column(Integer, default=0)
    last_error_at = Column(DateTime(timezone=True), nullable=True)

    # Scheduling
    schedule_type = Column(String(20), nullable=True)  # "once", "recurring", "continuous"
    schedule_cron = Column(String(100), nullable=True)  # Cron expression for recurring
    next_run_at = Column(DateTime(timezone=True), nullable=True)

    # Resource limits
    max_tool_calls = Column(Integer, default=500)  # Max tool calls per job
    max_llm_calls = Column(Integer, default=200)   # Max LLM calls per job
    max_runtime_minutes = Column(Integer, default=60)  # Max runtime in minutes

    # Usage tracking
    tool_calls_used = Column(Integer, default=0)
    llm_calls_used = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)

    # Celery task tracking
    celery_task_id = Column(String(100), nullable=True)

    # Job chaining - allows jobs to trigger other jobs on completion
    parent_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True
    )
    # Self-referential relationship for chained jobs
    chained_jobs = relationship(
        "AgentJob",
        backref="parent_job",
        remote_side=[id],
        foreign_keys=[parent_job_id]
    )

    # Chain configuration
    chain_config = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "trigger_condition": "on_complete",  # ChainTriggerCondition value
    #   "progress_threshold": 50,            # For ON_PROGRESS trigger
    #   "findings_threshold": 10,            # For ON_FINDINGS trigger
    #   "inherit_results": true,             # Pass parent results to child
    #   "inherit_config": false,             # Inherit parent config
    #   "chain_data": {...}                  # Custom data passed to child
    # }

    # Chain status tracking
    chain_triggered = Column(Boolean, default=False)  # Whether this job has triggered its children
    chain_depth = Column(Integer, default=0)  # Depth in chain hierarchy (0 = root)
    root_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="SET NULL"),
        nullable=True
    )  # Reference to the original root job in a chain

    # Memory integration
    # Enable jobs to use memories from past jobs and store new memories
    enable_memory = Column(Boolean, default=True)  # Whether to use memory for this job
    memory_injection_count = Column(Integer, default=0)  # Number of memories injected
    memories_created_count = Column(Integer, default=0)  # Number of memories created

    # Memories relationship (defined in ConversationMemory)
    memories = relationship("ConversationMemory", back_populates="source_job", foreign_keys="ConversationMemory.job_id")

    def __repr__(self):
        return f"<AgentJob(id={self.id}, name='{self.name}', status={self.status})>"

    def can_continue(self) -> bool:
        """Check if the job can continue executing."""
        if self.status not in [AgentJobStatus.RUNNING.value, AgentJobStatus.PENDING.value]:
            return False
        if self.iteration >= self.max_iterations:
            return False
        if self.tool_calls_used >= self.max_tool_calls:
            return False
        if self.llm_calls_used >= self.max_llm_calls:
            return False
        return True

    def is_resource_limited(self) -> tuple[bool, str]:
        """Check if job has hit resource limits."""
        if self.iteration >= self.max_iterations:
            return True, "max_iterations"
        if self.tool_calls_used >= self.max_tool_calls:
            return True, "max_tool_calls"
        if self.llm_calls_used >= self.max_llm_calls:
            return True, "max_llm_calls"
        return False, ""

    def add_log_entry(self, entry: Dict[str, Any]):
        """Add an entry to the execution log."""
        if self.execution_log is None:
            self.execution_log = []
        entry["timestamp"] = datetime.utcnow().isoformat()
        entry["iteration"] = self.iteration
        self.execution_log.append(entry)

    def get_recent_log(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent log entries."""
        if not self.execution_log:
            return []
        return self.execution_log[-count:]

    # Chain management methods
    def get_chain_trigger_condition(self) -> Optional[ChainTriggerCondition]:
        """Get the trigger condition for this job's chained jobs."""
        if not self.chain_config:
            return None
        condition = self.chain_config.get("trigger_condition")
        if condition:
            return ChainTriggerCondition(condition)
        return ChainTriggerCondition.ON_COMPLETE

    def should_trigger_chain(self, event: str, value: int = 0) -> bool:
        """
        Check if chained jobs should be triggered based on an event.

        Args:
            event: Event type ('complete', 'fail', 'progress', 'findings')
            value: Event value (progress percentage or findings count)

        Returns:
            True if chain should be triggered
        """
        if self.chain_triggered:
            return False  # Already triggered

        condition = self.get_chain_trigger_condition()
        if not condition:
            return False

        config = self.chain_config or {}

        if event == "complete" and condition in [
            ChainTriggerCondition.ON_COMPLETE,
            ChainTriggerCondition.ON_ANY_END
        ]:
            return True

        if event == "fail" and condition in [
            ChainTriggerCondition.ON_FAIL,
            ChainTriggerCondition.ON_ANY_END
        ]:
            return True

        if event == "progress" and condition == ChainTriggerCondition.ON_PROGRESS:
            threshold = config.get("progress_threshold", 50)
            return value >= threshold

        if event == "findings" and condition == ChainTriggerCondition.ON_FINDINGS:
            threshold = config.get("findings_threshold", 10)
            return value >= threshold

        return False

    def get_chain_data_for_child(self) -> Dict[str, Any]:
        """Get data to pass to chained child jobs."""
        config = self.chain_config or {}
        data = {
            "parent_job_id": str(self.id),
            "parent_job_name": self.name,
            "chain_depth": self.chain_depth + 1,
            "root_job_id": str(self.root_job_id or self.id),
        }

        # Optionally inherit results
        if config.get("inherit_results", True) and self.results:
            data["parent_results"] = self.results

        # Optionally inherit config
        if config.get("inherit_config", False) and self.config:
            data["inherited_config"] = self.config

        # Add custom chain data
        if config.get("chain_data"):
            data["chain_data"] = config["chain_data"]

        return data

    def is_chain_root(self) -> bool:
        """Check if this job is the root of a chain."""
        return self.parent_job_id is None and self.chain_depth == 0

    def get_chain_hierarchy(self) -> List[str]:
        """Get the chain hierarchy from root to this job."""
        hierarchy = []
        if self.root_job_id:
            hierarchy.append(str(self.root_job_id))
        if self.parent_job_id and self.parent_job_id != self.root_job_id:
            hierarchy.append(str(self.parent_job_id))
        hierarchy.append(str(self.id))
        return hierarchy


class AgentJobCheckpoint(Base):
    """
    Checkpoint for an autonomous agent job.

    Allows resuming jobs from a known good state if interrupted.
    """

    __tablename__ = "agent_job_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Job reference
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_jobs.id", ondelete="CASCADE"),
        nullable=False
    )
    job = relationship("AgentJob", backref="checkpoints")

    # Checkpoint data
    iteration = Column(Integer, nullable=False)
    phase = Column(String(100), nullable=True)
    state = Column(JSON, nullable=False)  # Serialized agent state

    # Context for resumption
    context = Column(JSON, nullable=True)  # Additional context needed to resume

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return f"<AgentJobCheckpoint(job_id={self.job_id}, iteration={self.iteration})>"


class AgentJobTemplate(Base):
    """
    Template for creating autonomous agent jobs.

    Pre-configured job templates for common autonomous tasks.
    """

    __tablename__ = "agent_job_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Template identification
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True)  # research, monitoring, analysis

    # Template configuration
    job_type = Column(String(50), nullable=False)
    default_goal = Column(Text, nullable=True)
    default_config = Column(JSON, nullable=True)

    # Agent to use (optional)
    agent_definition_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_definitions.id", ondelete="SET NULL"),
        nullable=True
    )

    # Resource defaults
    default_max_iterations = Column(Integer, default=100)
    default_max_tool_calls = Column(Integer, default=500)
    default_max_llm_calls = Column(Integer, default=200)
    default_max_runtime_minutes = Column(Integer, default=60)

    # Visibility
    is_system = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Ownership (null for system templates)
    owner_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True
    )

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AgentJobTemplate(name='{self.name}', type={self.job_type})>"


class AgentJobChainDefinition(Base):
    """
    Definition for a chain of autonomous agent jobs.

    Defines a sequence of jobs that execute in order, with configurable
    trigger conditions between each step.
    """

    __tablename__ = "agent_job_chain_definitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Chain identification
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Chain structure - ordered list of job configs and trigger conditions
    chain_steps = Column(JSON, nullable=False)
    # Structure:
    # [
    #   {
    #     "step_name": "Initial Research",
    #     "template_id": "uuid" or null,  # Optional template to use
    #     "job_type": "research",          # Job type
    #     "goal_template": "Research {topic}",  # Goal with placeholders
    #     "config": {...},                  # Job-specific config
    #     "trigger_condition": "on_complete",  # When to trigger next
    #     "trigger_thresholds": {...}       # Thresholds for progress/findings triggers
    #   },
    #   ...
    # ]

    # Default settings applied to all jobs in chain
    default_settings = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "max_iterations": 50,
    #   "max_runtime_minutes": 30,
    #   "inherit_results": true
    # }

    # Ownership
    owner_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True
    )
    is_system = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AgentJobChainDefinition(name='{self.name}')>"

    def get_step_count(self) -> int:
        """Get the number of steps in this chain."""
        return len(self.chain_steps) if self.chain_steps else 0

    def get_step(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific step by index."""
        if not self.chain_steps or index >= len(self.chain_steps):
            return None
        return self.chain_steps[index]

    def create_job_config_for_step(
        self,
        step_index: int,
        variables: Dict[str, str],
        parent_results: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create job configuration for a specific step.

        Args:
            step_index: Index of the step in the chain
            variables: Variables to substitute in goal_template (e.g., {"topic": "AI"})
            parent_results: Results from parent job (if any)

        Returns:
            Job configuration dict ready to create an AgentJob
        """
        step = self.get_step(step_index)
        if not step:
            return None

        # Start with default settings
        config = dict(self.default_settings) if self.default_settings else {}

        # Apply step-specific config
        if step.get("config"):
            config.update(step["config"])

        # Build goal from template
        goal_template = step.get("goal_template", "")
        goal = goal_template
        for key, value in variables.items():
            goal = goal.replace(f"{{{key}}}", value)

        # Build job config
        job_config = {
            "name": step.get("step_name", f"Chain Step {step_index + 1}"),
            "job_type": step.get("job_type", "custom"),
            "goal": goal,
            "config": config,
        }

        # Add template reference if specified
        if step.get("template_id"):
            job_config["template_id"] = step["template_id"]

        # Add chain configuration for next step trigger
        if step_index < len(self.chain_steps) - 1:
            next_step = self.chain_steps[step_index + 1]
            job_config["chain_config"] = {
                "trigger_condition": step.get("trigger_condition", "on_complete"),
                "inherit_results": config.get("inherit_results", True),
            }
            if step.get("trigger_thresholds"):
                job_config["chain_config"].update(step["trigger_thresholds"])

        # Pass parent results if configured
        if parent_results and config.get("inherit_results", True):
            job_config["inherited_data"] = {
                "parent_results": parent_results
            }

        return job_config
