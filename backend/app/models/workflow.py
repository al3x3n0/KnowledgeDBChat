"""
Workflow and Custom Tool database models.

Supports user-defined tools and visual workflow automation.
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Boolean, Integer, DateTime, ForeignKey, Index,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class UserTool(Base):
    """
    User-defined custom tool that can be used in workflows or by the agent.

    Tool types:
    - webhook: HTTP request to external API
    - transform: Data transformation using Jinja2/JSONPath
    - python: Sandboxed Python code execution
    - llm_prompt: LLM call with templated prompt
    """
    __tablename__ = "user_tools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Tool definition
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    tool_type = Column(String(50), nullable=False)  # webhook, transform, python, llm_prompt

    # JSON Schema for input parameters
    parameters_schema = Column(JSON, nullable=True, default=dict)

    # Type-specific configuration
    config = Column(JSON, nullable=False, default=dict)

    # State
    is_enabled = Column(Boolean, default=True, nullable=False)
    version = Column(Integer, default=1, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="custom_tools")
    workflow_nodes = relationship("WorkflowNode", back_populates="tool", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_user_tool_name"),
        Index("ix_user_tools_user_id", "user_id"),
        Index("ix_user_tools_type", "tool_type"),
    )


class Workflow(Base):
    """
    Workflow definition containing nodes and edges.

    Supports:
    - Linear execution
    - Conditional branching
    - Parallel execution
    - Loops
    """
    __tablename__ = "workflows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Workflow metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    # Trigger configuration
    # {
    #   "type": "manual" | "schedule" | "event" | "webhook",
    #   "schedule": "0 9 * * *",  # cron for scheduled
    #   "event": "document.uploaded",  # for event triggers
    #   "webhook_secret": "xxx"  # for webhook triggers
    # }
    trigger_config = Column(JSON, nullable=True, default=dict)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="workflows")
    nodes = relationship("WorkflowNode", back_populates="workflow", cascade="all, delete-orphan")
    edges = relationship("WorkflowEdge", back_populates="workflow", cascade="all, delete-orphan")
    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_workflows_user_id", "user_id"),
        Index("ix_workflows_is_active", "is_active"),
    )


class WorkflowNode(Base):
    """
    A node in a workflow graph.

    Node types:
    - start: Entry point (exactly one per workflow)
    - end: Exit point (at least one per workflow)
    - tool: Execute a tool (custom or built-in)
    - condition: Branch based on condition
    - parallel: Fork into parallel branches
    - loop: Iterate over items
    - wait: Pause for duration or event
    """
    __tablename__ = "workflow_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False)

    # Node identifier (unique within workflow, e.g., "node_1", "start", "end")
    node_id = Column(String(50), nullable=False)

    # Node type
    node_type = Column(String(50), nullable=False)

    # Tool reference (for tool nodes)
    tool_id = Column(UUID(as_uuid=True), ForeignKey("user_tools.id", ondelete="SET NULL"), nullable=True)
    builtin_tool = Column(String(100), nullable=True)  # Name of built-in agent tool

    # Node configuration
    # {
    #   "input_mapping": {"param1": "{{context.prev_output.field}}"},
    #   "output_key": "step1_result",
    #   "condition": {...},  # for condition nodes
    #   "loop_source": "{{context.items}}",  # for loop nodes
    #   "wait_seconds": 60,  # for wait nodes
    # }
    config = Column(JSON, nullable=False, default=dict)

    # Visual position in editor
    position_x = Column(Integer, default=0, nullable=False)
    position_y = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    workflow = relationship("Workflow", back_populates="nodes")
    tool = relationship("UserTool", back_populates="workflow_nodes")

    __table_args__ = (
        UniqueConstraint("workflow_id", "node_id", name="uq_workflow_node_id"),
        Index("ix_workflow_nodes_workflow_id", "workflow_id"),
    )


class WorkflowEdge(Base):
    """
    An edge connecting two nodes in a workflow.

    Supports conditional routing via the condition field.
    """
    __tablename__ = "workflow_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False)

    # Connection
    source_node_id = Column(String(50), nullable=False)
    target_node_id = Column(String(50), nullable=False)

    # For nodes with multiple outputs (e.g., condition: "true", "false")
    source_handle = Column(String(50), nullable=True)

    # Conditional routing
    # {
    #   "type": "expression",
    #   "expression": "{{context.value}} > 10"
    # }
    condition = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    workflow = relationship("Workflow", back_populates="edges")

    __table_args__ = (
        Index("ix_workflow_edges_workflow_id", "workflow_id"),
    )


class WorkflowExecution(Base):
    """
    A single execution instance of a workflow.

    Tracks overall progress and stores shared context.
    """
    __tablename__ = "workflow_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Parent execution for sub-workflow tracking
    parent_execution_id = Column(UUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="SET NULL"), nullable=True)

    # Trigger info
    trigger_type = Column(String(50), nullable=False)  # manual, schedule, event, webhook, subworkflow
    trigger_data = Column(JSON, nullable=True, default=dict)

    # Execution state
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed, paused, cancelled
    progress = Column(Integer, default=0, nullable=False)  # 0-100
    current_node_id = Column(String(50), nullable=True)

    # Shared execution context (variables, outputs from nodes)
    context = Column(JSON, nullable=False, default=dict)

    # Error info
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    user = relationship("User", back_populates="workflow_executions")
    node_executions = relationship("WorkflowNodeExecution", back_populates="execution", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_workflow_executions_workflow_id", "workflow_id"),
        Index("ix_workflow_executions_user_id", "user_id"),
        Index("ix_workflow_executions_status", "status"),
    )


class WorkflowNodeExecution(Base):
    """
    Execution log for a single node within a workflow execution.

    Provides detailed tracking of each step.
    """
    __tablename__ = "workflow_node_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    execution_id = Column(UUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False)

    # Node reference
    node_id = Column(String(50), nullable=False)

    # Execution state
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed, skipped

    # Input/output data
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)

    # Error info
    error = Column(Text, nullable=True)

    # Performance metrics
    execution_time_ms = Column(Integer, nullable=True)

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    execution = relationship("WorkflowExecution", back_populates="node_executions")

    __table_args__ = (
        Index("ix_workflow_node_executions_execution_id", "execution_id"),
        Index("ix_workflow_node_executions_status", "status"),
    )
