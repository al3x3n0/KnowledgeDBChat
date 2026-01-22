"""Add workflow and custom tool tables.

Revision ID: 0016_add_workflows
Revises: 0015_add_agent_conversations
Create Date: 2024-06-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0016_add_workflows"
down_revision = "0015_add_agent_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create user_tools table
    op.create_table(
        "user_tools",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Tool definition
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tool_type", sa.String(length=50), nullable=False),
        # Schema and config
        sa.Column("parameters_schema", postgresql.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
        sa.Column("config", postgresql.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        # State
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "name", name="uq_user_tool_name"),
    )
    op.create_index("ix_user_tools_user_id", "user_tools", ["user_id"])
    op.create_index("ix_user_tools_type", "user_tools", ["tool_type"])

    # Create workflows table
    op.create_table(
        "workflows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Workflow metadata
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # Trigger configuration
        sa.Column("trigger_config", postgresql.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_workflows_user_id", "workflows", ["user_id"])
    op.create_index("ix_workflows_is_active", "workflows", ["is_active"])

    # Create workflow_nodes table
    op.create_table(
        "workflow_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Node identification
        sa.Column("node_id", sa.String(length=50), nullable=False),
        sa.Column("node_type", sa.String(length=50), nullable=False),
        # Tool reference
        sa.Column("tool_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("builtin_tool", sa.String(length=100), nullable=True),
        # Configuration
        sa.Column("config", postgresql.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        # Visual position
        sa.Column("position_x", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("position_y", sa.Integer(), nullable=False, server_default=sa.text("0")),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tool_id"], ["user_tools.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("workflow_id", "node_id", name="uq_workflow_node_id"),
    )
    op.create_index("ix_workflow_nodes_workflow_id", "workflow_nodes", ["workflow_id"])

    # Create workflow_edges table
    op.create_table(
        "workflow_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Connection
        sa.Column("source_node_id", sa.String(length=50), nullable=False),
        sa.Column("target_node_id", sa.String(length=50), nullable=False),
        sa.Column("source_handle", sa.String(length=50), nullable=True),
        # Conditional routing
        sa.Column("condition", postgresql.JSON(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_workflow_edges_workflow_id", "workflow_edges", ["workflow_id"])

    # Create workflow_executions table
    op.create_table(
        "workflow_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Trigger info
        sa.Column("trigger_type", sa.String(length=50), nullable=False),
        sa.Column("trigger_data", postgresql.JSON(), nullable=True, server_default=sa.text("'{}'::json")),
        # Execution state
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("progress", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("current_node_id", sa.String(length=50), nullable=True),
        # Context
        sa.Column("context", postgresql.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        # Error
        sa.Column("error", sa.Text(), nullable=True),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_workflow_executions_workflow_id", "workflow_executions", ["workflow_id"])
    op.create_index("ix_workflow_executions_user_id", "workflow_executions", ["user_id"])
    op.create_index("ix_workflow_executions_status", "workflow_executions", ["status"])

    # Create workflow_node_executions table
    op.create_table(
        "workflow_node_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Node reference
        sa.Column("node_id", sa.String(length=50), nullable=False),
        # Execution state
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'pending'")),
        # Input/output
        sa.Column("input_data", postgresql.JSON(), nullable=True),
        sa.Column("output_data", postgresql.JSON(), nullable=True),
        # Error
        sa.Column("error", sa.Text(), nullable=True),
        # Performance
        sa.Column("execution_time_ms", sa.Integer(), nullable=True),
        # Timestamps
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["execution_id"], ["workflow_executions.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_workflow_node_executions_execution_id", "workflow_node_executions", ["execution_id"])
    op.create_index("ix_workflow_node_executions_status", "workflow_node_executions", ["status"])


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_index("ix_workflow_node_executions_status", table_name="workflow_node_executions")
    op.drop_index("ix_workflow_node_executions_execution_id", table_name="workflow_node_executions")
    op.drop_table("workflow_node_executions")

    op.drop_index("ix_workflow_executions_status", table_name="workflow_executions")
    op.drop_index("ix_workflow_executions_user_id", table_name="workflow_executions")
    op.drop_index("ix_workflow_executions_workflow_id", table_name="workflow_executions")
    op.drop_table("workflow_executions")

    op.drop_index("ix_workflow_edges_workflow_id", table_name="workflow_edges")
    op.drop_table("workflow_edges")

    op.drop_index("ix_workflow_nodes_workflow_id", table_name="workflow_nodes")
    op.drop_table("workflow_nodes")

    op.drop_index("ix_workflows_is_active", table_name="workflows")
    op.drop_index("ix_workflows_user_id", table_name="workflows")
    op.drop_table("workflows")

    op.drop_index("ix_user_tools_type", table_name="user_tools")
    op.drop_index("ix_user_tools_user_id", table_name="user_tools")
    op.drop_table("user_tools")
