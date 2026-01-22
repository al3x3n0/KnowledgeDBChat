"""Add agent conversations tables for memory persistence.

Revision ID: 0015_add_agent_conversations
Revises: 0014_add_llm_task_models
Create Date: 2024-06-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0015_add_agent_conversations"
down_revision = "0014_add_llm_task_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create agent_conversations table
    op.create_table(
        "agent_conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Conversation metadata
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        # Messages stored as JSON
        sa.Column("messages", postgresql.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        # Summary for quick context loading
        sa.Column("summary", sa.Text(), nullable=True),
        # Stats
        sa.Column("message_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("tool_calls_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_conversations_user_id", "agent_conversations", ["user_id"])
    op.create_index("ix_agent_conversations_status", "agent_conversations", ["status"])
    op.create_index("ix_agent_conversations_last_message_at", "agent_conversations", ["last_message_at"])

    # Create agent_tool_executions table
    op.create_table(
        "agent_tool_executions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Tool details
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("tool_input", postgresql.JSON(), nullable=True),
        sa.Column("tool_output", postgresql.JSON(), nullable=True),
        # Execution metadata
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("execution_time_ms", sa.Integer(), nullable=True),
        # Context
        sa.Column("message_id", sa.String(length=100), nullable=True),
        # Timestamp
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["conversation_id"], ["agent_conversations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_tool_executions_conversation_id", "agent_tool_executions", ["conversation_id"])
    op.create_index("ix_agent_tool_executions_tool_name", "agent_tool_executions", ["tool_name"])


def downgrade() -> None:
    op.drop_index("ix_agent_tool_executions_tool_name", table_name="agent_tool_executions")
    op.drop_index("ix_agent_tool_executions_conversation_id", table_name="agent_tool_executions")
    op.drop_table("agent_tool_executions")

    op.drop_index("ix_agent_conversations_last_message_at", table_name="agent_conversations")
    op.drop_index("ix_agent_conversations_status", table_name="agent_conversations")
    op.drop_index("ix_agent_conversations_user_id", table_name="agent_conversations")
    op.drop_table("agent_conversations")
