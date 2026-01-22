"""Add multi-agent and memory integration support.

Revision ID: 0019_add_multi_agent_support
Revises: 0018_add_subworkflow_support
Create Date: 2025-01-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0019_add_multi_agent_support"
down_revision = "0018_add_subworkflow_support"
branch_labels = None
depends_on = None


# Default agent definitions to seed
DEFAULT_AGENTS = [
    {
        "name": "document_expert",
        "display_name": "Document Expert",
        "description": "Specializes in document operations: search, CRUD, tagging, comparison",
        "system_prompt": """You are the Document Expert, a specialized assistant for document management.
Your expertise includes:
- Searching and finding documents in the knowledge base
- Managing document metadata, tags, and organization
- Comparing and analyzing documents
- Helping users discover related content

Focus on document-related tasks. If users ask questions that require synthesizing information
from documents to answer questions, suggest that they might want to use the Q&A assistant instead.""",
        "capabilities": ["document_search", "document_crud", "document_compare", "tag_management"],
        "tool_whitelist": [
            "search_documents", "get_document_details", "delete_document",
            "create_document_from_text", "update_document_tags", "find_similar_documents",
            "compare_documents", "list_recent_documents", "search_by_tags", "list_all_tags",
            "batch_delete_documents", "read_document_content", "list_document_sources",
            "list_documents_by_source", "search_documents_by_author", "search_documents_by_tag"
        ],
        "priority": 60
    },
    {
        "name": "qa_specialist",
        "display_name": "Q&A Specialist",
        "description": "Specializes in answering questions using RAG and knowledge synthesis",
        "system_prompt": """You are the Q&A Specialist, an expert at answering questions using the knowledge base.
Your expertise includes:
- Answering questions by synthesizing information from documents
- Summarizing content and extracting key insights
- Explaining concepts using relevant documentation
- Understanding relationships between entities and topics

Focus on providing accurate, well-sourced answers. When you don't know something,
say so rather than making things up. Cite your sources when possible.""",
        "capabilities": ["rag_qa", "summarization", "knowledge_synthesis"],
        "tool_whitelist": [
            "answer_question", "search_documents", "summarize_document",
            "batch_summarize_documents", "read_document_content", "get_document_details",
            "search_entities", "get_entity_relationships", "find_documents_by_entity",
            "get_document_knowledge_graph"
        ],
        "priority": 70
    },
    {
        "name": "workflow_assistant",
        "display_name": "Workflow Assistant",
        "description": "Specializes in automation, workflows, templates, and diagrams",
        "system_prompt": """You are the Workflow Assistant, an expert in automation and productivity.
Your expertise includes:
- Creating and running automated workflows
- Filling templates with data from documents
- Generating diagrams and visualizations
- Managing custom tools and integrations

Help users automate repetitive tasks and create visual representations of information.""",
        "capabilities": ["workflow_exec", "template_fill", "diagram_gen", "automation"],
        "tool_whitelist": [
            "run_workflow", "list_workflows", "create_workflow_from_description",
            "start_template_fill", "list_template_jobs", "get_template_job_status",
            "generate_diagram", "run_custom_tool", "list_custom_tools"
        ],
        "priority": 50
    },
    {
        "name": "generalist",
        "display_name": "General Assistant",
        "description": "Handles general requests and serves as fallback when no specialist matches",
        "system_prompt": """You are the General Assistant, a helpful AI that can handle a wide variety of tasks.
You have access to all available tools and can help with any request.
When a task would be better handled by a specialist, you may suggest switching to them,
but you're also capable of handling it yourself if the user prefers.""",
        "capabilities": ["general"],
        "tool_whitelist": None,  # All tools
        "priority": 10
    }
]


def upgrade() -> None:
    # Create agent_definitions table
    op.create_table(
        "agent_definitions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), unique=True, nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("capabilities", postgresql.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("tool_whitelist", postgresql.JSON(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("50")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agent_definitions_name", "agent_definitions", ["name"])
    op.create_index("ix_agent_definitions_is_active", "agent_definitions", ["is_active"])

    # Create agent_conversation_contexts table
    op.create_table(
        "agent_conversation_contexts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_definition_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("routing_reason", sa.Text(), nullable=True),
        sa.Column("handoff_context", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["conversation_id"], ["agent_conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["agent_definition_id"], ["agent_definitions.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_agent_conversation_contexts_conversation_id", "agent_conversation_contexts", ["conversation_id"])

    # Create agent_memory_injections table
    op.create_table(
        "agent_memory_injections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("injection_type", sa.String(length=50), nullable=False, server_default=sa.text("'automatic'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["conversation_id"], ["agent_conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["memory_id"], ["conversation_memories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_memory_injections_conversation_id", "agent_memory_injections", ["conversation_id"])
    op.create_index("ix_agent_memory_injections_memory_id", "agent_memory_injections", ["memory_id"])

    # Add columns to user_preferences for memory settings
    op.add_column("user_preferences", sa.Column(
        "enable_agent_memory",
        sa.Boolean(),
        nullable=False,
        server_default=sa.text("true")
    ))
    op.add_column("user_preferences", sa.Column(
        "memory_injection_types",
        postgresql.JSON(),
        nullable=False,
        server_default=sa.text("'[\"fact\", \"preference\", \"context\"]'::json")
    ))
    op.add_column("user_preferences", sa.Column(
        "max_injected_memories",
        sa.Integer(),
        nullable=False,
        server_default=sa.text("5")
    ))

    # Add columns to agent_conversations for agent tracking
    op.add_column("agent_conversations", sa.Column(
        "active_agent_id",
        postgresql.UUID(as_uuid=True),
        nullable=True
    ))
    op.add_column("agent_conversations", sa.Column(
        "agent_handoffs",
        sa.Integer(),
        nullable=False,
        server_default=sa.text("0")
    ))
    op.create_foreign_key(
        "fk_agent_conversations_active_agent",
        "agent_conversations",
        "agent_definitions",
        ["active_agent_id"],
        ["id"],
        ondelete="SET NULL"
    )

    # Seed default agent definitions
    from uuid import uuid4
    import json

    agent_definitions_table = sa.table(
        "agent_definitions",
        sa.column("id", postgresql.UUID),
        sa.column("name", sa.String),
        sa.column("display_name", sa.String),
        sa.column("description", sa.Text),
        sa.column("system_prompt", sa.Text),
        sa.column("capabilities", postgresql.JSON),
        sa.column("tool_whitelist", postgresql.JSON),
        sa.column("priority", sa.Integer),
        sa.column("is_active", sa.Boolean),
        sa.column("is_system", sa.Boolean),
    )

    for agent in DEFAULT_AGENTS:
        op.execute(
            agent_definitions_table.insert().values(
                id=uuid4(),
                name=agent["name"],
                display_name=agent["display_name"],
                description=agent["description"],
                system_prompt=agent["system_prompt"],
                capabilities=agent["capabilities"],
                tool_whitelist=agent["tool_whitelist"],
                priority=agent["priority"],
                is_active=True,
                is_system=True,
            )
        )


def downgrade() -> None:
    # Drop foreign key on agent_conversations
    op.drop_constraint("fk_agent_conversations_active_agent", "agent_conversations", type_="foreignkey")

    # Drop added columns from agent_conversations
    op.drop_column("agent_conversations", "agent_handoffs")
    op.drop_column("agent_conversations", "active_agent_id")

    # Drop added columns from user_preferences
    op.drop_column("user_preferences", "max_injected_memories")
    op.drop_column("user_preferences", "memory_injection_types")
    op.drop_column("user_preferences", "enable_agent_memory")

    # Drop agent_memory_injections table
    op.drop_index("ix_agent_memory_injections_memory_id", table_name="agent_memory_injections")
    op.drop_index("ix_agent_memory_injections_conversation_id", table_name="agent_memory_injections")
    op.drop_table("agent_memory_injections")

    # Drop agent_conversation_contexts table
    op.drop_index("ix_agent_conversation_contexts_conversation_id", table_name="agent_conversation_contexts")
    op.drop_table("agent_conversation_contexts")

    # Drop agent_definitions table
    op.drop_index("ix_agent_definitions_is_active", table_name="agent_definitions")
    op.drop_index("ix_agent_definitions_name", table_name="agent_definitions")
    op.drop_table("agent_definitions")
