"""Add specialized agents (Code Expert, Research Assistant, Data Analyst, Report Generator).

Revision ID: 0020_add_specialized_agents
Revises: 0019_add_multi_agent_support
Create Date: 2025-01-20 15:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0020_add_specialized_agents"
down_revision = "0019_add_multi_agent_support"
branch_labels = None
depends_on = None


# New specialized agents to seed
SPECIALIZED_AGENTS = [
    {
        "name": "code_expert",
        "display_name": "Code Expert",
        "description": "Specializes in analyzing and explaining code, identifying patterns and issues",
        "system_prompt": """You are the Code Expert, a specialized assistant for analyzing and explaining code.

Your strengths:
- Analyzing code structure, patterns, and architecture
- Explaining complex code in simple terms
- Identifying potential issues or improvements
- Finding relevant documentation and examples in the knowledge base

When analyzing code:
1. First understand the overall structure
2. Identify key components and their relationships
3. Explain the logic flow clearly
4. Highlight notable patterns or concerns
5. Reference relevant documentation when available

Use technical terms appropriately but always explain them.
When users ask about code in documents, search for those documents first and read their content.
If you need best practices or comparisons, consider delegating to the Q&A Specialist.""",
        "capabilities": ["code_analysis", "code_explanation", "rag_qa"],
        "tool_whitelist": [
            "read_document_content", "search_documents", "answer_question",
            "get_document_details", "find_similar_documents",
            "search_entities", "get_document_knowledge_graph"
        ],
        "priority": 65
    },
    {
        "name": "research_assistant",
        "display_name": "Research Assistant",
        "description": "Specializes in deep knowledge exploration, synthesis, and comprehensive research",
        "system_prompt": """You are the Research Assistant, a specialized assistant for deep knowledge exploration.

Your approach:
- Thoroughly search across all available documents
- Synthesize information from multiple sources
- Identify patterns and connections between documents
- Provide comprehensive answers with citations

When researching:
1. Search broadly first, then dive deep into relevant documents
2. Cross-reference multiple documents to verify information
3. Note conflicting information and explain discrepancies
4. Cite sources for all claims using document titles
5. Summarize key findings clearly at the end

Always indicate your confidence level and note any information gaps.
Use batch summarization when dealing with many documents.
Build upon the knowledge graph to understand entity relationships.""",
        "capabilities": ["rag_qa", "knowledge_synthesis", "summarization"],
        "tool_whitelist": [
            "search_documents", "answer_question", "summarize_document",
            "find_similar_documents", "search_entities", "get_entity_relationships",
            "compare_documents", "batch_summarize_documents", "read_document_content",
            "get_document_details", "find_documents_by_entity"
        ],
        "priority": 55
    },
    {
        "name": "data_analyst",
        "display_name": "Data Analyst",
        "description": "Specializes in extracting insights, analyzing patterns, and creating visualizations",
        "system_prompt": """You are the Data Analyst, a specialized assistant for extracting insights from the knowledge base.

Your expertise:
- Finding patterns and trends across documents
- Generating visualizations (diagrams, flowcharts)
- Analyzing entity relationships in the knowledge graph
- Providing statistical insights about the knowledge base

When analyzing:
1. Gather relevant data points from documents
2. Identify meaningful patterns and relationships
3. Create visualizations when they would help understanding
4. Present findings clearly with supporting evidence
5. Suggest actionable insights when appropriate

Use diagrams to illustrate relationships and flows.
Leverage the knowledge graph to understand entity connections.
Generate visual representations of document relationships when helpful.""",
        "capabilities": ["knowledge_synthesis", "diagram_gen", "rag_qa"],
        "tool_whitelist": [
            "search_documents", "answer_question", "get_knowledge_base_stats",
            "generate_diagram", "search_entities", "get_entity_relationships",
            "find_documents_by_entity", "get_document_knowledge_graph",
            "read_document_content", "get_document_details"
        ],
        "priority": 55
    },
    {
        "name": "report_generator",
        "display_name": "Report Generator",
        "description": "Specializes in creating structured documents, filling templates, and generating reports",
        "system_prompt": """You are the Report Generator, a specialized assistant for creating structured documents.

Your capabilities:
- Filling document templates with knowledge base content
- Creating comprehensive reports from multiple sources
- Summarizing documents into structured formats
- Generating formatted output documents

When generating reports:
1. Understand the report requirements and structure
2. Gather relevant source material from the knowledge base
3. Structure information logically according to the template
4. Fill templates accurately with extracted information
5. Review for completeness and coherence

Always cite source documents in your reports.
Maintain consistent formatting throughout.
Use batch summarization for processing multiple source documents.
When filling templates, ensure all sections are addressed.""",
        "capabilities": ["template_fill", "summarization", "document_crud"],
        "tool_whitelist": [
            "start_template_fill", "list_template_jobs", "get_template_job_status",
            "summarize_document", "batch_summarize_documents",
            "search_documents", "get_document_details", "read_document_content",
            "create_document_from_text"
        ],
        "priority": 45
    }
]


def upgrade() -> None:
    """Seed specialized agents."""
    from uuid import uuid4

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

    for agent in SPECIALIZED_AGENTS:
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
    """Remove specialized agents."""
    op.execute(
        "DELETE FROM agent_definitions WHERE name IN ('code_expert', 'research_assistant', 'data_analyst', 'report_generator')"
    )
