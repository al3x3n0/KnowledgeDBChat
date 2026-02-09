"""Add autonomous agent jobs tables.

Revision ID: 0034_add_autonomous_agent_jobs
Revises: 0033_add_mcp_configuration
Create Date: 2025-01-28

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '0034_add_autonomous_agent_jobs'
down_revision = '0033_add_mcp_configuration'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create agent_jobs table
    op.create_table(
        'agent_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Job identification
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),

        # Job type and configuration
        sa.Column('job_type', sa.String(50), nullable=False, server_default='custom'),

        # Goal definition
        sa.Column('goal', sa.Text, nullable=False),
        sa.Column('goal_criteria', postgresql.JSON, nullable=True),

        # Configuration
        sa.Column('config', postgresql.JSON, nullable=True),

        # Agent assignment
        sa.Column('agent_definition_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('agent_definitions.id', ondelete='SET NULL'), nullable=True),

        # Ownership
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),

        # Status and progress
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer, server_default='0'),
        sa.Column('current_phase', sa.String(100), nullable=True),
        sa.Column('phase_details', sa.Text, nullable=True),

        # Execution tracking
        sa.Column('iteration', sa.Integer, server_default='0'),
        sa.Column('max_iterations', sa.Integer, server_default='100'),
        sa.Column('execution_log', postgresql.JSON, nullable=True),

        # Results and outputs
        sa.Column('results', postgresql.JSON, nullable=True),
        sa.Column('output_artifacts', postgresql.JSON, nullable=True),

        # Error tracking
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('error_count', sa.Integer, server_default='0'),
        sa.Column('last_error_at', sa.DateTime(timezone=True), nullable=True),

        # Scheduling
        sa.Column('schedule_type', sa.String(20), nullable=True),
        sa.Column('schedule_cron', sa.String(100), nullable=True),
        sa.Column('next_run_at', sa.DateTime(timezone=True), nullable=True),

        # Resource limits
        sa.Column('max_tool_calls', sa.Integer, server_default='500'),
        sa.Column('max_llm_calls', sa.Integer, server_default='200'),
        sa.Column('max_runtime_minutes', sa.Integer, server_default='60'),

        # Usage tracking
        sa.Column('tool_calls_used', sa.Integer, server_default='0'),
        sa.Column('llm_calls_used', sa.Integer, server_default='0'),
        sa.Column('tokens_used', sa.Integer, server_default='0'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), nullable=True),

        # Celery task tracking
        sa.Column('celery_task_id', sa.String(100), nullable=True),
    )

    # Create indexes for agent_jobs
    op.create_index('ix_agent_jobs_user_id', 'agent_jobs', ['user_id'])
    op.create_index('ix_agent_jobs_status', 'agent_jobs', ['status'])
    op.create_index('ix_agent_jobs_job_type', 'agent_jobs', ['job_type'])
    op.create_index('ix_agent_jobs_created_at', 'agent_jobs', ['created_at'])
    op.create_index('ix_agent_jobs_next_run_at', 'agent_jobs', ['next_run_at'])

    # Create agent_job_checkpoints table
    op.create_table(
        'agent_job_checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Job reference
        sa.Column('job_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('agent_jobs.id', ondelete='CASCADE'), nullable=False),

        # Checkpoint data
        sa.Column('iteration', sa.Integer, nullable=False),
        sa.Column('phase', sa.String(100), nullable=True),
        sa.Column('state', postgresql.JSON, nullable=False),
        sa.Column('context', postgresql.JSON, nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Create index for checkpoints
    op.create_index('ix_agent_job_checkpoints_job_id', 'agent_job_checkpoints', ['job_id'])

    # Create agent_job_templates table
    op.create_table(
        'agent_job_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Template identification
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('category', sa.String(50), nullable=True),

        # Template configuration
        sa.Column('job_type', sa.String(50), nullable=False),
        sa.Column('default_goal', sa.Text, nullable=True),
        sa.Column('default_config', postgresql.JSON, nullable=True),

        # Agent to use
        sa.Column('agent_definition_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('agent_definitions.id', ondelete='SET NULL'), nullable=True),

        # Resource defaults
        sa.Column('default_max_iterations', sa.Integer, server_default='100'),
        sa.Column('default_max_tool_calls', sa.Integer, server_default='500'),
        sa.Column('default_max_llm_calls', sa.Integer, server_default='200'),
        sa.Column('default_max_runtime_minutes', sa.Integer, server_default='60'),

        # Visibility
        sa.Column('is_system', sa.Boolean, server_default='false'),
        sa.Column('is_active', sa.Boolean, server_default='true'),

        # Ownership
        sa.Column('owner_user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Insert default job templates
    op.execute("""
        INSERT INTO agent_job_templates (id, name, display_name, description, category, job_type, default_goal, default_config, is_system, default_max_iterations, default_max_runtime_minutes)
        VALUES
        (gen_random_uuid(), 'research_topic', 'Research Topic', 'Research a topic by finding and analyzing relevant papers and documents', 'research', 'research',
         'Research the given topic comprehensively. Find relevant papers, analyze key findings, identify trends, and synthesize insights.',
         '{"sources": ["arxiv", "documents"], "max_papers": 30, "synthesis_format": "report", "depth": "comprehensive"}',
         true, 50, 30),

        (gen_random_uuid(), 'literature_review', 'Literature Review', 'Conduct a systematic literature review on a research area', 'research', 'research',
         'Conduct a systematic literature review. Search for papers, categorize by methodology and findings, identify research gaps.',
         '{"sources": ["arxiv"], "max_papers": 50, "categorize": true, "identify_gaps": true, "synthesis_format": "structured_review"}',
         true, 100, 60),

        (gen_random_uuid(), 'knowledge_monitor', 'Knowledge Monitor', 'Monitor for new papers and updates on specified topics', 'monitoring', 'monitor',
         'Monitor for new papers and developments in the specified research areas. Alert on significant new findings.',
         '{"check_interval_hours": 24, "sources": ["arxiv"], "alert_threshold": 0.8}',
         true, 10, 15),

        (gen_random_uuid(), 'document_analysis', 'Document Analysis', 'Analyze a set of documents to extract insights and patterns', 'analysis', 'analysis',
         'Analyze the specified documents. Extract key themes, identify patterns, and generate a comprehensive analysis report.',
         '{"analysis_depth": "detailed", "extract_entities": true, "find_relationships": true}',
         true, 30, 20),

        (gen_random_uuid(), 'knowledge_expansion', 'Knowledge Expansion', 'Expand the knowledge base by finding and ingesting related content', 'knowledge', 'knowledge_expansion',
         'Expand the knowledge base around existing topics. Find related papers and documents, ingest them, and build connections.',
         '{"expansion_strategy": "related_topics", "max_new_documents": 20, "build_graph": true}',
         true, 50, 45)
    """)


def downgrade() -> None:
    op.drop_table('agent_job_templates')
    op.drop_table('agent_job_checkpoints')
    op.drop_table('agent_jobs')
