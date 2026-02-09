"""Add agent job chaining support.

Revision ID: 0035_add_agent_job_chaining
Revises: 0034_add_autonomous_agent_jobs
Create Date: 2025-01-28

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '0035_add_agent_job_chaining'
down_revision = '0034_add_autonomous_agent_jobs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add chaining columns to agent_jobs table
    op.add_column('agent_jobs', sa.Column(
        'parent_job_id',
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey('agent_jobs.id', ondelete='SET NULL'),
        nullable=True
    ))

    op.add_column('agent_jobs', sa.Column(
        'chain_config',
        postgresql.JSON,
        nullable=True,
        comment='Configuration for job chaining: trigger_condition, thresholds, inherit settings'
    ))

    op.add_column('agent_jobs', sa.Column(
        'chain_triggered',
        sa.Boolean,
        server_default='false',
        nullable=False,
        comment='Whether this job has triggered its chained children'
    ))

    op.add_column('agent_jobs', sa.Column(
        'chain_depth',
        sa.Integer,
        server_default='0',
        nullable=False,
        comment='Depth in chain hierarchy (0 = root job)'
    ))

    op.add_column('agent_jobs', sa.Column(
        'root_job_id',
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey('agent_jobs.id', ondelete='SET NULL'),
        nullable=True,
        comment='Reference to the original root job in a chain'
    ))

    # Create indexes for chain queries
    op.create_index('ix_agent_jobs_parent_job_id', 'agent_jobs', ['parent_job_id'])
    op.create_index('ix_agent_jobs_root_job_id', 'agent_jobs', ['root_job_id'])
    op.create_index('ix_agent_jobs_chain_depth', 'agent_jobs', ['chain_depth'])

    # Create agent_job_chain_definitions table for reusable chain templates
    op.create_table(
        'agent_job_chain_definitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),

        # Chain identification
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text, nullable=True),

        # Chain structure - ordered list of job templates/configs
        sa.Column('chain_steps', postgresql.JSON, nullable=False,
                  comment='Ordered list of chain steps with job configs and trigger conditions'),
        # Structure:
        # [
        #   {
        #     "step_name": "Initial Research",
        #     "template_id": "uuid" or null,
        #     "job_config": {...},  # Override config
        #     "trigger_condition": "on_complete",
        #     "trigger_thresholds": {...}
        #   },
        #   ...
        # ]

        # Default settings
        sa.Column('default_settings', postgresql.JSON, nullable=True,
                  comment='Default settings applied to all jobs in chain'),

        # Ownership
        sa.Column('owner_user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('is_system', sa.Boolean, server_default='false'),
        sa.Column('is_active', sa.Boolean, server_default='true'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Insert example chain definitions
    op.execute("""
        INSERT INTO agent_job_chain_definitions (id, name, display_name, description, chain_steps, is_system)
        VALUES
        (gen_random_uuid(), 'research_to_presentation', 'Research to Presentation',
         'Research a topic and automatically generate a presentation from findings',
         '[
           {
             "step_name": "Research Phase",
             "job_type": "research",
             "goal_template": "Research {topic} comprehensively",
             "config": {"sources": ["arxiv", "documents"], "max_papers": 30},
             "trigger_condition": "on_complete"
           },
           {
             "step_name": "Generate Presentation",
             "job_type": "synthesis",
             "goal_template": "Create a presentation summarizing research findings on {topic}",
             "config": {"output_format": "pptx", "slides": 15},
             "trigger_condition": "on_complete"
           }
         ]'::jsonb,
         true),

        (gen_random_uuid(), 'literature_review_pipeline', 'Literature Review Pipeline',
         'Comprehensive literature review with gap analysis and synthesis',
         '[
           {
             "step_name": "Paper Discovery",
             "job_type": "research",
             "goal_template": "Find all relevant papers on {topic}",
             "config": {"sources": ["arxiv"], "max_papers": 100, "depth": "broad"},
             "trigger_condition": "on_complete"
           },
           {
             "step_name": "Deep Analysis",
             "job_type": "analysis",
             "goal_template": "Analyze papers from discovery phase, categorize by methodology",
             "config": {"inherit_results": true, "categorize": true},
             "trigger_condition": "on_complete"
           },
           {
             "step_name": "Gap Analysis",
             "job_type": "analysis",
             "goal_template": "Identify research gaps and future directions",
             "config": {"inherit_results": true, "focus": "gaps"},
             "trigger_condition": "on_complete"
           },
           {
             "step_name": "Synthesis Report",
             "job_type": "synthesis",
             "goal_template": "Generate comprehensive literature review document",
             "config": {"output_format": "docx", "include_citations": true},
             "trigger_condition": "on_complete"
           }
         ]'::jsonb,
         true),

        (gen_random_uuid(), 'continuous_monitoring_with_alerts', 'Continuous Monitoring with Alerts',
         'Monitor topics and trigger analysis when significant papers are found',
         '[
           {
             "step_name": "Topic Monitoring",
             "job_type": "monitor",
             "goal_template": "Monitor for new papers on {topic}",
             "config": {"check_interval_hours": 12, "sources": ["arxiv"]},
             "trigger_condition": "on_findings",
             "trigger_thresholds": {"findings_threshold": 5}
           },
           {
             "step_name": "New Paper Analysis",
             "job_type": "analysis",
             "goal_template": "Analyze newly discovered papers for significance",
             "config": {"inherit_results": true, "quick_analysis": true},
             "trigger_condition": "on_complete"
           }
         ]'::jsonb,
         true)
    """)


def downgrade() -> None:
    # Drop chain definitions table
    op.drop_table('agent_job_chain_definitions')

    # Drop indexes
    op.drop_index('ix_agent_jobs_chain_depth', table_name='agent_jobs')
    op.drop_index('ix_agent_jobs_root_job_id', table_name='agent_jobs')
    op.drop_index('ix_agent_jobs_parent_job_id', table_name='agent_jobs')

    # Drop columns
    op.drop_column('agent_jobs', 'root_job_id')
    op.drop_column('agent_jobs', 'chain_depth')
    op.drop_column('agent_jobs', 'chain_triggered')
    op.drop_column('agent_jobs', 'chain_config')
    op.drop_column('agent_jobs', 'parent_job_id')
