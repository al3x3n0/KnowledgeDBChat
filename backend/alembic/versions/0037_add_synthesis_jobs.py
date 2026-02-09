"""Add synthesis jobs table.

Revision ID: 0037
Revises: 0036_agent_job_memory
Create Date: 2024-01-28

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '0037'
down_revision = '0036_agent_job_memory'
branch_labels = None
depends_on = None


def upgrade():
    # Create synthesis_jobs table
    op.create_table(
        'synthesis_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_type', sa.String(50), nullable=False, server_default='multi_doc_summary'),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('document_ids', postgresql.JSON(), nullable=False, server_default='[]'),
        sa.Column('search_query', sa.Text(), nullable=True),
        sa.Column('topic', sa.String(500), nullable=True),
        sa.Column('options', postgresql.JSON(), nullable=True, server_default='{}'),
        sa.Column('output_format', sa.String(20), nullable=False, server_default='markdown'),
        sa.Column('output_style', sa.String(50), nullable=False, server_default='professional'),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_stage', sa.String(100), nullable=True),
        sa.Column('result_content', sa.Text(), nullable=True),
        sa.Column('result_metadata', postgresql.JSON(), nullable=True),
        sa.Column('artifacts', postgresql.JSON(), nullable=True, server_default='[]'),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_synthesis_jobs_user_id', 'synthesis_jobs', ['user_id'])
    op.create_index('ix_synthesis_jobs_status', 'synthesis_jobs', ['status'])
    op.create_index('ix_synthesis_jobs_job_type', 'synthesis_jobs', ['job_type'])
    op.create_index('ix_synthesis_jobs_created_at', 'synthesis_jobs', ['created_at'])


def downgrade():
    op.drop_index('ix_synthesis_jobs_created_at', table_name='synthesis_jobs')
    op.drop_index('ix_synthesis_jobs_job_type', table_name='synthesis_jobs')
    op.drop_index('ix_synthesis_jobs_status', table_name='synthesis_jobs')
    op.drop_index('ix_synthesis_jobs_user_id', table_name='synthesis_jobs')
    op.drop_table('synthesis_jobs')
