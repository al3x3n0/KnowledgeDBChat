"""Add agent job memory integration.

Extends the memory system to support autonomous agent jobs:
- Add job_id to conversation_memories for job-sourced memories
- Add memory tracking fields to agent_jobs
- Support new memory types: finding, insight, pattern, lesson

Revision ID: 0036_agent_job_memory
Revises: 0035_add_agent_job_chaining
Create Date: 2025-01-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '0036_agent_job_memory'
down_revision = '0035_add_agent_job_chaining'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add job_id to conversation_memories for job-sourced memories
    op.add_column(
        'conversation_memories',
        sa.Column(
            'job_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('agent_jobs.id', ondelete='SET NULL'),
            nullable=True
        )
    )

    # Add index for job_id
    op.create_index(
        'ix_conversation_memories_job_id',
        'conversation_memories',
        ['job_id']
    )

    # Add memory tracking fields to agent_jobs
    op.add_column(
        'agent_jobs',
        sa.Column('enable_memory', sa.Boolean(), nullable=True, default=True)
    )
    op.add_column(
        'agent_jobs',
        sa.Column('memory_injection_count', sa.Integer(), nullable=True, default=0)
    )
    op.add_column(
        'agent_jobs',
        sa.Column('memories_created_count', sa.Integer(), nullable=True, default=0)
    )

    # Set defaults for existing rows
    op.execute("UPDATE agent_jobs SET enable_memory = true WHERE enable_memory IS NULL")
    op.execute("UPDATE agent_jobs SET memory_injection_count = 0 WHERE memory_injection_count IS NULL")
    op.execute("UPDATE agent_jobs SET memories_created_count = 0 WHERE memories_created_count IS NULL")

    # Add agent job memory settings to user_preferences
    op.add_column(
        'user_preferences',
        sa.Column('agent_job_memory_types', sa.JSON(), nullable=True)
    )
    op.add_column(
        'user_preferences',
        sa.Column('max_job_memories', sa.Integer(), nullable=True, default=10)
    )
    op.add_column(
        'user_preferences',
        sa.Column('auto_extract_job_memories', sa.Boolean(), nullable=True, default=True)
    )
    op.add_column(
        'user_preferences',
        sa.Column('share_memories_with_chat', sa.Boolean(), nullable=True, default=True)
    )

    # Set defaults for user_preferences
    op.execute("""
        UPDATE user_preferences
        SET agent_job_memory_types = '["finding", "insight", "pattern", "lesson"]'::json
        WHERE agent_job_memory_types IS NULL
    """)
    op.execute("UPDATE user_preferences SET max_job_memories = 10 WHERE max_job_memories IS NULL")
    op.execute("UPDATE user_preferences SET auto_extract_job_memories = true WHERE auto_extract_job_memories IS NULL")
    op.execute("UPDATE user_preferences SET share_memories_with_chat = true WHERE share_memories_with_chat IS NULL")


def downgrade() -> None:
    # Remove columns from user_preferences
    op.drop_column('user_preferences', 'share_memories_with_chat')
    op.drop_column('user_preferences', 'auto_extract_job_memories')
    op.drop_column('user_preferences', 'max_job_memories')
    op.drop_column('user_preferences', 'agent_job_memory_types')

    # Remove memory tracking columns from agent_jobs
    op.drop_column('agent_jobs', 'memories_created_count')
    op.drop_column('agent_jobs', 'memory_injection_count')
    op.drop_column('agent_jobs', 'enable_memory')

    # Remove index and column from conversation_memories
    op.drop_index('ix_conversation_memories_job_id', table_name='conversation_memories')
    op.drop_column('conversation_memories', 'job_id')
