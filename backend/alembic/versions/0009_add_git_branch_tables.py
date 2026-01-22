"""Add git branch tables for comparisons.

Revision ID: 0009_add_git_branch_tables
Revises: 0008_repair_schema
Create Date: 2024-05-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
import uuid


# revision identifiers, used by Alembic.
revision = '0009_add_git_branch_tables'
down_revision = '0008_repair_schema'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'git_branches',
        sa.Column('id', sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('source_id', sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey('document_sources.id'), nullable=False),
        sa.Column('repository', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('head_sha', sa.String(length=100), nullable=True),
        sa.Column('head_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('branch_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_git_branches_source_repo', 'git_branches', ['source_id', 'repository'])
    op.create_table(
        'git_branch_diffs',
        sa.Column('id', sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('source_id', sa.dialects.postgresql.UUID(as_uuid=True), sa.ForeignKey('document_sources.id'), nullable=False),
        sa.Column('repository', sa.String(length=255), nullable=False),
        sa.Column('base_branch', sa.String(length=120), nullable=False),
        sa.Column('compare_branch', sa.String(length=120), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='queued'),
        sa.Column('task_id', sa.String(length=120), nullable=True),
        sa.Column('diff_summary', sa.JSON(), nullable=True),
        sa.Column('llm_summary', sa.Text(), nullable=True),
        sa.Column('options', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_git_branch_diffs_source', 'git_branch_diffs', ['source_id'])


def downgrade():
    op.drop_index('ix_git_branch_diffs_source', table_name='git_branch_diffs')
    op.drop_table('git_branch_diffs')
    op.drop_index('ix_git_branches_source_repo', table_name='git_branches')
    op.drop_table('git_branches')
