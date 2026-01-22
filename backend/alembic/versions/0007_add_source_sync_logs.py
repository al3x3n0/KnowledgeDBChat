"""
Add document_source_sync_logs table

Revision ID: 0007_add_source_sync_logs
Revises: 0006_add_source_sync_status
Create Date: 2025-12-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = '0007_add_source_sync_logs'
down_revision = '0006_add_source_sync_status'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'document_source_sync_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='running'),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_documents', sa.Integer(), nullable=True),
        sa.Column('processed', sa.Integer(), nullable=True),
        sa.Column('created', sa.Integer(), nullable=True),
        sa.Column('updated', sa.Integer(), nullable=True),
        sa.Column('errors', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['source_id'], ['document_sources.id'], ),
    )
    op.create_index('ix_sync_logs_source_started', 'document_source_sync_logs', ['source_id', 'started_at'])


def downgrade() -> None:
    op.drop_index('ix_sync_logs_source_started', table_name='document_source_sync_logs')
    op.drop_table('document_source_sync_logs')

