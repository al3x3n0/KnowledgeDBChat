"""add kg audit log table

Revision ID: 0004_add_kg_audit_log
Revises: 0003_add_knowledge_graph
Create Date: 2025-11-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = '0004_add_kg_audit_log'
down_revision = '0003_add_knowledge_graph'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'kg_audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('action', sa.String(length=64), nullable=False),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_kg_audit_action', 'kg_audit_logs', ['action'])
    op.create_index('ix_kg_audit_created', 'kg_audit_logs', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_kg_audit_created', table_name='kg_audit_logs')
    op.drop_index('ix_kg_audit_action', table_name='kg_audit_logs')
    op.drop_table('kg_audit_logs')

