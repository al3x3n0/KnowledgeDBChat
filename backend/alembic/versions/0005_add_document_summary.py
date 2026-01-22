"""add document summary fields

Revision ID: 0005_add_document_summary
Revises: 0004_add_kg_audit_log
Create Date: 2025-11-28
"""

from alembic import op
import sqlalchemy as sa


revision = '0005_add_document_summary'
down_revision = '0004_add_kg_audit_log'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('documents', sa.Column('summary', sa.Text(), nullable=True))
    op.add_column('documents', sa.Column('summary_model', sa.String(length=100), nullable=True))
    op.add_column('documents', sa.Column('summary_generated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('documents', 'summary_generated_at')
    op.drop_column('documents', 'summary_model')
    op.drop_column('documents', 'summary')

