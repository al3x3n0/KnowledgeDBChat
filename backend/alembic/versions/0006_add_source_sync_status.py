"""
Add is_syncing and last_error to document_sources

Revision ID: 0006
Revises: 0005
Create Date: 2024-12-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0006_add_source_sync_status'
down_revision = '0005_add_document_summary'
branch_labels = None
depends_on = None


def upgrade() -> None:
    try:
        op.add_column('document_sources', sa.Column('is_syncing', sa.Boolean(), nullable=False, server_default=sa.text('false')))
    except Exception:
        pass
    try:
        op.add_column('document_sources', sa.Column('last_error', sa.Text(), nullable=True))
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_column('document_sources', 'last_error')
    except Exception:
        pass
    try:
        op.drop_column('document_sources', 'is_syncing')
    except Exception:
        pass
