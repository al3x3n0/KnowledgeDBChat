"""Merge heads 0018_add_presentation_templates and 0021_add_notifications

Revision ID: 0022_merge_heads
Revises: 0018_add_presentation_templates, 0021_add_notifications
Create Date: 2026-01-22
"""

from alembic import op  # noqa: F401

# revision identifiers, used by Alembic.
revision = "0022_merge_heads"
down_revision = ("0018_add_presentation_templates", "0021_add_notifications")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

