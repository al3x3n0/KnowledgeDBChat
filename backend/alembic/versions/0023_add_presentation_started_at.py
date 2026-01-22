"""Add started_at timestamp to presentation_jobs.

Revision ID: 0023_add_presentation_started_at
Revises: 0022_merge_heads
Create Date: 2026-01-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "0023_add_presentation_started_at"
down_revision = "0022_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_columns = {col["name"] for col in inspector.get_columns("presentation_jobs")}

    if "started_at" not in existing_columns:
        op.add_column(
            "presentation_jobs",
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    existing_columns = {col["name"] for col in inspector.get_columns("presentation_jobs")}

    if "started_at" in existing_columns:
        op.drop_column("presentation_jobs", "started_at")

