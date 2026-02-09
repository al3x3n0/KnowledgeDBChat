"""Widen alembic_version.version_num to support long revision ids.

Revision ID: 0038a_widen_alembic_version_num
Revises: 0038
Create Date: 2026-02-05
"""

from alembic import op
import sqlalchemy as sa


# IMPORTANT: keep this revision id <= 32 chars because older deployments may
# still have alembic_version.version_num as VARCHAR(32).
revision = "0038a_widen_alembic_version_num"
down_revision = "0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Default Alembic uses VARCHAR(32) for version_num; some of our revision ids exceed that.
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=32),
        type_=sa.String(length=128),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=128),
        type_=sa.String(length=32),
        existing_nullable=False,
    )

