"""Add coding swarm profile sharing fields.

Revision ID: 0064_add_coding_swarm_profile_sharing
Revises: 0063_add_coding_swarm_profiles_and_backlog_lineage
Create Date: 2026-03-26 20:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0064_add_coding_swarm_profile_sharing"
down_revision = "0063_add_coding_swarm_profiles_and_backlog_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "coding_swarm_profiles",
        sa.Column("visibility", sa.String(length=24), nullable=False, server_default="private"),
    )
    op.add_column(
        "coding_swarm_profiles",
        sa.Column("shared_with_user_ids", sa.JSON(), nullable=True),
    )
    op.create_index("ix_coding_swarm_profiles_visibility", "coding_swarm_profiles", ["visibility"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_coding_swarm_profiles_visibility", table_name="coding_swarm_profiles")
    op.drop_column("coding_swarm_profiles", "shared_with_user_ids")
    op.drop_column("coding_swarm_profiles", "visibility")
