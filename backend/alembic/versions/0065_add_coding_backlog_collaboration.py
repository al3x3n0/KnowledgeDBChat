"""Add coding backlog collaboration fields.

Revision ID: 0065_add_coding_backlog_collaboration
Revises: 0064_add_coding_swarm_profile_sharing
Create Date: 2026-03-26 23:10:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0065_add_coding_backlog_collaboration"
down_revision = "0064_add_coding_swarm_profile_sharing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "coding_backlog_items",
        sa.Column("visibility", sa.String(length=24), nullable=False, server_default="private"),
    )
    op.add_column(
        "coding_backlog_items",
        sa.Column("shared_with_user_ids", sa.JSON(), nullable=True),
    )
    op.add_column(
        "coding_backlog_items",
        sa.Column("assigned_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "coding_backlog_items",
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "coding_backlog_items",
        sa.Column("assigned_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "coding_backlog_items",
        sa.Column("collaboration", sa.JSON(), nullable=True),
    )
    op.create_index("ix_coding_backlog_items_visibility", "coding_backlog_items", ["visibility"], unique=False)
    op.create_index("ix_coding_backlog_items_assigned_user_id", "coding_backlog_items", ["assigned_user_id"], unique=False)
    op.create_index("ix_coding_backlog_items_assigned_by_user_id", "coding_backlog_items", ["assigned_by_user_id"], unique=False)
    op.create_foreign_key(
        "fk_coding_backlog_items_assigned_user_id_users",
        "coding_backlog_items",
        "users",
        ["assigned_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_coding_backlog_items_assigned_by_user_id_users",
        "coding_backlog_items",
        "users",
        ["assigned_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_coding_backlog_items_assigned_by_user_id_users", "coding_backlog_items", type_="foreignkey")
    op.drop_constraint("fk_coding_backlog_items_assigned_user_id_users", "coding_backlog_items", type_="foreignkey")
    op.drop_index("ix_coding_backlog_items_assigned_by_user_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_assigned_user_id", table_name="coding_backlog_items")
    op.drop_index("ix_coding_backlog_items_visibility", table_name="coding_backlog_items")
    op.drop_column("coding_backlog_items", "collaboration")
    op.drop_column("coding_backlog_items", "assigned_by_user_id")
    op.drop_column("coding_backlog_items", "assigned_at")
    op.drop_column("coding_backlog_items", "assigned_user_id")
    op.drop_column("coding_backlog_items", "shared_with_user_ids")
    op.drop_column("coding_backlog_items", "visibility")
