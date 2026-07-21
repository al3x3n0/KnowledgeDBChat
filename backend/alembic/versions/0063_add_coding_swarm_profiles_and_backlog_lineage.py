"""Add coding swarm profiles and backlog lineage.

Revision ID: 0063_add_coding_swarm_profiles_and_backlog_lineage
Revises: 0062_add_scientific_sandbox_profiles
Create Date: 2026-03-26 18:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0063_add_coding_swarm_profiles_and_backlog_lineage"
down_revision = "0062_add_scientific_sandbox_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "coding_swarm_profiles",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("source_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("preset_key", sa.String(length=48), nullable=False),
        sa.Column("scope_default", sa.String(length=32), nullable=False),
        sa.Column("default_commands", sa.JSON(), nullable=True),
        sa.Column("default_file_paths", sa.JSON(), nullable=True),
        sa.Column("max_agents", sa.Integer(), nullable=False),
        sa.Column("safe_command_policy", sa.String(length=32), nullable=False),
        sa.Column("saved_search_query", sa.String(length=500), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latest_job_id", sa.UUID(), nullable=True),
        sa.Column("profile_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["latest_job_id"], ["agent_jobs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_id"], ["document_sources.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_coding_swarm_profiles_user_id", "coding_swarm_profiles", ["user_id"], unique=False)
    op.create_index("ix_coding_swarm_profiles_source_id", "coding_swarm_profiles", ["source_id"], unique=False)
    op.create_index("ix_coding_swarm_profiles_preset_key", "coding_swarm_profiles", ["preset_key"], unique=False)
    op.create_index("ix_coding_swarm_profiles_status", "coding_swarm_profiles", ["status"], unique=False)

    op.add_column("coding_backlog_items", sa.Column("lineage", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("coding_backlog_items", "lineage")
    op.drop_index("ix_coding_swarm_profiles_status", table_name="coding_swarm_profiles")
    op.drop_index("ix_coding_swarm_profiles_preset_key", table_name="coding_swarm_profiles")
    op.drop_index("ix_coding_swarm_profiles_source_id", table_name="coding_swarm_profiles")
    op.drop_index("ix_coding_swarm_profiles_user_id", table_name="coding_swarm_profiles")
    op.drop_table("coding_swarm_profiles")
