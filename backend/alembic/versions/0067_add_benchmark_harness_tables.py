"""add benchmark harness tables

Revision ID: 0067_add_benchmark_harness_tables
Revises: 0066_add_synthesis_job_research_note_id
Create Date: 2026-03-27 12:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0067_add_benchmark_harness_tables"
down_revision = "0066_add_synthesis_job_research_note_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_suites",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("track_type", sa.String(length=32), nullable=False),
        sa.Column("benchmark_family", sa.String(length=64), nullable=False),
        sa.Column("suite_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("system_managed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_suites_user_id", "benchmark_suites", ["user_id"], unique=False)
    op.create_index("ix_benchmark_suites_track_type", "benchmark_suites", ["track_type"], unique=False)
    op.create_index("ix_benchmark_suites_benchmark_family", "benchmark_suites", ["benchmark_family"], unique=False)
    op.create_index("ix_benchmark_suites_enabled", "benchmark_suites", ["enabled"], unique=False)
    op.create_index("ix_benchmark_suites_system_managed", "benchmark_suites", ["system_managed"], unique=False)

    op.create_table(
        "benchmark_cases",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("suite_id", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_ref", sa.String(length=500), nullable=True),
        sa.Column("benchmark_query", sa.String(length=500), nullable=True),
        sa.Column("compile_command_template", sa.Text(), nullable=True),
        sa.Column("run_command_template", sa.Text(), nullable=True),
        sa.Column("expected_artifacts", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["suite_id"], ["benchmark_suites.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_cases_suite_id", "benchmark_cases", ["suite_id"], unique=False)

    op.create_table(
        "benchmark_baselines",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("suite_id", sa.String(length=120), nullable=False),
        sa.Column("case_id", sa.String(length=120), nullable=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("compiler_revision", sa.String(length=120), nullable=True),
        sa.Column("toolchain_id", sa.String(length=120), nullable=True),
        sa.Column("sandbox_profile_id", sa.String(length=120), nullable=True),
        sa.Column("measurements", sa.JSON(), nullable=False),
        sa.Column("environment_snapshot", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("system_managed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["suite_id"], ["benchmark_suites.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["case_id"], ["benchmark_cases.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_baselines_suite_id", "benchmark_baselines", ["suite_id"], unique=False)
    op.create_index("ix_benchmark_baselines_case_id", "benchmark_baselines", ["case_id"], unique=False)
    op.create_index("ix_benchmark_baselines_enabled", "benchmark_baselines", ["enabled"], unique=False)
    op.create_index("ix_benchmark_baselines_system_managed", "benchmark_baselines", ["system_managed"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_benchmark_baselines_system_managed", table_name="benchmark_baselines")
    op.drop_index("ix_benchmark_baselines_enabled", table_name="benchmark_baselines")
    op.drop_index("ix_benchmark_baselines_case_id", table_name="benchmark_baselines")
    op.drop_index("ix_benchmark_baselines_suite_id", table_name="benchmark_baselines")
    op.drop_table("benchmark_baselines")

    op.drop_index("ix_benchmark_cases_suite_id", table_name="benchmark_cases")
    op.drop_table("benchmark_cases")

    op.drop_index("ix_benchmark_suites_system_managed", table_name="benchmark_suites")
    op.drop_index("ix_benchmark_suites_enabled", table_name="benchmark_suites")
    op.drop_index("ix_benchmark_suites_benchmark_family", table_name="benchmark_suites")
    op.drop_index("ix_benchmark_suites_track_type", table_name="benchmark_suites")
    op.drop_index("ix_benchmark_suites_user_id", table_name="benchmark_suites")
    op.drop_table("benchmark_suites")
