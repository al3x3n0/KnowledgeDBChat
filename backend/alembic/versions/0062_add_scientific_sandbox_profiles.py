"""Add persisted scientific sandbox profiles.

Revision ID: 0062_add_scientific_sandbox_profiles
Revises: 0061_add_scientific_validation_execution_fields
Create Date: 2026-03-25 13:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0062_add_scientific_sandbox_profiles"
down_revision = "0061_add_scientific_validation_execution_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scientific_sandbox_profiles",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("track_type", sa.String(length=32), nullable=False),
        sa.Column("backend", sa.String(length=24), nullable=False),
        sa.Column("docker_image", sa.String(length=255), nullable=True),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("resource_caps", sa.JSON(), nullable=False),
        sa.Column("allowed_benchmark_families", sa.JSON(), nullable=False),
        sa.Column("allowed_perf_collectors", sa.JSON(), nullable=False),
        sa.Column("required_capabilities", sa.JSON(), nullable=False),
        sa.Column("toolchains", sa.JSON(), nullable=False),
        sa.Column("budget_limit_default", sa.Float(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("system_managed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_by_user_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_scientific_sandbox_profiles_enabled", "scientific_sandbox_profiles", ["enabled"], unique=False)
    op.create_index("ix_scientific_sandbox_profiles_system_managed", "scientific_sandbox_profiles", ["system_managed"], unique=False)
    op.create_index("ix_scientific_sandbox_profiles_track_type", "scientific_sandbox_profiles", ["track_type"], unique=False)

    profiles = sa.table(
        "scientific_sandbox_profiles",
        sa.column("id", sa.String(length=80)),
        sa.column("name", sa.String(length=200)),
        sa.column("description", sa.Text()),
        sa.column("track_type", sa.String(length=32)),
        sa.column("backend", sa.String(length=24)),
        sa.column("docker_image", sa.String(length=255)),
        sa.column("timeout_seconds", sa.Integer()),
        sa.column("resource_caps", sa.JSON()),
        sa.column("allowed_benchmark_families", sa.JSON()),
        sa.column("allowed_perf_collectors", sa.JSON()),
        sa.column("required_capabilities", sa.JSON()),
        sa.column("toolchains", sa.JSON()),
        sa.column("budget_limit_default", sa.Float()),
        sa.column("enabled", sa.Boolean()),
        sa.column("system_managed", sa.Boolean()),
        sa.column("is_default", sa.Boolean()),
    )
    op.bulk_insert(
        profiles,
        [
            {
                "id": "scientific-compiler-sandbox",
                "name": "Compiler Validation Sandbox",
                "description": "Docker-isolated compiler research sandbox for compile/codegen/regression validation.",
                "track_type": "compiler",
                "backend": "docker",
                "docker_image": "ghcr.io/knowledgedb/compiler-research:latest",
                "timeout_seconds": 1200,
                "resource_caps": {"memory_mb": 4096, "cpus": 2.0, "pids_limit": 256},
                "allowed_benchmark_families": ["compiler_regression", "codegen_quality", "kernel_compile"],
                "allowed_perf_collectors": ["benchmark_output", "compile_time", "artifact_diff"],
                "required_capabilities": ["repo_reconstruction"],
                "toolchains": ["clang", "llvm-opt", "cmake", "ninja", "pytest"],
                "budget_limit_default": 35.0,
                "enabled": True,
                "system_managed": True,
                "is_default": True,
            },
            {
                "id": "scientific-microarchitecture-sandbox",
                "name": "Microarchitecture Validation Sandbox",
                "description": "Docker-isolated sandbox for perf-counter and benchmark-based microarchitecture validation.",
                "track_type": "microarchitecture",
                "backend": "docker",
                "docker_image": "ghcr.io/knowledgedb/microarch-research:latest",
                "timeout_seconds": 1200,
                "resource_caps": {"memory_mb": 4096, "cpus": 2.0, "pids_limit": 256},
                "allowed_benchmark_families": ["perf_counter_regression", "cache_branch_analysis", "throughput_latency"],
                "allowed_perf_collectors": ["perf_stat", "cache_miss", "branch_miss", "benchmark_output"],
                "required_capabilities": ["repo_reconstruction", "perf_counters"],
                "toolchains": ["python", "pytest", "perf"],
                "budget_limit_default": 40.0,
                "enabled": True,
                "system_managed": True,
                "is_default": True,
            },
            {
                "id": "scientific-generic-sandbox",
                "name": "Scientific Validation Sandbox",
                "description": "Default docker-isolated sandbox for bounded technical validation runs.",
                "track_type": "generic",
                "backend": "docker",
                "docker_image": "python:3.11-slim",
                "timeout_seconds": 900,
                "resource_caps": {"memory_mb": 2048, "cpus": 1.5, "pids_limit": 192},
                "allowed_benchmark_families": ["generic_validation"],
                "allowed_perf_collectors": ["benchmark_output"],
                "required_capabilities": ["repo_reconstruction"],
                "toolchains": ["python", "pytest"],
                "budget_limit_default": 25.0,
                "enabled": True,
                "system_managed": True,
                "is_default": True,
            },
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_scientific_sandbox_profiles_track_type", table_name="scientific_sandbox_profiles")
    op.drop_index("ix_scientific_sandbox_profiles_system_managed", table_name="scientific_sandbox_profiles")
    op.drop_index("ix_scientific_sandbox_profiles_enabled", table_name="scientific_sandbox_profiles")
    op.drop_table("scientific_sandbox_profiles")
