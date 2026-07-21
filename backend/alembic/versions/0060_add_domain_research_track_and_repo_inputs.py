"""Add track-aware scientific research fields to domain research profiles.

Revision ID: 0060_add_domain_research_track_and_repo_inputs
Revises: 0059_add_domain_research_memo_fields
Create Date: 2026-03-24 19:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0060_add_domain_research_track_and_repo_inputs"
down_revision = "0059_add_domain_research_memo_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "domain_research_profiles",
        sa.Column("track_type", sa.String(length=32), nullable=False, server_default="generic"),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("repo_source_ids", sa.JSON(), nullable=True),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("benchmark_queries", sa.JSON(), nullable=True),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("validation_policy", sa.JSON(), nullable=True),
    )
    op.alter_column("domain_research_profiles", "track_type", server_default=None)


def downgrade() -> None:
    op.drop_column("domain_research_profiles", "validation_policy")
    op.drop_column("domain_research_profiles", "benchmark_queries")
    op.drop_column("domain_research_profiles", "repo_source_ids")
    op.drop_column("domain_research_profiles", "track_type")
