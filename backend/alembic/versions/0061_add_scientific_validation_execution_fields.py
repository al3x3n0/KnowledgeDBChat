"""Add sandbox and validation-run tracking fields for scientific research.

Revision ID: 0061_add_scientific_validation_execution_fields
Revises: 0060_add_domain_research_track_and_repo_inputs
Create Date: 2026-03-25 11:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0061_add_scientific_validation_execution_fields"
down_revision = "0060_add_domain_research_track_and_repo_inputs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "domain_research_profiles",
        sa.Column("sandbox_profile_id", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("latest_validation_run_ids", sa.JSON(), nullable=True),
    )
    op.add_column(
        "research_portfolios",
        sa.Column("sandbox_profile_id", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "research_portfolios",
        sa.Column("latest_validation_run_ids", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("research_portfolios", "latest_validation_run_ids")
    op.drop_column("research_portfolios", "sandbox_profile_id")
    op.drop_column("domain_research_profiles", "latest_validation_run_ids")
    op.drop_column("domain_research_profiles", "sandbox_profile_id")
