"""Add domain research policy fields and structured memo payloads.

Revision ID: 0059_add_domain_research_memo_fields
Revises: 0058_add_research_portfolios
Create Date: 2026-03-24 12:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0059_add_domain_research_memo_fields"
down_revision = "0058_add_research_portfolios"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "domain_research_profiles",
        sa.Column("research_mode", sa.String(length=48), nullable=False, server_default="literature_to_hypothesis"),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("scoring_policy", sa.JSON(), nullable=True),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("selection_policy", sa.JSON(), nullable=True),
    )
    op.add_column(
        "research_notes",
        sa.Column("structured_payload", sa.JSON(), nullable=True),
    )
    op.alter_column("domain_research_profiles", "research_mode", server_default=None)


def downgrade() -> None:
    op.drop_column("research_notes", "structured_payload")
    op.drop_column("domain_research_profiles", "selection_policy")
    op.drop_column("domain_research_profiles", "scoring_policy")
    op.drop_column("domain_research_profiles", "research_mode")
