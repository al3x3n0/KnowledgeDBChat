"""add domain research profile automation fields

Revision ID: 0069_add_domain_research_profile_automation_fields
Revises: 0068_add_research_portfolio_automation_profile
Create Date: 2026-04-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "0069_add_domain_research_profile_automation_fields"
down_revision = "0068_add_research_portfolio_automation_profile"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "domain_research_profiles",
        sa.Column("automation_profile", sa.String(length=24), nullable=False, server_default="balanced"),
    )
    op.add_column(
        "domain_research_profiles",
        sa.Column("automation_policy", sa.JSON(), nullable=True),
    )
    op.alter_column("domain_research_profiles", "automation_profile", server_default=None)


def downgrade() -> None:
    op.drop_column("domain_research_profiles", "automation_policy")
    op.drop_column("domain_research_profiles", "automation_profile")
