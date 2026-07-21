"""add research portfolio automation profile

Revision ID: 0068_add_research_portfolio_automation_profile
Revises: 0067_add_benchmark_harness_tables
Create Date: 2026-04-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0068_add_research_portfolio_automation_profile"
down_revision = "0067_add_benchmark_harness_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_portfolios",
        sa.Column("automation_profile", sa.String(length=24), nullable=False, server_default="balanced"),
    )
    op.execute("UPDATE research_portfolios SET automation_profile = 'balanced' WHERE automation_profile IS NULL")
    op.alter_column("research_portfolios", "automation_profile", server_default=None)


def downgrade() -> None:
    op.drop_column("research_portfolios", "automation_profile")
