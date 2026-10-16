"""A campaign may approve a plan its own job proposed, within limits.

A campaign stopped for a person at every plan, so one left to run alone
stalled at its first proposal. ``plan_approval`` holds the limits under which
it may approve one itself; NULL keeps the old behaviour.

Revision ID: 0110_campaign_plan_approval
Revises: 0109_api_key_prefix_width
"""

import sqlalchemy as sa

from alembic import op

revision = "0110_campaign_plan_approval"
down_revision = "0109_api_key_prefix_width"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_campaigns", sa.Column("plan_approval", sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("research_campaigns", "plan_approval")
