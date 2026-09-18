"""Why an inbox item was rejected.

Nullable with no backfill on purpose: rows triaged before this column existed
were learned from under the old rules, where every rejection taught the topic
counters. ``research_rejection_reasons.teaches_topic`` reads NULL as exactly
that, so history keeps meaning what it meant.

Revision ID: 0102_inbox_rejection_reason
Revises: 0101_campaign_conclusion_detail
"""

import sqlalchemy as sa
from alembic import op

revision = "0102_inbox_rejection_reason"
down_revision = "0101_campaign_conclusion_detail"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_inbox_items",
        sa.Column("rejection_reason", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("research_inbox_items", "rejection_reason")
