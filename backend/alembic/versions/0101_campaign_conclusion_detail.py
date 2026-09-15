"""Let a campaign say what it concluded, and show its working.

`research_campaigns.conclusion` has existed since campaigns did: declared on
the model, returned by the API, and written by nothing. Every campaign that
ever completed did so silently -- four of them, in this deployment, all
`completed`, none with an answer. A campaign exists to settle a goal, and one
that runs jobs and ends without saying what the answer turned out to be is a
batch runner with extra steps.

The text column now gets written. This adds the structure beside it: the
evidence the answer rests on, the gaps it could not close, its confidence, and
how many findings were weighed. Beside rather than instead -- the line is what
a list shows, and an answer with no way to see what it rested on is one nobody
should act on.

Revision ID: 0101_campaign_conclusion_detail
Revises: 0100_workflow_plugin_origin
"""

import sqlalchemy as sa

from alembic import op

revision = "0101_campaign_conclusion_detail"
down_revision = "0100_workflow_plugin_origin"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_campaigns", sa.Column("conclusion_detail", sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("research_campaigns", "conclusion_detail")
