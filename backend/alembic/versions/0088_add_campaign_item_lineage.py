"""Give campaign items a lineage, so a cold line can be told from a cold item

A campaign worked its backlog in creation order because it had nothing to
judge on. The missing fact was lineage: a discovered item recorded no pointer
to the item whose job revealed it, so the tenth offshoot of a line that had
produced nothing was indistinguishable from the first candidate out of a job
that met its contract.

parent_item_id and generation make the walk up the ancestry possible, which is
what abandoning a cold line requires -- one bad job is a bad job, and only a
run of them is a line worth stopping. launched_at gives launch order, which
creation order does not: an item spawned early may run late, and the guard
against a campaign chasing its own tail needs to know what actually ran
recently. priority and priority_reason record what the campaign thought and
why, so a choice can be read afterwards rather than only observed.

Existing rows default to generation 0 with no parent, which reads them all as
seed-level work. That is right for them: they were created before anything
spawned anything.

Revision ID: 0088_add_campaign_item_lineage
Revises: 0087_add_research_campaigns
Create Date: 2026-08-22

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0088_add_campaign_item_lineage"
down_revision = "0087_add_research_campaigns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_campaign_items",
        sa.Column("parent_item_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "research_campaign_items",
        sa.Column("generation", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "research_campaign_items",
        sa.Column("launched_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "research_campaign_items",
        sa.Column("priority", sa.Float(), nullable=True),
    )
    op.add_column(
        "research_campaign_items",
        sa.Column("priority_reason", sa.String(length=400), nullable=True),
    )
    op.create_index(
        "ix_research_campaign_items_parent_item_id",
        "research_campaign_items",
        ["parent_item_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_campaign_items_parent_item_id",
        table_name="research_campaign_items",
    )
    op.drop_column("research_campaign_items", "priority_reason")
    op.drop_column("research_campaign_items", "priority")
    op.drop_column("research_campaign_items", "launched_at")
    op.drop_column("research_campaign_items", "generation")
    op.drop_column("research_campaign_items", "parent_item_id")
