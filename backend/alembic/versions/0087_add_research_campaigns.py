"""Add research_campaigns: a line of enquiry that outlives one job

Every agent goal so far has been scoped to a job, which is the right unit for
one experiment and the wrong one for a question worth a week. Job chaining
fires children at completion and then nobody is watching, and a restart ends
the sequence silently. These two tables hold the sequence instead: a standing
goal with a backlog under it, advanced a step at a time by a caller that can be
a scheduler, so all the state that matters survives the process.

The job budget is a column rather than a convention because an agent that
creates work from its own findings can create it without end.

Revision ID: 0087_add_research_campaigns
Revises: 0086_add_agent_method_outcomes
Create Date: 2026-08-22

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0087_add_research_campaigns"
down_revision = "0086_add_agent_method_outcomes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_campaigns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=300), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(length=40), nullable=False, server_default="active"
        ),
        sa.Column("max_jobs", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("jobs_launched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("job_template", sa.JSON(), nullable=True),
        sa.Column("conclusion", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_research_campaigns_user_id", "research_campaigns", ["user_id"])
    op.create_index("ix_research_campaigns_status", "research_campaigns", ["status"])
    op.create_index(
        "ix_research_campaigns_status_user", "research_campaigns", ["status", "user_id"]
    )

    op.create_table(
        "research_campaign_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "campaign_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("research_campaigns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.String(length=40), nullable=False, server_default="pending"
        ),
        sa.Column(
            "origin", sa.String(length=40), nullable=False, server_default="seed"
        ),
        # Not a foreign key: the job may be deleted, and losing the record that
        # this item was worked on would be the wrong repair.
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("outcome", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_research_campaign_items_campaign_id",
        "research_campaign_items",
        ["campaign_id"],
    )
    op.create_index(
        "ix_research_campaign_items_status", "research_campaign_items", ["status"]
    )
    op.create_index(
        "ix_research_campaign_items_job_id", "research_campaign_items", ["job_id"]
    )
    op.create_index(
        "ix_research_campaign_items_campaign_status",
        "research_campaign_items",
        ["campaign_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_campaign_items_campaign_status",
        table_name="research_campaign_items",
    )
    op.drop_index(
        "ix_research_campaign_items_job_id", table_name="research_campaign_items"
    )
    op.drop_index(
        "ix_research_campaign_items_status", table_name="research_campaign_items"
    )
    op.drop_index(
        "ix_research_campaign_items_campaign_id", table_name="research_campaign_items"
    )
    op.drop_table("research_campaign_items")

    op.drop_index("ix_research_campaigns_status_user", table_name="research_campaigns")
    op.drop_index("ix_research_campaigns_status", table_name="research_campaigns")
    op.drop_index("ix_research_campaigns_user_id", table_name="research_campaigns")
    op.drop_table("research_campaigns")
