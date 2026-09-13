"""Add agent_retractions: a result that was believed, then checked

The knowledge base is append-only, which is right for findings and wrong for
findings that turn out to be defective. This project retracted a whole
per-instruction measurement table -- four of nine classes had been timed on
chains that reached infinity within a few iterations -- and nothing in the
system noticed. Methods validated against those numbers kept their standing,
and a campaign running unattended would have gone on citing them.

Three subject kinds, because the real cases differ in scope: a whole run, a
*class* of finding across every run that produced it (which is what happened
here), or a single recorded method.

`reason` is NOT NULL on purpose. A later run has to be able to tell a
measurement withdrawn for a harness defect from one withdrawn because the
question changed, and only the reason distinguishes them.

`subject_ref` is text rather than a foreign key: the subject may be deleted,
and losing the record that it was retracted would be the wrong repair.

Revision ID: 0090_add_agent_retractions
Revises: 0089_reconcile_control_plane_upload_experiment
Create Date: 2026-08-24

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0090_add_agent_retractions"
down_revision = "0089_reconcile_control_plane_upload_experiment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_retractions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("subject_kind", sa.String(length=40), nullable=False),
        sa.Column("subject_ref", sa.String(length=300), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=200), nullable=True),
        sa.Column("source_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_agent_retractions_user_id", "agent_retractions", ["user_id"])
    op.create_index(
        "ix_agent_retractions_source_job_id", "agent_retractions", ["source_job_id"]
    )
    op.create_index(
        "ix_agent_retractions_user_kind",
        "agent_retractions",
        ["user_id", "subject_kind"],
    )
    op.create_index(
        "ix_agent_retractions_subject",
        "agent_retractions",
        ["subject_kind", "subject_ref"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_retractions_subject", table_name="agent_retractions")
    op.drop_index("ix_agent_retractions_user_kind", table_name="agent_retractions")
    op.drop_index("ix_agent_retractions_source_job_id", table_name="agent_retractions")
    op.drop_index("ix_agent_retractions_user_id", table_name="agent_retractions")
    op.drop_table("agent_retractions")
