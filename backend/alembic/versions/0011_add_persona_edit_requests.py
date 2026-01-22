"""Add persona edit request table.

Revision ID: 0011_add_persona_edit_requests
Revises: 0010_add_personas
Create Date: 2024-05-26 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0011_add_persona_edit_requests"
down_revision = "0010_add_personas"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "persona_edit_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("persona_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["persona_id"], ["personas.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["requested_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_persona_edit_requests_persona", "persona_edit_requests", ["persona_id"])
    op.create_index("ix_persona_edit_requests_requested_by", "persona_edit_requests", ["requested_by"])


def downgrade() -> None:
    op.drop_index("ix_persona_edit_requests_requested_by", table_name="persona_edit_requests")
    op.drop_index("ix_persona_edit_requests_persona", table_name="persona_edit_requests")
    op.drop_table("persona_edit_requests")
