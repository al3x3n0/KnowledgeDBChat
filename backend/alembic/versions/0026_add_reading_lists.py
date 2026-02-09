"""Add reading lists tables.

Revision ID: 0026_add_reading_lists
Revises: 0025_enable_lit_review
Create Date: 2026-01-26 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "0026_add_reading_lists"
down_revision = "0025_enable_lit_review"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)

    if not insp.has_table("reading_lists"):
        op.create_table(
            "reading_lists",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["source_id"], ["document_sources.id"], ondelete="SET NULL"),
        )

    # Ensure indexes/constraints exist (idempotent on Postgres).
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_list_user_name ON reading_lists(user_id, name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_reading_lists_user_id ON reading_lists(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_reading_lists_source_id ON reading_lists(source_id)")

    if not insp.has_table("reading_list_items"):
        op.create_table(
            "reading_list_items",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("reading_list_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'to-read'")),
            sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["reading_list_id"], ["reading_lists.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        )

    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_reading_list_item_document_once ON reading_list_items(reading_list_id, document_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_reading_list_items_reading_list_id ON reading_list_items(reading_list_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_reading_list_items_document_id ON reading_list_items(document_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_reading_list_items_list_position ON reading_list_items(reading_list_id, position)"
    )


def downgrade() -> None:
    op.drop_index("ix_reading_list_items_list_position", table_name="reading_list_items")
    op.drop_index("ix_reading_list_items_document_id", table_name="reading_list_items")
    op.drop_index("ix_reading_list_items_reading_list_id", table_name="reading_list_items")
    op.drop_table("reading_list_items")

    op.drop_index("ix_reading_lists_source_id", table_name="reading_lists")
    op.drop_index("ix_reading_lists_user_id", table_name="reading_lists")
    op.drop_table("reading_lists")
