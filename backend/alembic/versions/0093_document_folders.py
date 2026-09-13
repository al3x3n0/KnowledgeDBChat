"""Document folders: a per-user tree over shared documents.

Two tables and nothing else. System folders (by source, by type, recent,
unfiled) are computed at read time from what documents already carry, so this
migration neither creates them nor touches a single existing document row —
there is nothing here to undo if the feature is dropped.

Revision ID: 0093_document_folders
Revises: 0092_synthesis_from_agent_runs
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0093_document_folders"
down_revision = "0092_synthesis_from_agent_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_folders",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("color", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # Self-referential and cascading: deleting a folder deletes its subtree
        # in one statement rather than leaving orphans behind.
        sa.ForeignKeyConstraint(
            ["parent_id"], ["document_folders.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "user_id", "parent_id", "name", name="uq_document_folder_user_parent_name"
        ),
    )
    op.create_index(
        "ix_document_folders_user_id", "document_folders", ["user_id"], unique=False
    )
    op.create_index(
        "ix_document_folders_parent_id", "document_folders", ["parent_id"], unique=False
    )
    op.create_index(
        "ix_document_folders_user_parent",
        "document_folders",
        ["user_id", "parent_id"],
        unique=False,
    )

    op.create_table(
        "document_folder_items",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column("folder_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["folder_id"], ["document_folders.id"], ondelete="CASCADE"
        ),
        # A deleted document leaves no dangling membership rows.
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("folder_id", "document_id", name="uq_document_folder_item"),
    )
    op.create_index(
        "ix_document_folder_items_folder_id",
        "document_folder_items",
        ["folder_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_folder_items_document_id",
        "document_folder_items",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_folder_items_document_id", "document_folder_items")
    op.drop_index("ix_document_folder_items_folder_id", "document_folder_items")
    op.drop_table("document_folder_items")
    op.drop_index("ix_document_folders_user_parent", "document_folders")
    op.drop_index("ix_document_folders_parent_id", "document_folders")
    op.drop_index("ix_document_folders_user_id", "document_folders")
    op.drop_table("document_folders")
