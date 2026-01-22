"""Add persona model and document persona detections.

Revision ID: 0010_add_personas
Revises: 0009_add_git_branch_tables
Create Date: 2024-05-25 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0010_add_personas"
down_revision = "0009_add_git_branch_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Personas table
    op.create_table(
        "personas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("platform_id", sa.String(length=255), nullable=True, unique=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("extra_metadata", postgresql.JSON, nullable=True),
        sa.Column("avatar_url", sa.String(length=500), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_personas_name", "personas", ["name"], unique=False)
    op.create_index("ix_personas_user_id", "personas", ["user_id"], unique=False)

    # Add owner_persona_id to documents
    op.add_column(
        "documents",
        sa.Column("owner_persona_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_documents_owner_persona_id",
        "documents",
        "personas",
        ["owner_persona_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Document persona detections
    op.create_table(
        "document_persona_detections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("persona_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default=sa.text("'detected'")),
        sa.Column("detection_type", sa.String(length=50), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("details", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["persona_id"], ["personas.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_document_persona_detection_doc", "document_persona_detections", ["document_id"])
    op.create_index("ix_document_persona_detection_persona", "document_persona_detections", ["persona_id"])


def downgrade() -> None:
    op.drop_index("ix_document_persona_detection_persona", table_name="document_persona_detections")
    op.drop_index("ix_document_persona_detection_doc", table_name="document_persona_detections")
    op.drop_table("document_persona_detections")

    op.drop_constraint("fk_documents_owner_persona_id", "documents", type_="foreignkey")
    op.drop_column("documents", "owner_persona_id")

    op.drop_index("ix_personas_user_id", table_name="personas")
    op.drop_index("ix_personas_name", table_name="personas")
    op.drop_table("personas")
