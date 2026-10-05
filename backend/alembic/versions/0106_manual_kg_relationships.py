"""Allow a knowledge-graph relationship that no document produced.

`KnowledgeGraphService.create_relationship` records a relationship somebody
stated -- an agent, or a person in the graph editor -- with `document_id` NULL,
and looks for duplicates among rows where it IS NULL. The column was NOT NULL,
so every such insert was refused: `create_kg_relationship`, `link_entities`
and `POST /kg/relationship` have never been able to succeed.

Downgrading deletes the manual relationships, since the column cannot be made
NOT NULL while they exist.

Revision ID: 0106_manual_kg_relationships
Revises: 0105_specialists_sandbox_skills
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0106_manual_kg_relationships"
down_revision = "0105_specialists_sandbox_skills"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "kg_relationships",
        "document_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM kg_relationships WHERE document_id IS NULL"))
    op.alter_column(
        "kg_relationships",
        "document_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )
