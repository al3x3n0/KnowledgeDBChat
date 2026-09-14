"""Let a chat message carry structured data beside its prose.

A message could hold text, the documents it cited, and timing -- but nothing
open-ended. That was enough while the assistant only ever answered, and stops
being enough the moment a reply can also *offer* something: a drafted research
campaign the user reviews and launches.

The draft has to persist. Returning it only in the POST response would make
the widget vanish when the session is reopened, which turns an offer into a
thing you had to accept immediately or lose.

JSON rather than JSONB to match the column type the rest of this table and
`chat_sessions.extra_metadata` already use. Nothing compares these values, so
the equality operator JSONB adds is not needed here -- and mixing the two in
one table is how `SELECT DISTINCT` starts failing.

Revision ID: 0097_chat_message_metadata
Revises: 0096_coding_workspaces
"""

import sqlalchemy as sa
from alembic import op

revision = "0097_chat_message_metadata"
down_revision = "0096_coding_workspaces"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("extra_metadata", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "extra_metadata")
