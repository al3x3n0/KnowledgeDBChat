"""Let a person shape their own navigation, and have it follow them.

The navigation was a literal in `Layout.tsx`: five doors, twenty-odd
destinations, the same for everyone. A person who never opens LaTeX Studio
looked at it every day, and a person who lives in the Control Plane started
from Chat every morning.

What was already user-adjustable lived in `localStorage`, which is the right
home for a collapsed panel and the wrong one for this: hiding a destination is
a decision about your work, not about this browser, and making it again on
every machine is the tell that it was stored in the wrong place.

One JSON column rather than a table per concern. The shape grows with each kind
of customization -- order, hidden, renames, pins, landing page, and later the
entries plugins contribute -- and none of it is ever queried across users.

Revision ID: 0099_user_ui_preferences
Revises: 0098_plugins
"""

import sqlalchemy as sa

from alembic import op

revision = "0099_user_ui_preferences"
down_revision = "0098_plugins"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable with no default: null means "has never customized anything",
    # which is different from "customized it back to the defaults" and is the
    # state every existing row is genuinely in.
    op.add_column("user_preferences", sa.Column("ui", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("user_preferences", "ui")
