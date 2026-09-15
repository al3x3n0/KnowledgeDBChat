"""Make a contributed capability one installable thing.

The application had four ways to extend it -- custom tools, workflows,
pipelines and MCP tool configuration -- and no way to treat a set of those as
a unit. A capability could be built but not installed, versioned, disabled or
shared.

Two tables. `plugins` holds the manifest, which is kept whole rather than
exploded into columns: it is the contributor's document and the kinds of
contribution it may declare will grow. `plugin_installations` holds one user's
decision to run one plugin, separately from the plugin itself, so a builtin
bundle shipped with the deployment has no owner and is still something each
user opts into.

Disabling is not uninstalling, and the two are different rows' worth of
meaning: `is_enabled=false` keeps the settings and the tools; deleting the
installation does not.

Revision ID: 0098_plugins
Revises: 0097_chat_message_metadata
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0098_plugins"
down_revision = "0097_chat_message_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "plugins",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # Short on purpose: the slug becomes part of every contributed tool's
        # name, and a provider caps a function name at 64 characters.
        sa.Column("slug", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "version", sa.String(length=32), nullable=False, server_default="0.1.0"
        ),
        # 'builtin' ships in the repository and is loaded from disk;
        # 'user' was authored here.
        sa.Column(
            "source", sa.String(length=16), nullable=False, server_default="user"
        ),
        # Null for a builtin, which belongs to the deployment rather than a
        # person.
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # Unique per owner rather than globally: two users may each keep a
        # plugin called `bench`, because a contributed tool's name is only ever
        # resolved within one user's own installations.
        sa.UniqueConstraint("owner_id", "slug", name="uq_plugins_owner_slug"),
    )
    op.create_index("ix_plugins_owner_id", "plugins", ["owner_id"])
    op.create_index("ix_plugins_source", "plugins", ["source"])

    op.create_table(
        "plugin_installations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "plugin_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("plugins.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "settings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "installed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("plugin_id", "user_id", name="uq_plugin_installation"),
    )
    op.create_index(
        "ix_plugin_installations_user_id", "plugin_installations", ["user_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_plugin_installations_user_id", table_name="plugin_installations")
    op.drop_table("plugin_installations")
    op.drop_index("ix_plugins_source", table_name="plugins")
    op.drop_index("ix_plugins_owner_id", table_name="plugins")
    op.drop_table("plugins")
