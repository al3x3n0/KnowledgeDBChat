"""A plugin: one installable unit that may contribute tools, flows and UI.

The application already had four ways to extend it -- custom tools, workflows,
pipelines and MCP tool configuration -- and no way to treat a *set* of those as
one thing. You could not install, version, enable, disable or share a
capability; you could only build its pieces separately and remember that they
belonged together.

A plugin is that missing envelope. It owns a manifest describing what it
contributes, and installation is per user, so a plugin is a capability someone
chose rather than a change to the deployment.

Two sources, one shape. ``source='builtin'`` is a bundle shipped in this
repository and loaded from disk at startup; it has no owner and every user may
install it. ``source='user'`` was authored in the application and belongs to
the user who wrote it. The loader merges both; nothing downstream needs to know
which a plugin came from.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Plugin(Base):
    """A contributed bundle, as declared by its manifest."""

    __tablename__ = "plugins"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)

    #: Stable identifier, and the namespace its tools are offered under. Short
    #: because it becomes part of every contributed tool's name, which a
    #: provider caps at 64 characters.
    slug = Column(String(32), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(32), nullable=False, default="0.1.0")

    #: 'builtin' (shipped in this repository) or 'user' (authored here).
    source = Column(String(16), nullable=False, default="user")

    #: Null for a builtin, which belongs to the deployment rather than a person.
    owner_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
    )

    #: The manifest as validated at install. Kept whole rather than exploded
    #: into columns: it is the contributor's document, and the fields it may
    #: carry will grow with each kind of contribution we support.
    manifest = Column(JSONB, nullable=False, default=dict)

    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    installations = relationship(
        "PluginInstallation",
        back_populates="plugin",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # A slug is unique per owner: two users may each have a plugin called
        # `bench` without colliding, because a contributed tool's name is only
        # ever resolved within one user's installations.
        UniqueConstraint("owner_id", "slug", name="uq_plugins_owner_slug"),
        Index("ix_plugins_owner_id", "owner_id"),
        Index("ix_plugins_source", "source"),
    )


class PluginInstallation(Base):
    """One user's decision to run one plugin, and how they configured it."""

    __tablename__ = "plugin_installations"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    plugin_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("plugins.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    #: Disabled is not uninstalled. A plugin that misbehaves should be
    #: switchable off without losing its settings or the tools it created.
    is_enabled = Column(Boolean, nullable=False, default=True)

    #: Values for the settings the manifest declares.
    settings = Column(JSONB, nullable=False, default=dict)

    installed_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    plugin = relationship("Plugin", back_populates="installations")

    __table_args__ = (
        UniqueConstraint("plugin_id", "user_id", name="uq_plugin_installation"),
        Index("ix_plugin_installations_user_id", "user_id"),
    )
