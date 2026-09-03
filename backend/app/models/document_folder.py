"""User-defined folders over the document corpus.

Documents in this application have no `user_id`: they belong to a
`DocumentSource` and are shared by everyone. Folders are therefore a per-user
*view* over shared content, not a location within it — which is why membership
is a join table rather than a column on `documents`. Filing a document into
one of your folders adds it to your view and changes nobody else's, and the
same document can sit in as many folders as are useful.

System folders (by source, by type, recent, unfiled) are deliberately absent
from these tables. They are computed from what documents already carry, so
they cannot drift out of date and a newly synced source appears in the tree
without anything having to seed it. See `document_folder_service`.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class DocumentFolder(Base):
    """One node in a user's folder tree."""

    __tablename__ = "document_folders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Self-referential: NULL parent means a root folder. Deleting a folder
    # takes its subtree with it, which is what the API's `recursive` flag
    # exists to make the caller say out loud.
    parent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_folders.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    # Explicit ordering among siblings; ties break on name.
    position = Column(Integer, nullable=False, default=0)
    # A hint for the UI only — never trusted for anything but decoration.
    color = Column(String(32), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    children = relationship(
        "DocumentFolder",
        back_populates="parent",
        cascade="all, delete-orphan",
    )
    parent = relationship("DocumentFolder", back_populates="children", remote_side=[id])
    items = relationship(
        "DocumentFolderItem",
        back_populates="folder",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Two folders may share a name only if they sit under different
        # parents. Postgres treats NULLs as distinct in a unique index, so
        # this does NOT constrain root folders; the service checks those.
        UniqueConstraint(
            "user_id", "parent_id", "name", name="uq_document_folder_user_parent_name"
        ),
        Index("ix_document_folders_user_parent", "user_id", "parent_id"),
    )

    def __repr__(self):
        return f"<DocumentFolder(id={self.id}, name='{self.name}')>"


class DocumentFolderItem(Base):
    """A document filed into a folder.

    `document_id` cascades on delete, so removing a document from the corpus
    removes it from every folder rather than leaving a row pointing at nothing.
    """

    __tablename__ = "document_folder_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_folders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    added_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    folder = relationship("DocumentFolder", back_populates="items")
    document = relationship("Document")

    __table_args__ = (
        UniqueConstraint("folder_id", "document_id", name="uq_document_folder_item"),
    )

    def __repr__(self):
        return f"<DocumentFolderItem(folder={self.folder_id}, doc={self.document_id})>"
