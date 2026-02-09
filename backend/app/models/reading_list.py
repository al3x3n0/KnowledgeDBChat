"""
Reading lists (collections) for organizing documents.
"""

from datetime import datetime
import uuid

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ReadingList(Base):
    __tablename__ = "reading_lists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    items = relationship("ReadingListItem", back_populates="reading_list", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_reading_list_user_name"),
    )


class ReadingListItem(Base):
    __tablename__ = "reading_list_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    reading_list_id = Column(UUID(as_uuid=True), ForeignKey("reading_lists.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    status = Column(String(16), nullable=False, default="to-read")
    priority = Column(Integer, nullable=False, default=0)
    position = Column(Integer, nullable=False, default=0)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    reading_list = relationship("ReadingList", back_populates="items")

    __table_args__ = (
        UniqueConstraint("reading_list_id", "document_id", name="uq_reading_list_item_document_once"),
        Index("ix_reading_list_items_list_position", "reading_list_id", "position"),
    )

