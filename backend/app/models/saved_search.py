"""
Saved searches and share links.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.sql import func

from app.core.database import Base


class SavedSearch(Base):
    __tablename__ = "saved_searches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    # user_id can be NULL for system-wide pre-built searches
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)

    name = Column(String(255), nullable=False)
    query = Column(Text, nullable=False)
    filters = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class SearchShare(Base):
    __tablename__ = "search_shares"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    token = Column(String(64), nullable=False, unique=True, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    query = Column(Text, nullable=False)
    filters = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

