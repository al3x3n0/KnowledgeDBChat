"""
LaTeX Studio project model.
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class LatexProject(Base):
    __tablename__ = "latex_projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    title = Column(String(500), nullable=False, default="Untitled LaTeX Project")
    tex_source = Column(Text, nullable=False, default="")

    tex_file_path = Column(String(1000), nullable=True)
    pdf_file_path = Column(String(1000), nullable=True)

    last_compile_engine = Column(String(50), nullable=True)
    last_compile_log = Column(Text, nullable=True)
    last_compiled_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

