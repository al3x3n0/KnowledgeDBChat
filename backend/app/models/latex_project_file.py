"""
LaTeX Studio project file model (assets like images, .bib, included .tex).
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class LatexProjectFile(Base):
    __tablename__ = "latex_project_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("latex_projects.id", ondelete="CASCADE"), nullable=False, index=True)

    filename = Column(String(255), nullable=False)
    content_type = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=False, default=0)
    sha256 = Column(String(64), nullable=True)
    file_path = Column(String(1000), nullable=False)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

