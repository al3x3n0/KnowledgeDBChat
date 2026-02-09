"""
LaTeX Studio compile job model (Celery-backed async compilation).
"""

from __future__ import annotations

from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.postgresql import JSONB

from app.core.database import Base


class LatexCompileJob(Base):
    __tablename__ = "latex_compile_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celery_task_id = Column(String(255), nullable=True, index=True)

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("latex_projects.id", ondelete="CASCADE"), nullable=True, index=True)

    status = Column(String(50), nullable=False, default="queued")  # queued|running|succeeded|failed
    safe_mode = Column(Boolean, nullable=False, default=True)
    preferred_engine = Column(String(50), nullable=True)

    engine = Column(String(50), nullable=True)
    log = Column(Text, nullable=True)
    violations = Column(JSONB, nullable=False, default=list)

    pdf_file_path = Column(String(1000), nullable=True)

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

