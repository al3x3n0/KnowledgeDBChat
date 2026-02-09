"""
Pydantic schemas for LaTeX Studio compile jobs (async / Celery worker).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LatexCompileJobCreateRequest(BaseModel):
    safe_mode: bool = True
    preferred_engine: Optional[str] = Field(default=None, description="Optional: 'tectonic' or 'pdflatex'")


class LatexCompileJobResponse(BaseModel):
    id: UUID
    project_id: Optional[UUID] = None
    status: str
    safe_mode: bool = True
    preferred_engine: Optional[str] = None
    engine: Optional[str] = None
    log: Optional[str] = None
    violations: List[str] = []
    pdf_file_path: Optional[str] = None
    pdf_download_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

