"""
Pydantic schemas for LaTeX project files.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LatexProjectFileResponse(BaseModel):
    id: UUID
    project_id: UUID
    filename: str
    content_type: Optional[str] = None
    file_size: int
    sha256: Optional[str] = None
    file_path: str
    download_url: Optional[str] = None
    created_at: Optional[datetime] = None


class LatexProjectFileListResponse(BaseModel):
    items: List[LatexProjectFileResponse]
    total: int


class LatexProjectFileUploadResponse(BaseModel):
    file: LatexProjectFileResponse
    replaced: bool = False

