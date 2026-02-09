"""
Pydantic schemas for LaTeX Studio projects.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LatexProjectCreate(BaseModel):
    title: str = Field(default="Untitled LaTeX Project", min_length=1, max_length=500)
    tex_source: str = Field(..., min_length=1)


class LatexProjectUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    tex_source: Optional[str] = Field(default=None, min_length=1)


class LatexProjectResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    tex_source: str
    tex_file_path: Optional[str] = None
    pdf_file_path: Optional[str] = None
    pdf_download_url: Optional[str] = None
    last_compile_engine: Optional[str] = None
    last_compile_log: Optional[str] = None
    last_compiled_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class LatexProjectListItem(BaseModel):
    id: UUID
    title: str
    updated_at: Optional[datetime] = None
    last_compiled_at: Optional[datetime] = None


class LatexProjectListResponse(BaseModel):
    items: List[LatexProjectListItem]
    total: int
    limit: int
    offset: int


class LatexProjectCompileResponse(BaseModel):
    success: bool
    engine: Optional[str] = None
    pdf_file_path: Optional[str] = None
    pdf_download_url: Optional[str] = None
    log: str = ""
    violations: List[str] = []


class LatexProjectCompileRequest(BaseModel):
    safe_mode: bool = True
    preferred_engine: Optional[str] = None


class LatexProjectPublishRequest(BaseModel):
    include_tex: bool = True
    include_pdf: bool = True
    safe_mode: bool = True
    tags: Optional[List[str]] = None


class LatexProjectPublishItem(BaseModel):
    kind: str = Field(..., pattern="^(tex|pdf)$")
    document_id: UUID
    title: str
    file_type: Optional[str] = None
    file_path: Optional[str] = None


class LatexProjectPublishSkipped(BaseModel):
    kind: str = Field(..., pattern="^(tex|pdf)$")
    reason: str


class LatexProjectPublishResponse(BaseModel):
    project_id: UUID
    published: List[LatexProjectPublishItem]
    skipped: List[LatexProjectPublishSkipped] = []
