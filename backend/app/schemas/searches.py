from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SavedSearchCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    query: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[Dict[str, Any]] = None


class SavedSearchResponse(BaseModel):
    id: UUID
    name: str
    query: str
    filters: Optional[Dict[str, Any]] = None
    is_system: bool = False  # True for pre-built system-wide searches
    created_at: datetime
    updated_at: datetime


class SearchShareCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[Dict[str, Any]] = None


class SearchShareResponse(BaseModel):
    token: str
    query: str
    filters: Optional[Dict[str, Any]] = None
    created_at: datetime

