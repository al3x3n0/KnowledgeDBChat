"""
Pydantic schemas for reading lists.
"""

from datetime import datetime
from typing import List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field


ReadingStatus = Literal["to-read", "reading", "done"]


class ReadingListItemResponse(BaseModel):
    id: UUID
    reading_list_id: UUID
    document_id: UUID
    document_title: Optional[str] = None
    status: ReadingStatus
    priority: int
    position: int
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ReadingListResponse(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    description: Optional[str] = None
    source_id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    items: Optional[List[ReadingListItemResponse]] = None


class ReadingListListResponse(BaseModel):
    items: List[ReadingListResponse]
    total: int
    limit: int
    offset: int


class ReadingListCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    source_id: Optional[UUID] = None
    auto_populate_from_source: bool = False


class ReadingListUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None


class ReadingListItemCreate(BaseModel):
    document_id: UUID
    status: ReadingStatus = "to-read"
    priority: int = 0
    position: Optional[int] = None
    notes: Optional[str] = None


class ReadingListItemUpdate(BaseModel):
    status: Optional[ReadingStatus] = None
    priority: Optional[int] = None
    position: Optional[int] = None
    notes: Optional[str] = None

