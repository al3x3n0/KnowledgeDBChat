"""
Persona-related Pydantic schemas.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class PersonaBase(BaseModel):
    """Shared fields for persona schemas."""

    name: str = Field(..., min_length=1, max_length=255)
    platform_id: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    avatar_url: Optional[str] = Field(None, max_length=500)
    extra_metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[UUID] = None
    is_active: bool = True
    is_system: bool = False


class PersonaCreate(PersonaBase):
    """Schema for creating personas."""

    pass


class PersonaUpdate(BaseModel):
    """Schema for updating personas."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    platform_id: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    avatar_url: Optional[str] = Field(None, max_length=500)
    extra_metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[UUID] = None
    is_active: Optional[bool] = None
    is_system: Optional[bool] = None


class PersonaResponse(PersonaBase):
    """Response schema for personas."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentPersonaDetectionBase(BaseModel):
    """Shared persona detection fields."""

    persona_id: UUID
    role: str = Field("detected", max_length=50)
    detection_type: Optional[str] = Field(None, max_length=50)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    start_time: Optional[float] = Field(None, ge=0)
    end_time: Optional[float] = Field(None, ge=0)
    details: Optional[Dict[str, Any]] = None


class DocumentPersonaDetectionCreate(DocumentPersonaDetectionBase):
    """Schema for creating persona detections."""

    pass


class DocumentPersonaDetectionResponse(DocumentPersonaDetectionBase):
    """Response schema for persona detections."""

    id: UUID
    created_at: datetime
    persona: PersonaResponse

    class Config:
        from_attributes = True


class PersonaEditRequestCreate(BaseModel):
    """Schema for creating persona edit requests."""

    message: str = Field(..., min_length=5, max_length=4000)
    document_id: Optional[UUID] = None


class PersonaEditRequestResponse(BaseModel):
    """Response schema for persona edit requests."""

    id: UUID
    persona_id: UUID
    requested_by: Optional[UUID]
    requested_by_name: Optional[str]
    document_id: Optional[UUID]
    message: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True
