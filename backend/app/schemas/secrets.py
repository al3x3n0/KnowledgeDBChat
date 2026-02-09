from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SecretCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    value: str = Field(..., min_length=1, max_length=10000)


class SecretResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime


class SecretRevealResponse(SecretResponse):
    value: Optional[str] = None

