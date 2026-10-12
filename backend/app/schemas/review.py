"""What is waiting on one person, as the Review page reads it."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ReviewItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    detail: Optional[str] = None
    waiting_since: Optional[datetime] = None
    link: Optional[str] = None
    kind: Optional[str] = None


class ReviewSectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    title: str
    description: str
    link: str
    count: int = 0
    oldest: Optional[datetime] = None
    items: List[ReviewItemResponse] = []
    error: Optional[str] = None


class ReviewInboxResponse(BaseModel):
    #: Across every section that could be read.
    total: int
    sections: List[ReviewSectionResponse]
