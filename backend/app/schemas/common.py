"""
Common Pydantic schemas for API responses.
"""

from typing import Generic, TypeVar, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated response schema.
    
    Attributes:
        items: List of items in the current page
        total: Total number of items
        page: Current page number (1-indexed)
        page_size: Number of items per page
        total_pages: Total number of pages
    """
    items: List[T]
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, le=100, description="Number of items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        page: int,
        page_size: int
    ) -> "PaginatedResponse[T]":
        """
        Create a paginated response.
        
        Args:
            items: List of items for current page
            total: Total number of items
            page: Current page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            PaginatedResponse instance
        """
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )


class SuccessResponse(BaseModel):
    """
    Standard success response schema.
    
    Attributes:
        success: Whether the operation was successful
        message: Success message
        data: Optional response data
        timestamp: Response timestamp
    """
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorDetail(BaseModel):
    """Error detail schema."""
    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    
    Attributes:
        error: Error type/name
        detail: Error message
        status_code: HTTP status code
        errors: List of detailed errors (for validation errors)
        correlation_id: Request correlation ID
        timestamp: Error timestamp
    """
    error: str
    detail: str
    status_code: int
    errors: Optional[List[ErrorDetail]] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetadataResponse(BaseModel, Generic[T]):
    """
    Response with metadata.
    
    Attributes:
        data: Response data
        metadata: Additional metadata
    """
    data: T
    metadata: Optional[Dict[str, Any]] = None

