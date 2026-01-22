"""
Pydantic schemas for DOCX editor API.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class DocxEditResponse(BaseModel):
    """Response when fetching a document for editing."""
    html_content: str
    document_title: str
    document_id: str
    version: str  # Content hash for conflict detection
    editable: bool = True
    warnings: Optional[list] = None  # Any conversion warnings


class DocxEditRequest(BaseModel):
    """Request to save edited document content."""
    html_content: str
    version: str  # Must match current version to save
    create_backup: bool = True


class DocxSaveResponse(BaseModel):
    """Response after saving document edits."""
    success: bool
    document_id: str
    new_version: str
    message: str
    backup_path: Optional[str] = None
