"""
Custom exception classes for the Knowledge Database application.
"""

from typing import Optional


class KnowledgeDBException(Exception):
    """Base exception for all Knowledge Database errors."""
    
    def __init__(self, message: str, detail: Optional[str] = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class DocumentNotFoundError(KnowledgeDBException):
    """Raised when a document is not found."""
    
    def __init__(self, document_id: str, detail: Optional[str] = None):
        message = f"Document not found: {document_id}"
        super().__init__(message, detail)
        self.document_id = document_id


class VectorStoreError(KnowledgeDBException):
    """Raised when vector store operations fail."""
    
    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(f"Vector store error: {message}", detail)


class LLMServiceError(KnowledgeDBException):
    """Raised when LLM service operations fail."""
    
    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(f"LLM service error: {message}", detail)


class AuthenticationError(KnowledgeDBException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", detail: Optional[str] = None):
        super().__init__(message, detail)


class ValidationError(KnowledgeDBException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, detail: Optional[str] = None):
        if field:
            message = f"Validation error for field '{field}': {message}"
        super().__init__(message, detail)
        self.field = field

