"""
Base connector class for data source integrations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class BaseConnector(ABC):
    """Base class for all data source connectors."""
    
    def __init__(self):
        self.config = {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the connector with configuration.
        
        Args:
            config: Connector-specific configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test the connection to the data source.
        
        Returns:
            True if connection is successful
        """
        pass
    
    @abstractmethod
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all available documents from the source.
        
        Returns:
            List of document metadata dictionaries with required fields:
            - identifier: Unique identifier for the document
            - title: Document title
            - url: Document URL (optional)
            - last_modified: Last modification date (optional)
            - author: Document author (optional)
            - file_type: File type/extension (optional)
            - metadata: Additional metadata (optional)
        """
        pass
    
    @abstractmethod
    async def get_document_content(self, identifier: str) -> str:
        """
        Get the content of a specific document.
        
        Args:
            identifier: Document identifier from list_documents
            
        Returns:
            Document content as text
        """
        pass
    
    @abstractmethod
    async def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        """
        Get metadata for a specific document.
        
        Args:
            identifier: Document identifier
            
        Returns:
            Document metadata dictionary
        """
        pass
    
    async def list_changed_documents(self, since: datetime) -> List[Dict[str, Any]]:
        """
        List documents that have changed since a specific date.
        
        Args:
            since: Date to check changes from
            
        Returns:
            List of changed document metadata
            
        Note: Override this method if the connector supports incremental sync
        """
        # Default implementation returns all documents
        return await self.list_documents()
    
    async def cleanup(self):
        """Clean up resources used by the connector."""
        pass
    
    def _ensure_initialized(self):
        """Ensure the connector is initialized."""
        if not self.is_initialized:
            raise RuntimeError("Connector not initialized. Call initialize() first.")


class DocumentInfo:
    """Standard document information structure."""
    
    def __init__(
        self,
        identifier: str,
        title: str,
        url: Optional[str] = None,
        last_modified: Optional[datetime] = None,
        author: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.identifier = identifier
        self.title = title
        self.url = url
        self.last_modified = last_modified
        self.author = author
        self.file_type = file_type
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "identifier": self.identifier,
            "title": self.title,
            "url": self.url,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "author": self.author,
            "file_type": self.file_type,
            "metadata": self.metadata
        }







