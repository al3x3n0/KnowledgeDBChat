"""
MCP Configuration models.

Stores configuration for MCP tools and access control per API key.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class MCPToolConfig(Base):
    """Configuration for individual MCP tools per API key."""

    __tablename__ = "mcp_tool_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to API key
    api_key_id = Column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False
    )

    # Tool identification
    tool_name = Column(String(50), nullable=False)

    # Enable/disable
    is_enabled = Column(Boolean, default=True)

    # Tool-specific configuration (e.g., max results, allowed operations)
    config = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    api_key = relationship("APIKey", backref="tool_configs")

    __table_args__ = (
        UniqueConstraint("api_key_id", "tool_name", name="uq_mcp_tool_config"),
    )

    def __repr__(self):
        return f"<MCPToolConfig(api_key_id={self.api_key_id}, tool={self.tool_name}, enabled={self.is_enabled})>"


class MCPSourceAccess(Base):
    """Access control for document sources per API key."""

    __tablename__ = "mcp_source_access"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to API key
    api_key_id = Column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="CASCADE"),
        nullable=False
    )

    # Link to document source
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_sources.id", ondelete="CASCADE"),
        nullable=False
    )

    # Permissions
    can_read = Column(Boolean, default=True)
    can_search = Column(Boolean, default=True)
    can_chat = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    api_key = relationship("APIKey", backref="source_access")
    source = relationship("DocumentSource")

    __table_args__ = (
        UniqueConstraint("api_key_id", "source_id", name="uq_mcp_source_access"),
    )

    def __repr__(self):
        return f"<MCPSourceAccess(api_key_id={self.api_key_id}, source_id={self.source_id})>"


# Available MCP tools definition
MCP_TOOLS = {
    "search": {
        "name": "search",
        "display_name": "Semantic Search",
        "description": "Search documents using semantic similarity",
        "category": "read",
        "required_scope": "read",
        "config_schema": {
            "max_results": {"type": "integer", "default": 50, "min": 1, "max": 100},
        }
    },
    "list_documents": {
        "name": "list_documents",
        "display_name": "List Documents",
        "description": "List and browse documents",
        "category": "read",
        "required_scope": "read",
        "config_schema": {
            "max_results": {"type": "integer", "default": 100, "min": 1, "max": 500},
        }
    },
    "get_document": {
        "name": "get_document",
        "display_name": "Get Document",
        "description": "Retrieve a specific document by ID",
        "category": "read",
        "required_scope": "read",
        "config_schema": {
            "include_content": {"type": "boolean", "default": True},
            "max_content_length": {"type": "integer", "default": 100000},
        }
    },
    "list_sources": {
        "name": "list_sources",
        "display_name": "List Sources",
        "description": "List available document sources",
        "category": "read",
        "required_scope": "read",
        "config_schema": {}
    },
    "chat": {
        "name": "chat",
        "display_name": "Chat / Q&A",
        "description": "Ask questions and get AI-powered answers using RAG",
        "category": "chat",
        "required_scope": "chat",
        "config_schema": {
            "max_context_chunks": {"type": "integer", "default": 10, "min": 1, "max": 20},
            "include_sources": {"type": "boolean", "default": True},
        }
    },
    "web_scrape": {
        "name": "web_scrape",
        "display_name": "Web Scrape",
        "description": "Fetch web pages and extract readable text (useful for wikis/portals)",
        "category": "read",
        "required_scope": "read",
        "config_schema": {
            "max_pages": {"type": "integer", "default": 5, "min": 1, "max": 25},
            "max_depth": {"type": "integer", "default": 1, "min": 0, "max": 5},
            "same_domain_only": {"type": "boolean", "default": True},
            "include_links": {"type": "boolean", "default": True},
            "allow_private_networks": {"type": "boolean", "default": False},
            "max_content_chars": {"type": "integer", "default": 50000, "min": 1000, "max": 500000},
        },
    },
    "create_presentation": {
        "name": "create_presentation",
        "display_name": "Create Presentation",
        "description": "Generate PowerPoint presentations from topics",
        "category": "write",
        "required_scope": "write",
        "config_schema": {
            "max_slides": {"type": "integer", "default": 30, "min": 5, "max": 50},
            "allowed_styles": {"type": "array", "default": ["professional", "technical", "modern", "minimal"]},
        }
    },
    "create_repo_report": {
        "name": "create_repo_report",
        "display_name": "Create Repository Report",
        "description": "Generate reports from GitHub/GitLab repositories",
        "category": "write",
        "required_scope": "write",
        "config_schema": {
            "allowed_formats": {"type": "array", "default": ["docx", "pdf", "pptx"]},
        }
    },
    "get_job_status": {
        "name": "get_job_status",
        "display_name": "Get Job Status",
        "description": "Check status of generation jobs",
        "category": "read",
        "required_scope": "read",
        "config_schema": {}
    },
    "list_jobs": {
        "name": "list_jobs",
        "display_name": "List Jobs",
        "description": "List generation jobs",
        "category": "read",
        "required_scope": "read",
        "config_schema": {}
    },
}
