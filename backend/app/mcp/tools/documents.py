"""
MCP Documents tool for document retrieval.
"""

from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.auth import MCPAuthContext
from app.models.document import Document, DocumentSource
from app.services.config_values import bounded_int

# The most rows one list call returns, and the furthest it may skip.
MAX_LIST_LIMIT = 200
MAX_LIST_OFFSET = 1_000_000


def _sync_status(source: DocumentSource) -> str:
    """One word for where a source's sync stands."""
    if source.is_syncing:
        return "syncing"
    if source.last_error:
        return "error"
    return "synced" if source.last_sync else "never_synced"


class DocumentsTool:
    """
    Documents tool for MCP.

    Provides document listing and retrieval with user permission filtering.
    """

    name = "documents"
    description = "List and retrieve documents from the knowledge base"

    # Tool has multiple operations
    operations = {
        "list": {
            "description": "List available documents",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "offset": {"type": "integer", "default": 0, "minimum": 0},
                    "source_id": {
                        "type": "string",
                        "description": "Filter by source ID",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file type",
                    },
                    "search": {
                        "type": "string",
                        "description": "Search in document titles",
                    },
                },
            },
        },
        "get": {
            "description": "Get a specific document by ID",
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Document UUID"},
                    "include_content": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include full document content",
                    },
                },
                "required": ["document_id"],
            },
        },
        "list_sources": {
            "description": "List available document sources",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 100,
                    }
                },
            },
        },
    }

    async def list_documents(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0,
        source_id: Optional[str] = None,
        file_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List documents accessible to the user."""
        auth.require_scope("read")

        # The number comes from an API client or from a model's tool call: a
        # negative limit is "no limit" to the database.
        limit = bounded_int(limit, 20, 1, MAX_LIST_LIMIT)
        offset = bounded_int(offset, 0, 0, MAX_LIST_OFFSET)

        logger.info(f"MCP list_documents: user={auth.user.username}, limit={limit}")

        try:
            # Build query
            query = select(Document)

            if source_id:
                query = query.where(Document.source_id == UUID(source_id))

            if file_type:
                query = query.where(Document.file_type == file_type)

            if search:
                query = query.where(Document.title.ilike(f"%{search}%"))

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total = total_result.scalar() or 0

            # Get paginated results
            query = query.order_by(Document.created_at.desc())
            query = query.offset(offset).limit(limit)

            result = await db.execute(query)
            documents = result.scalars().all()

            return {
                "documents": [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "file_type": doc.file_type,
                        "source_id": str(doc.source_id) if doc.source_id else None,
                        "file_size": doc.file_size,
                        "created_at": doc.created_at.isoformat()
                        if doc.created_at
                        else None,
                        "updated_at": doc.updated_at.isoformat()
                        if doc.updated_at
                        else None,
                        "summary": doc.summary[:500] if doc.summary else None,
                    }
                    for doc in documents
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            logger.error(f"MCP list_documents error: {e}")
            return {"documents": [], "total": 0, "error": str(e)}

    async def get_document(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        document_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Get a specific document by ID."""
        auth.require_scope("read")

        logger.info(f"MCP get_document: id={document_id}, user={auth.user.username}")

        try:
            doc_uuid = UUID(document_id)
            result = await db.execute(select(Document).where(Document.id == doc_uuid))
            doc = result.scalar_one_or_none()

            if not doc:
                return {"error": "Document not found", "document_id": document_id}

            response = {
                "id": str(doc.id),
                "title": doc.title,
                "file_type": doc.file_type,
                "source_id": str(doc.source_id) if doc.source_id else None,
                "file_path": doc.file_path,
                "file_size": doc.file_size,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "summary": doc.summary,
                # The column is mapped as `extra_metadata`; `doc.metadata` is
                # SQLAlchemy's table registry, which no response can carry.
                "metadata": doc.extra_metadata,
            }

            if include_content and doc.content:
                # Limit content size for response
                max_content_length = 100000  # 100KB
                content = doc.content
                if len(content) > max_content_length:
                    content = content[:max_content_length]
                    response["content_truncated"] = True
                response["content"] = content

            return response

        except ValueError:
            return {"error": "Invalid document ID format", "document_id": document_id}
        except Exception as e:
            logger.error(f"MCP get_document error: {e}")
            return {"error": str(e), "document_id": document_id}

    async def list_sources(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List available document sources."""
        auth.require_scope("read")

        limit = bounded_int(limit, 50, 1, MAX_LIST_LIMIT)

        logger.info(f"MCP list_sources: user={auth.user.username}")

        try:
            result = await db.execute(
                select(DocumentSource)
                .where(DocumentSource.is_active.is_(True))
                .order_by(DocumentSource.name)
                .limit(limit)
            )
            sources = result.scalars().all()

            counts: Dict[Any, int] = {}
            if sources:
                counted = await db.execute(
                    select(Document.source_id, func.count())
                    .where(Document.source_id.in_([src.id for src in sources]))
                    .group_by(Document.source_id)
                )
                counts = {source_id: count for source_id, count in counted.all()}

            # Four of these keys were read from attributes the model does not
            # have, so every call ended in the `except` below with no sources.
            return {
                "sources": [
                    {
                        "id": str(src.id),
                        "name": src.name,
                        "source_type": src.source_type,
                        "description": (src.config or {}).get("description")
                        if isinstance(src.config, dict)
                        else None,
                        "document_count": counts.get(src.id, 0),
                        "last_sync_at": src.last_sync.isoformat()
                        if src.last_sync
                        else None,
                        "sync_status": _sync_status(src),
                    }
                    for src in sources
                ],
                "total": len(sources),
            }

        except Exception as e:
            logger.error(f"MCP list_sources error: {e}")
            return {"sources": [], "error": str(e)}
