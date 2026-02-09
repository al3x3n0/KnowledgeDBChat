"""
MCP Search tool for semantic document search.
"""

from typing import List, Optional, Any, Dict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.mcp.auth import MCPAuthContext
from app.models.document import Document, DocumentSource
from app.services.vector_store import vector_store_service
from app.services.search_service import SearchService


class SearchTool:
    """
    Search tool for MCP.

    Provides semantic search over documents with user permission filtering.
    """

    name = "search"
    description = "Search documents in the knowledge base using semantic search"

    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
                "minimum": 1,
                "maximum": 50
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Filter by specific source IDs"
            },
            "file_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Filter by file types (e.g., 'pdf', 'md', 'docx')"
            },
        },
        "required": ["query"]
    }

    def __init__(self):
        self.search_service = SearchService()

    async def execute(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        query: str,
        limit: int = 10,
        source_ids: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute semantic search with user permission filtering.

        Args:
            auth: MCP authentication context
            db: Database session
            query: Search query string
            limit: Maximum results
            source_ids: Optional source ID filter
            file_types: Optional file type filter

        Returns:
            Search results with metadata
        """
        auth.require_scope("read")

        logger.info(f"MCP search: query='{query[:50]}...', user={auth.user.username}")

        try:
            # Get user's accessible document IDs
            accessible_doc_ids = await self._get_accessible_documents(auth, db, source_ids)

            if not accessible_doc_ids:
                return {
                    "results": [],
                    "total": 0,
                    "query": query,
                    "message": "No accessible documents found"
                }

            # Initialize vector store
            await vector_store_service.initialize(background=True)

            # Perform search
            raw_results = await vector_store_service.search(
                query=query,
                limit=limit * 2  # Get more to filter
            )

            # Filter results by accessible documents
            doc_id_set = set(str(doc_id) for doc_id in accessible_doc_ids)
            filtered_results = []

            for result in raw_results:
                metadata = result.get("metadata", {})
                result_doc_id = metadata.get("document_id", "")

                if result_doc_id in doc_id_set:
                    # Apply file type filter if specified
                    if file_types:
                        file_type = metadata.get("file_type", "")
                        if file_type.lower() not in [ft.lower() for ft in file_types]:
                            continue

                    filtered_results.append({
                        "content": result.get("content", result.get("page_content", "")),
                        "document_id": result_doc_id,
                        "title": metadata.get("title", "Unknown"),
                        "file_type": metadata.get("file_type"),
                        "source": metadata.get("source"),
                        "score": result.get("score", 0),
                        "chunk_index": metadata.get("chunk_index"),
                    })

                    if len(filtered_results) >= limit:
                        break

            return {
                "results": filtered_results,
                "total": len(filtered_results),
                "query": query,
            }

        except Exception as e:
            logger.error(f"MCP search error: {e}")
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": str(e)
            }

    async def _get_accessible_documents(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        source_ids: Optional[List[str]] = None,
    ) -> List[UUID]:
        """Get document IDs the user can access."""
        # Build query for documents
        query = select(Document.id)

        # If user is not admin, filter by their sources
        # For now, we allow access to all documents for authenticated users
        # In a more restrictive setup, you'd filter by ownership or explicit grants

        if source_ids:
            # Filter by specific sources
            source_uuids = [UUID(sid) for sid in source_ids]
            query = query.where(Document.source_id.in_(source_uuids))

        result = await db.execute(query.limit(10000))  # Reasonable limit
        return [row[0] for row in result.fetchall()]
