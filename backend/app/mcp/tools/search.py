"""
MCP Search tool for semantic document search.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.auth import MCPAuthContext
from app.models.document import Document
from app.services.config_values import bounded_int, parse_uuid
from app.services.search_service import SearchService
from app.services.vector_store import vector_store_service

# The most results one call returns, and the most documents a source filter
# hands to the store.
MAX_RESULTS = 50
MAX_SOURCE_DOCUMENTS = 10000


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
            "query": {"type": "string", "description": "The search query"},
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Filter by specific source IDs",
            },
            "file_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Filter by file types (e.g., 'pdf', 'md', 'docx')",
            },
        },
        "required": ["query"],
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

        # The number comes from an API client or a model's tool call.
        limit = bounded_int(limit, 10, 1, MAX_RESULTS)

        try:
            # Every signed-in caller may read every document, so only a named
            # source narrows the search. The ids of *all* documents used to be
            # loaded here, capped at 10,000, and results filtered against
            # them: in a larger knowledge base every document past that cap
            # was silently unsearchable.
            document_ids: Optional[List[str]] = None
            if source_ids:
                source_uuids = [parse_uuid(sid) for sid in source_ids]
                if any(uid is None for uid in source_uuids):
                    return {
                        "results": [],
                        "total": 0,
                        "query": query,
                        "error": "source_ids must be UUIDs",
                    }
                document_ids = [
                    str(doc_id) for doc_id in await self._documents_of(db, source_uuids)
                ]
                if not document_ids:
                    return {
                        "results": [],
                        "total": 0,
                        "query": query,
                        "message": "Those sources hold no documents",
                    }

            # Initialize vector store
            await vector_store_service.initialize(background=True)

            # The store filters by document, so a source's matches are not
            # crowded out of a window of other sources' chunks. More than
            # asked for is fetched only because file type is checked here.
            raw_results = await vector_store_service.search(
                query=query,
                limit=limit * 2 if file_types else limit,
                document_ids=document_ids,
            )

            wanted_types = {str(ft).lower() for ft in (file_types or [])}
            doc_id_set = set(document_ids) if document_ids is not None else None
            filtered_results = []

            for result in raw_results:
                metadata = result.get("metadata") or {}
                result_doc_id = str(metadata.get("document_id") or "")

                if doc_id_set is not None and result_doc_id not in doc_id_set:
                    continue
                if wanted_types:
                    if str(metadata.get("file_type") or "").lower() not in wanted_types:
                        continue

                filtered_results.append(
                    {
                        "content": result.get(
                            "content", result.get("page_content", "")
                        ),
                        "document_id": result_doc_id,
                        "title": metadata.get("title", "Unknown"),
                        "file_type": metadata.get("file_type"),
                        "source": metadata.get("source"),
                        "score": result.get("score", 0),
                        "chunk_index": metadata.get("chunk_index"),
                    }
                )

                if len(filtered_results) >= limit:
                    break

            return {
                "results": filtered_results,
                "total": len(filtered_results),
                "query": query,
            }

        except Exception as e:
            logger.error(f"MCP search error: {e}")
            return {"results": [], "total": 0, "query": query, "error": str(e)}

    async def _documents_of(
        self, db: AsyncSession, source_ids: List[UUID]
    ) -> List[UUID]:
        """The documents held by these sources."""
        result = await db.execute(
            select(Document.id)
            .where(Document.source_id.in_(source_ids))
            .limit(MAX_SOURCE_DOCUMENTS)
        )
        return [row[0] for row in result.fetchall()]
