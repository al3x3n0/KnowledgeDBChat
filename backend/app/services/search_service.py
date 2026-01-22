"""
Search service for multi-mode document search.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, desc, asc, func
from loguru import logger

from app.models.document import Document, DocumentChunk
from app.services.vector_store import VectorStoreService
from app.services.storage_service import storage_service
from app.core.config import settings


class SearchService:
    """Service for searching documents with multiple modes."""

    def __init__(self):
        self.vector_store = VectorStoreService()
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure vector store is initialized."""
        if not self._initialized:
            await self.vector_store.initialize()
            self._initialized = True

    async def search(
        self,
        query: str,
        mode: str = "smart",
        page: int = 1,
        page_size: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        source_id: Optional[str] = None,
        file_type: Optional[str] = None,
        db: Optional[AsyncSession] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Search documents using specified mode.

        Args:
            query: Search query text
            mode: Search mode - 'smart' (hybrid), 'keyword' (BM25), 'exact' (SQL LIKE)
            page: Page number (1-indexed)
            page_size: Results per page
            sort_by: Sort field - 'relevance', 'date', 'title'
            sort_order: Sort direction - 'asc', 'desc'
            source_id: Filter by source ID
            file_type: Filter by file type
            db: Database session (required for exact mode)

        Returns:
            Tuple of (results list, total count, search time in ms)
        """
        start_time = time.time()

        if mode == "exact":
            if not db:
                raise ValueError("Database session required for exact search mode")
            results, total = await self._exact_search(
                query=query,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
                source_id=source_id,
                file_type=file_type,
                db=db,
            )
        else:
            # Smart or keyword mode - use vector store
            await self._ensure_initialized()
            results, total = await self._vector_search(
                query=query,
                mode=mode,
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order,
                source_id=source_id,
                file_type=file_type,
                db=db,
            )

        took_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Search completed: mode={mode}, query='{query[:50]}...', results={len(results)}, total={total}, took={took_ms}ms")

        return results, total, took_ms

    async def _vector_search(
        self,
        query: str,
        mode: str,
        page: int,
        page_size: int,
        sort_by: str,
        sort_order: str,
        source_id: Optional[str],
        file_type: Optional[str],
        db: Optional[AsyncSession],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Execute vector-based search (smart or keyword mode)."""
        # Build metadata filter
        filter_metadata = {}
        if source_id:
            filter_metadata["source_id"] = source_id
        if file_type:
            filter_metadata["file_type"] = file_type

        # Get more results than needed for pagination
        # Vector search doesn't support offset, so we fetch extra and slice
        fetch_limit = page * page_size + page_size

        # Execute search
        raw_results = await self.vector_store.search(
            query=query,
            limit=fetch_limit,
            filter_metadata=filter_metadata if filter_metadata else None,
        )

        # Deduplicate by document_id (keep highest scoring chunk per document)
        seen_docs: Dict[str, Dict[str, Any]] = {}
        for result in raw_results:
            doc_id = result.get("metadata", {}).get("document_id", result.get("id"))
            if doc_id not in seen_docs:
                seen_docs[doc_id] = result
            else:
                # Keep higher score
                existing_score = seen_docs[doc_id].get("score", 0)
                new_score = result.get("score", 0)
                if new_score > existing_score:
                    seen_docs[doc_id] = result

        deduped_results = list(seen_docs.values())

        # Sort results
        if sort_by == "relevance":
            deduped_results.sort(key=lambda x: x.get("score", 0), reverse=(sort_order == "desc"))
        elif sort_by == "date":
            deduped_results.sort(
                key=lambda x: x.get("metadata", {}).get("updated_at", ""),
                reverse=(sort_order == "desc")
            )
        elif sort_by == "title":
            deduped_results.sort(
                key=lambda x: x.get("metadata", {}).get("title", "").lower(),
                reverse=(sort_order == "desc")
            )

        total = len(deduped_results)

        # Paginate
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = deduped_results[start_idx:end_idx]

        # Transform to response format
        results = []
        for item in paginated:
            metadata = item.get("metadata", {})
            content = item.get("content") or item.get("page_content", "")

            # Generate download URL if we have db session
            download_url = None
            if db:
                doc_id = metadata.get("document_id")
                if doc_id:
                    try:
                        stmt = select(Document).where(Document.id == UUID(doc_id))
                        doc_result = await db.execute(stmt)
                        document = doc_result.scalar_one_or_none()
                        if document and document.file_path:
                            download_url = await storage_service.get_presigned_download_url(document.file_path)
                    except Exception as e:
                        logger.warning(f"Failed to get download URL for {doc_id}: {e}")

            results.append({
                "id": metadata.get("document_id", item.get("id")),
                "title": metadata.get("title", "Unknown"),
                "source": metadata.get("source_name", metadata.get("source", "Unknown")),
                "source_type": metadata.get("source_type", metadata.get("source", "unknown")),
                "file_type": metadata.get("file_type"),
                "author": metadata.get("author"),
                "snippet": content[:300] if content else "",
                "relevance_score": item.get("score", 0.0),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", ""),
                "url": metadata.get("url"),
                "download_url": download_url,
                "chunk_id": metadata.get("chunk_id"),
            })

        return results, total

    async def _exact_search(
        self,
        query: str,
        page: int,
        page_size: int,
        sort_by: str,
        sort_order: str,
        source_id: Optional[str],
        file_type: Optional[str],
        db: AsyncSession,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Execute exact SQL LIKE search."""
        from app.models.document import DocumentSource

        search_term = f"%{query}%"

        # Build base query
        base_query = select(Document).join(DocumentSource)

        # Apply filters
        conditions = [
            Document.is_processed == True,
            or_(
                Document.title.ilike(search_term),
                Document.content.ilike(search_term),
                Document.author.ilike(search_term),
            )
        ]

        if source_id:
            conditions.append(Document.source_id == UUID(source_id))
        if file_type:
            conditions.append(Document.file_type == file_type)

        base_query = base_query.where(*conditions)

        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply sorting
        if sort_by == "date":
            order_col = Document.updated_at
        elif sort_by == "title":
            order_col = Document.title
        else:
            # For exact search, relevance doesn't apply, default to date
            order_col = Document.updated_at

        if sort_order == "desc":
            base_query = base_query.order_by(desc(order_col))
        else:
            base_query = base_query.order_by(asc(order_col))

        # Apply pagination
        offset = (page - 1) * page_size
        base_query = base_query.offset(offset).limit(page_size)

        # Execute
        result = await db.execute(base_query)
        documents = result.scalars().all()

        # Transform to response format
        results = []
        for doc in documents:
            # Generate snippet from content
            snippet = ""
            if doc.content:
                # Find query in content and extract surrounding text
                content_lower = doc.content.lower()
                query_lower = query.lower()
                idx = content_lower.find(query_lower)
                if idx >= 0:
                    start = max(0, idx - 100)
                    end = min(len(doc.content), idx + len(query) + 200)
                    snippet = doc.content[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(doc.content):
                        snippet = snippet + "..."
                else:
                    snippet = doc.content[:300]

            # Get download URL
            download_url = None
            if doc.file_path:
                try:
                    download_url = await storage_service.get_presigned_download_url(doc.file_path)
                except Exception as e:
                    logger.warning(f"Failed to get download URL for {doc.id}: {e}")

            results.append({
                "id": str(doc.id),
                "title": doc.title,
                "source": doc.source.name if doc.source else "Unknown",
                "source_type": doc.source.source_type if doc.source else "unknown",
                "file_type": doc.file_type,
                "author": doc.author,
                "snippet": snippet,
                "relevance_score": 1.0,  # Exact match doesn't have relevance score
                "created_at": doc.created_at.isoformat() if doc.created_at else "",
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else "",
                "url": doc.url,
                "download_url": download_url,
                "chunk_id": None,  # No specific chunk for exact search
            })

        return results, total


# Singleton instance
search_service = SearchService()
