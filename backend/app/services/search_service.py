"""
Search service for multi-mode document search.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, desc, asc, func
from loguru import logger

from app.models.document import Document, DocumentChunk
from app.services.vector_store import vector_store_service
from app.core.config import settings


class SearchService:
    """Service for searching documents with multiple modes."""

    def __init__(self):
        self.vector_store = vector_store_service
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure vector store is initialized."""
        if not self._initialized:
            await self.vector_store.initialize(background=True)
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
            apply_postprocessing=False,
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
                "download_url": None,
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
                "download_url": None,
                "chunk_id": None,  # No specific chunk for exact search
            })

        return results, total


    async def faceted_search(
        self,
        query: str,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a faceted search with aggregations.

        Args:
            query: Search query text
            db: Database session
            page: Page number
            page_size: Results per page
            filters: Filter criteria (source_type, file_type, author, tags, date_range)

        Returns:
            Search results with facet aggregations
        """
        from app.models.document import DocumentSource
        from collections import Counter

        # Apply filters
        source_id = None
        file_type = None

        if filters:
            if filters.get("source_id"):
                source_id = filters["source_id"]
            if filters.get("file_type"):
                file_type = filters["file_type"]

        # Execute search
        results, total, took_ms = await self.search(
            query=query,
            mode="smart",
            page=page,
            page_size=page_size,
            source_id=source_id,
            file_type=file_type,
            db=db,
        )

        # Get all matching documents for facet computation (limited sample)
        await self._ensure_initialized()
        all_results = await self.vector_store.search(
            query=query,
            limit=500,  # Sample for facets
            apply_postprocessing=False,
        )

        # Compute facets
        source_types = Counter()
        file_types = Counter()
        authors = Counter()
        tags = Counter()
        date_buckets = Counter()

        for item in all_results:
            metadata = item.get("metadata", {})

            source_type = metadata.get("source_type")
            if source_type:
                source_types[source_type] += 1

            ft = metadata.get("file_type")
            if ft:
                file_types[ft] += 1

            author = metadata.get("author")
            if author:
                authors[author] += 1

            item_tags = metadata.get("tags", [])
            if item_tags:
                for tag in item_tags:
                    tags[tag] += 1

            created_at = metadata.get("created_at")
            if created_at:
                try:
                    from datetime import datetime
                    if isinstance(created_at, str):
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        dt = created_at
                    year_month = dt.strftime("%Y-%m")
                    date_buckets[year_month] += 1
                except Exception:
                    pass

        return {
            "query": query,
            "results": results,
            "total": total,
            "took_ms": took_ms,
            "page": page,
            "page_size": page_size,
            "facets": {
                "source_type": dict(source_types.most_common(10)),
                "file_type": dict(file_types.most_common(10)),
                "author": dict(authors.most_common(10)),
                "tags": dict(tags.most_common(20)),
                "date": dict(sorted(date_buckets.items(), reverse=True)[:12]),
            },
            "filters_applied": filters or {},
        }

    async def get_search_suggestions(
        self,
        partial_query: str,
        db: AsyncSession,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get search suggestions/autocomplete for partial query.

        Args:
            partial_query: Partial search query
            db: Database session
            limit: Maximum suggestions

        Returns:
            List of search suggestions
        """
        if len(partial_query) < 2:
            return []

        suggestions = []
        partial_lower = partial_query.lower()

        # Search in document titles
        title_query = select(Document.title).where(
            and_(
                Document.is_processed == True,
                Document.title.ilike(f"%{partial_query}%")
            )
        ).limit(limit * 2)
        title_result = await db.execute(title_query)
        titles = [row[0] for row in title_result.fetchall() if row[0]]

        for title in titles[:limit]:
            suggestions.append({
                "type": "title",
                "text": title,
                "display": f"📄 {title}",
            })

        # Search in tags
        tag_query = select(Document.tags).where(
            and_(
                Document.is_processed == True,
                Document.tags != None
            )
        ).limit(100)
        tag_result = await db.execute(tag_query)

        all_tags = set()
        for row in tag_result.fetchall():
            if row[0]:
                for tag in row[0]:
                    if partial_lower in tag.lower():
                        all_tags.add(tag)

        for tag in list(all_tags)[:limit - len(suggestions)]:
            suggestions.append({
                "type": "tag",
                "text": f"tag:{tag}",
                "display": f"🏷️ {tag}",
            })

        # Search in authors
        author_query = select(func.distinct(Document.author)).where(
            and_(
                Document.is_processed == True,
                Document.author != None,
                Document.author.ilike(f"%{partial_query}%")
            )
        ).limit(limit)
        author_result = await db.execute(author_query)
        authors = [row[0] for row in author_result.fetchall() if row[0]]

        for author in authors[:limit - len(suggestions)]:
            suggestions.append({
                "type": "author",
                "text": f"author:{author}",
                "display": f"👤 {author}",
            })

        return suggestions[:limit]

    async def get_related_searches(
        self,
        query: str,
        db: AsyncSession,
        limit: int = 5,
    ) -> List[str]:
        """
        Get related search queries based on current query.

        Args:
            query: Current search query
            db: Database session
            limit: Maximum related searches

        Returns:
            List of related search queries
        """
        await self._ensure_initialized()

        # Search for documents matching the query
        results = await self.vector_store.search(
            query=query,
            limit=20,
            apply_postprocessing=False,
        )

        if not results:
            return []

        # Extract terms from result titles and content
        from collections import Counter
        import re

        query_words = set(query.lower().split())
        term_counter = Counter()

        for item in results:
            metadata = item.get("metadata", {})
            title = metadata.get("title", "")
            content = item.get("content", "") or item.get("page_content", "")

            # Extract words from title
            title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            for word in title_words:
                if word not in query_words and len(word) > 3:
                    term_counter[word] += 3  # Weight title words higher

            # Extract words from content snippet
            content_words = re.findall(r'\b[a-zA-Z]{4,}\b', content[:500].lower())
            for word in content_words:
                if word not in query_words:
                    term_counter[word] += 1

            # Add tags as related searches
            tags = metadata.get("tags", [])
            if tags:
                for tag in tags:
                    if tag.lower() not in query_words:
                        term_counter[tag.lower()] += 5

        # Generate related queries
        related = []
        common_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 'could', 'should', 'about', 'into', 'more', 'some', 'such', 'than', 'then', 'there', 'these', 'they', 'their', 'what', 'when', 'where', 'which', 'while', 'your', 'other'}

        for term, count in term_counter.most_common(limit * 3):
            if term not in common_words and count >= 2:
                # Create related search combining original query with new term
                if len(query.split()) <= 3:
                    related.append(f"{query} {term}")
                else:
                    related.append(term)

            if len(related) >= limit:
                break

        return related

    async def get_search_history_suggestions(
        self,
        user_id: Optional[UUID],
        db: AsyncSession,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get search suggestions based on user's search history.

        Args:
            user_id: User ID for personalized suggestions
            db: Database session
            limit: Maximum suggestions

        Returns:
            List of recent/popular searches
        """
        # For now, return popular document topics as suggestions
        # This could be extended to track actual search history

        from app.models.document import DocumentSource

        # Get popular sources
        source_query = select(
            DocumentSource.name,
            func.count(Document.id).label('doc_count')
        ).join(Document).where(
            Document.is_processed == True
        ).group_by(DocumentSource.name).order_by(
            desc(func.count(Document.id))
        ).limit(limit)

        source_result = await db.execute(source_query)
        sources = source_result.fetchall()

        suggestions = []
        for source_name, count in sources:
            suggestions.append({
                "type": "popular_source",
                "text": f"source:{source_name}",
                "display": f"📁 {source_name} ({count} docs)",
            })

        return suggestions


# Singleton instance
search_service = SearchService()
