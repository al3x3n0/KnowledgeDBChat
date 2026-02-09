"""
Analytics service for data analysis and visualization tools.

Provides statistics, charts, and data export capabilities for the knowledge base.
"""

import json
import csv
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from collections import Counter, defaultdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from loguru import logger

from app.models.document import Document, DocumentChunk, DocumentSource
from app.services.vector_store import vector_store_service


class AnalyticsService:
    """Service for analytics, statistics, and data export."""

    def __init__(self):
        self.vector_store = vector_store_service

    async def get_collection_statistics(
        self,
        db: AsyncSession,
        source_id: Optional[UUID] = None,
        tag: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a document collection.

        Args:
            db: Database session
            source_id: Filter by document source
            tag: Filter by tag
            date_from: Start date filter
            date_to: End date filter

        Returns:
            Dictionary with collection statistics
        """
        # Build base query
        base_conditions = [Document.is_processed == True]

        if source_id:
            base_conditions.append(Document.source_id == source_id)
        if date_from:
            base_conditions.append(Document.created_at >= date_from)
        if date_to:
            base_conditions.append(Document.created_at <= date_to)

        # Document counts
        count_query = select(func.count(Document.id)).where(and_(*base_conditions))
        total_docs = (await db.execute(count_query)).scalar() or 0

        # If filtering by tag, apply separately
        if tag:
            tag_count_query = select(func.count(Document.id)).where(
                and_(*base_conditions, Document.tags.contains([tag]))
            )
            total_docs = (await db.execute(tag_count_query)).scalar() or 0

        # Total content size
        size_query = select(
            func.sum(func.coalesce(Document.file_size, 0)),
            func.sum(func.coalesce(func.length(Document.content), 0))
        ).where(and_(*base_conditions))
        size_result = await db.execute(size_query)
        file_size, content_chars = size_result.one()
        file_size = file_size or 0
        content_chars = content_chars or 0

        # Word count estimation (rough: chars / 5)
        estimated_words = content_chars // 5

        # Documents by source type
        source_type_query = select(
            DocumentSource.source_type,
            func.count(Document.id)
        ).join(DocumentSource).where(
            and_(*base_conditions)
        ).group_by(DocumentSource.source_type)
        source_type_result = await db.execute(source_type_query)
        docs_by_source_type = dict(source_type_result.fetchall())

        # Documents by file type
        file_type_query = select(
            Document.file_type,
            func.count(Document.id)
        ).where(and_(*base_conditions)).group_by(Document.file_type)
        file_type_result = await db.execute(file_type_query)
        docs_by_file_type = {k or "unknown": v for k, v in file_type_result.fetchall()}

        # Documents over time (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        timeline_query = select(
            func.date(Document.created_at).label('date'),
            func.count(Document.id)
        ).where(
            and_(*base_conditions, Document.created_at >= thirty_days_ago)
        ).group_by(func.date(Document.created_at)).order_by(func.date(Document.created_at))
        timeline_result = await db.execute(timeline_query)
        timeline_data = [(str(row[0]), row[1]) for row in timeline_result.fetchall()]

        # Top tags
        tag_query = select(Document.tags).where(
            and_(*base_conditions, Document.tags != None)
        )
        tag_result = await db.execute(tag_query)
        all_tags = []
        for row in tag_result.fetchall():
            if row[0]:
                all_tags.extend(row[0])
        tag_counts = Counter(all_tags).most_common(20)

        # Top authors
        author_query = select(
            Document.author,
            func.count(Document.id)
        ).where(
            and_(*base_conditions, Document.author != None)
        ).group_by(Document.author).order_by(desc(func.count(Document.id))).limit(10)
        author_result = await db.execute(author_query)
        top_authors = [(row[0], row[1]) for row in author_result.fetchall()]

        # Chunk statistics
        chunk_query = select(
            func.count(DocumentChunk.id),
            func.avg(func.length(DocumentChunk.content))
        ).join(Document).where(and_(*base_conditions))
        chunk_result = await db.execute(chunk_query)
        total_chunks, avg_chunk_size = chunk_result.one()
        total_chunks = total_chunks or 0
        avg_chunk_size = int(avg_chunk_size or 0)

        # Processing status
        processed_query = select(func.count(Document.id)).where(
            and_(Document.is_processed == True)
        )
        pending_query = select(func.count(Document.id)).where(
            and_(Document.is_processed == False)
        )
        processed_count = (await db.execute(processed_query)).scalar() or 0
        pending_count = (await db.execute(pending_query)).scalar() or 0

        # Summarization status
        summarized_query = select(func.count(Document.id)).where(
            and_(*base_conditions, Document.summary != None)
        )
        summarized_count = (await db.execute(summarized_query)).scalar() or 0

        return {
            "total_documents": total_docs,
            "total_file_size_bytes": file_size,
            "total_content_chars": content_chars,
            "estimated_word_count": estimated_words,
            "total_chunks": total_chunks,
            "avg_chunk_size_chars": avg_chunk_size,
            "documents_by_source_type": docs_by_source_type,
            "documents_by_file_type": docs_by_file_type,
            "timeline_last_30_days": timeline_data,
            "top_tags": tag_counts,
            "top_authors": top_authors,
            "processing_status": {
                "processed": processed_count,
                "pending": pending_count
            },
            "summarized_documents": summarized_count,
            "filters_applied": {
                "source_id": str(source_id) if source_id else None,
                "tag": tag,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None
            }
        }

    async def get_source_analytics(
        self,
        db: AsyncSession,
        source_id: Optional[UUID] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get analytics for document sources.

        Args:
            db: Database session
            source_id: Optional specific source to analyze

        Returns:
            List of source analytics
        """
        # Build query for source statistics
        base_query = select(
            DocumentSource.id,
            DocumentSource.name,
            DocumentSource.source_type,
            DocumentSource.is_active,
            DocumentSource.last_sync,
            DocumentSource.created_at,
            func.count(Document.id).label('doc_count'),
            func.sum(func.coalesce(Document.file_size, 0)).label('total_size'),
            func.sum(func.coalesce(func.length(Document.content), 0)).label('total_chars'),
            func.max(Document.updated_at).label('last_doc_update'),
            func.count(Document.id).filter(Document.is_processed == True).label('processed_count'),
            func.count(Document.id).filter(Document.summary != None).label('summarized_count'),
        ).outerjoin(Document).group_by(
            DocumentSource.id,
            DocumentSource.name,
            DocumentSource.source_type,
            DocumentSource.is_active,
            DocumentSource.last_sync,
            DocumentSource.created_at
        )

        if source_id:
            base_query = base_query.where(DocumentSource.id == source_id)

        result = await db.execute(base_query)
        rows = result.fetchall()

        sources = []
        for row in rows:
            doc_count = row.doc_count or 0
            processed_count = row.processed_count or 0
            summarized_count = row.summarized_count or 0

            sources.append({
                "id": str(row.id),
                "name": row.name,
                "source_type": row.source_type,
                "is_active": row.is_active,
                "last_sync": row.last_sync.isoformat() if row.last_sync else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "document_count": doc_count,
                "total_size_bytes": row.total_size or 0,
                "total_chars": row.total_chars or 0,
                "last_document_update": row.last_doc_update.isoformat() if row.last_doc_update else None,
                "processing_rate": f"{(processed_count / doc_count * 100):.1f}%" if doc_count > 0 else "N/A",
                "summarization_rate": f"{(summarized_count / doc_count * 100):.1f}%" if doc_count > 0 else "N/A",
                "health_status": "healthy" if row.is_active else "inactive"
            })

        return sources

    async def get_trending_topics(
        self,
        db: AsyncSession,
        days: int = 7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find trending topics based on recent document content.

        Args:
            db: Database session
            days: Number of days to look back
            limit: Maximum topics to return

        Returns:
            List of trending topics with counts
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get recent tags
        tag_query = select(Document.tags).where(
            and_(
                Document.created_at >= cutoff_date,
                Document.tags != None
            )
        )
        tag_result = await db.execute(tag_query)

        recent_tags = []
        for row in tag_result.fetchall():
            if row[0]:
                recent_tags.extend(row[0])

        # Count tag frequency
        tag_counts = Counter(recent_tags)

        # Get older tag counts for comparison
        older_query = select(Document.tags).where(
            and_(
                Document.created_at < cutoff_date,
                Document.created_at >= cutoff_date - timedelta(days=days),
                Document.tags != None
            )
        )
        older_result = await db.execute(older_query)

        older_tags = []
        for row in older_result.fetchall():
            if row[0]:
                older_tags.extend(row[0])
        older_counts = Counter(older_tags)

        # Calculate trend scores
        trending = []
        for tag, current_count in tag_counts.most_common(limit * 2):
            old_count = older_counts.get(tag, 0)
            if old_count > 0:
                growth_rate = (current_count - old_count) / old_count
            else:
                growth_rate = float(current_count)  # New topic

            trending.append({
                "topic": tag,
                "current_count": current_count,
                "previous_count": old_count,
                "growth_rate": round(growth_rate, 2),
                "trend": "rising" if growth_rate > 0.2 else "stable" if growth_rate >= -0.2 else "declining"
            })

        # Sort by growth rate
        trending.sort(key=lambda x: x["growth_rate"], reverse=True)

        return trending[:limit]

    async def generate_chart_data(
        self,
        db: AsyncSession,
        chart_type: str,
        metric: str,
        group_by: str = "source_type",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate data for various chart types.

        Args:
            db: Database session
            chart_type: Type of chart (bar, line, pie, area)
            metric: Metric to visualize (document_count, file_size, word_count)
            group_by: Grouping field (source_type, file_type, author, tag, date)
            date_from: Start date filter
            date_to: End date filter
            limit: Maximum data points

        Returns:
            Chart data structure
        """
        base_conditions = [Document.is_processed == True]

        if date_from:
            base_conditions.append(Document.created_at >= date_from)
        if date_to:
            base_conditions.append(Document.created_at <= date_to)

        # Determine metric aggregation
        metric_map = {
            "document_count": func.count(Document.id),
            "file_size": func.sum(func.coalesce(Document.file_size, 0)),
            "content_size": func.sum(func.coalesce(func.length(Document.content), 0)),
        }
        agg_func = metric_map.get(metric, func.count(Document.id))

        # Group by different fields
        if group_by == "source_type":
            query = select(
                DocumentSource.source_type.label('label'),
                agg_func.label('value')
            ).join(DocumentSource).where(
                and_(*base_conditions)
            ).group_by(DocumentSource.source_type).order_by(desc('value')).limit(limit)

        elif group_by == "file_type":
            query = select(
                func.coalesce(Document.file_type, 'unknown').label('label'),
                agg_func.label('value')
            ).where(
                and_(*base_conditions)
            ).group_by(Document.file_type).order_by(desc('value')).limit(limit)

        elif group_by == "author":
            query = select(
                func.coalesce(Document.author, 'Unknown').label('label'),
                agg_func.label('value')
            ).where(
                and_(*base_conditions)
            ).group_by(Document.author).order_by(desc('value')).limit(limit)

        elif group_by == "date":
            # Group by date for time series
            query = select(
                func.date(Document.created_at).label('label'),
                agg_func.label('value')
            ).where(
                and_(*base_conditions)
            ).group_by(func.date(Document.created_at)).order_by('label')

        else:
            raise ValueError(f"Unknown group_by: {group_by}")

        result = await db.execute(query)
        rows = result.fetchall()

        labels = []
        values = []
        for row in rows:
            labels.append(str(row.label) if row.label else "Unknown")
            values.append(int(row.value) if row.value else 0)

        return {
            "chart_type": chart_type,
            "metric": metric,
            "group_by": group_by,
            "labels": labels,
            "values": values,
            "datasets": [{
                "label": metric.replace("_", " ").title(),
                "data": values
            }],
            "options": {
                "responsive": True,
                "title": f"{metric.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}"
            }
        }

    async def export_data(
        self,
        db: AsyncSession,
        format: str = "json",
        source_id: Optional[UUID] = None,
        tag: Optional[str] = None,
        include_content: bool = False,
        include_chunks: bool = False,
        limit: int = 1000,
    ) -> Tuple[str, str, str]:
        """
        Export document data to various formats.

        Args:
            db: Database session
            format: Export format (json, csv, jsonl)
            source_id: Filter by source
            tag: Filter by tag
            include_content: Include full document content
            include_chunks: Include chunk data
            limit: Maximum documents to export

        Returns:
            Tuple of (content, filename, content_type)
        """
        # Build query
        conditions = [Document.is_processed == True]
        if source_id:
            conditions.append(Document.source_id == source_id)

        query = select(Document).where(and_(*conditions)).limit(limit)

        if include_chunks:
            from sqlalchemy.orm import selectinload
            query = query.options(selectinload(Document.chunks))

        result = await db.execute(query)
        documents = result.scalars().all()

        # Filter by tag if specified
        if tag:
            documents = [d for d in documents if d.tags and tag in d.tags]

        # Build export data
        export_rows = []
        for doc in documents:
            row = {
                "id": str(doc.id),
                "title": doc.title,
                "author": doc.author,
                "source_type": doc.source.source_type if doc.source else None,
                "source_name": doc.source.name if doc.source else None,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "tags": doc.tags,
                "url": doc.url,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "has_summary": doc.summary is not None,
                "is_processed": doc.is_processed,
            }

            if include_content:
                row["content"] = doc.content
                row["summary"] = doc.summary

            if include_chunks and hasattr(doc, 'chunks'):
                row["chunks"] = [
                    {
                        "index": c.chunk_index,
                        "content": c.content,
                        "metadata": c.metadata
                    }
                    for c in doc.chunks
                ]

            export_rows.append(row)

        # Generate output
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            content = json.dumps(export_rows, indent=2, default=str)
            filename = f"documents_export_{timestamp}.json"
            content_type = "application/json"

        elif format == "jsonl":
            lines = [json.dumps(row, default=str) for row in export_rows]
            content = "\n".join(lines)
            filename = f"documents_export_{timestamp}.jsonl"
            content_type = "application/x-ndjson"

        elif format == "csv":
            output = io.StringIO()
            if export_rows:
                # Flatten nested fields for CSV
                flat_rows = []
                for row in export_rows:
                    flat_row = {
                        k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in row.items()
                        if k != "chunks"  # Exclude chunks from CSV
                    }
                    flat_rows.append(flat_row)

                fieldnames = list(flat_rows[0].keys())
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)
            content = output.getvalue()
            filename = f"documents_export_{timestamp}.csv"
            content_type = "text/csv"

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(export_rows)} documents to {format}")
        return content, filename, content_type

    async def get_search_analytics(
        self,
        db: AsyncSession,
        query: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate analytics about search results.

        Args:
            db: Database session
            query: Search query
            results: Search results

        Returns:
            Search analytics
        """
        if not results:
            return {
                "query": query,
                "total_results": 0,
                "relevance_distribution": {},
                "source_distribution": {},
                "suggestions": []
            }

        # Relevance score distribution
        scores = [r.get("relevance_score", 0) for r in results]
        score_buckets = {
            "high (0.8-1.0)": sum(1 for s in scores if s >= 0.8),
            "medium (0.5-0.8)": sum(1 for s in scores if 0.5 <= s < 0.8),
            "low (0-0.5)": sum(1 for s in scores if s < 0.5),
        }

        # Source distribution
        sources = Counter(r.get("source_type", "unknown") for r in results)

        # File type distribution
        file_types = Counter(r.get("file_type", "unknown") for r in results)

        # Related queries suggestion (based on result titles)
        title_words = []
        for r in results[:10]:
            title = r.get("title", "")
            words = title.lower().split()
            title_words.extend([w for w in words if len(w) > 3 and w not in query.lower()])

        suggested_terms = [term for term, _ in Counter(title_words).most_common(5)]

        return {
            "query": query,
            "total_results": len(results),
            "avg_relevance_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "relevance_distribution": score_buckets,
            "source_distribution": dict(sources),
            "file_type_distribution": dict(file_types),
            "suggested_related_terms": suggested_terms,
        }


# Singleton instance
analytics_service = AnalyticsService()
