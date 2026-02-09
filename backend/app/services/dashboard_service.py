"""
Dashboard service for aggregating data for frontend dashboard widgets.

Provides optimized queries and pre-formatted data for:
- Overview statistics
- Activity feeds
- Charts and visualizations
- Health monitoring
- Trending content
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID
from collections import Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_, case
from loguru import logger

from app.models.document import Document, DocumentChunk, DocumentSource
from app.models.memory import AgentConversation
from app.models.agent_definition import AgentDefinition, AgentConversationContext
from app.models.workflow import Workflow, WorkflowExecution
from app.core.config import settings


class DashboardService:
    """Service for dashboard data aggregation."""

    async def get_overview_stats(
        self,
        db: AsyncSession,
        user_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Get overview statistics for the main dashboard.

        Returns counts and key metrics for the knowledge base.
        """
        # Document counts
        doc_query = select(
            func.count(Document.id).label('total'),
            func.count(Document.id).filter(Document.is_processed == True).label('processed'),
            func.count(Document.id).filter(Document.is_processed == False).label('pending'),
            func.count(Document.id).filter(Document.summary != None).label('summarized'),
            func.sum(func.coalesce(Document.file_size, 0)).label('total_size'),
        )
        doc_result = await db.execute(doc_query)
        doc_stats = doc_result.one()

        # Source counts
        source_query = select(
            func.count(DocumentSource.id).label('total'),
            func.count(DocumentSource.id).filter(DocumentSource.is_active == True).label('active'),
        )
        source_result = await db.execute(source_query)
        source_stats = source_result.one()

        # Chunk count
        chunk_query = select(func.count(DocumentChunk.id))
        chunk_result = await db.execute(chunk_query)
        chunk_count = chunk_result.scalar() or 0

        # Recent activity (last 24h)
        yesterday = datetime.utcnow() - timedelta(hours=24)
        recent_query = select(func.count(Document.id)).where(
            Document.created_at >= yesterday
        )
        recent_result = await db.execute(recent_query)
        recent_docs = recent_result.scalar() or 0

        # Workflow stats
        workflow_query = select(
            func.count(Workflow.id).label('total'),
            func.count(Workflow.id).filter(Workflow.is_active == True).label('active'),
        )
        if user_id:
            workflow_query = workflow_query.where(Workflow.user_id == user_id)
        workflow_result = await db.execute(workflow_query)
        workflow_stats = workflow_result.one()

        # Agent stats
        agent_query = select(
            func.count(AgentDefinition.id).label('total'),
            func.count(AgentDefinition.id).filter(AgentDefinition.is_active == True).label('active'),
            func.count(AgentDefinition.id).filter(AgentDefinition.is_system == True).label('system'),
        )
        agent_result = await db.execute(agent_query)
        agent_stats = agent_result.one()

        return {
            "documents": {
                "total": doc_stats.total or 0,
                "processed": doc_stats.processed or 0,
                "pending": doc_stats.pending or 0,
                "summarized": doc_stats.summarized or 0,
                "total_size_bytes": doc_stats.total_size or 0,
                "total_size_mb": round((doc_stats.total_size or 0) / (1024 * 1024), 2),
                "added_last_24h": recent_docs,
            },
            "chunks": {
                "total": chunk_count,
            },
            "sources": {
                "total": source_stats.total or 0,
                "active": source_stats.active or 0,
            },
            "workflows": {
                "total": workflow_stats.total or 0,
                "active": workflow_stats.active or 0,
            },
            "agents": {
                "total": agent_stats.total or 0,
                "active": agent_stats.active or 0,
                "system": agent_stats.system or 0,
                "custom": (agent_stats.total or 0) - (agent_stats.system or 0),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def get_activity_feed(
        self,
        db: AsyncSession,
        user_id: Optional[UUID] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get recent activity feed for the dashboard.

        Combines document uploads, workflow executions, and agent conversations.
        """
        activities = []

        # Recent documents
        doc_query = select(Document).order_by(desc(Document.created_at)).limit(limit)
        doc_result = await db.execute(doc_query)
        for doc in doc_result.scalars():
            activities.append({
                "type": "document_added",
                "title": f"Document added: {doc.title}",
                "description": f"New document from {doc.source.name if doc.source else 'upload'}",
                "timestamp": doc.created_at.isoformat() if doc.created_at else None,
                "icon": "document",
                "metadata": {
                    "document_id": str(doc.id),
                    "file_type": doc.file_type,
                }
            })

        # Recent workflow executions
        if user_id:
            exec_query = select(WorkflowExecution).join(Workflow).where(
                Workflow.user_id == user_id
            ).order_by(desc(WorkflowExecution.created_at)).limit(limit)
        else:
            exec_query = select(WorkflowExecution).order_by(
                desc(WorkflowExecution.created_at)
            ).limit(limit)

        exec_result = await db.execute(exec_query)
        for execution in exec_result.scalars():
            status_icon = {
                "completed": "check",
                "failed": "error",
                "running": "sync",
                "pending": "clock"
            }.get(execution.status, "workflow")

            activities.append({
                "type": f"workflow_{execution.status}",
                "title": f"Workflow {execution.status}",
                "description": f"Execution completed" if execution.status == "completed" else f"Status: {execution.status}",
                "timestamp": execution.updated_at.isoformat() if execution.updated_at else execution.created_at.isoformat() if execution.created_at else None,
                "icon": status_icon,
                "metadata": {
                    "execution_id": str(execution.id),
                    "workflow_id": str(execution.workflow_id),
                    "status": execution.status,
                }
            })

        # Sort by timestamp and return top items
        activities.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        return activities[:limit]

    async def get_documents_by_type_chart(
        self,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Get document distribution by file type for pie/donut chart."""
        query = select(
            func.coalesce(Document.file_type, 'unknown').label('type'),
            func.count(Document.id).label('count')
        ).where(
            Document.is_processed == True
        ).group_by(Document.file_type).order_by(desc(func.count(Document.id)))

        result = await db.execute(query)
        rows = result.fetchall()

        labels = []
        values = []
        colors = {
            'application/pdf': '#FF6384',
            'text/plain': '#36A2EB',
            'text/markdown': '#FFCE56',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '#4BC0C0',
            'text/html': '#9966FF',
            'application/json': '#FF9F40',
            'unknown': '#C9CBCF',
        }

        for row in rows:
            file_type = row.type or 'unknown'
            # Simplify type names for display
            display_name = file_type.split('/')[-1].split('.')[-1].upper()
            if len(display_name) > 10:
                display_name = display_name[:10]

            labels.append(display_name)
            values.append(row.count)

        return {
            "chart_type": "doughnut",
            "title": "Documents by Type",
            "labels": labels,
            "datasets": [{
                "data": values,
                "backgroundColor": [colors.get(row.type, '#C9CBCF') for row in rows],
            }],
        }

    async def get_documents_by_source_chart(
        self,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Get document distribution by source for bar chart."""
        query = select(
            DocumentSource.name,
            DocumentSource.source_type,
            func.count(Document.id).label('count')
        ).join(Document, isouter=True).group_by(
            DocumentSource.id, DocumentSource.name, DocumentSource.source_type
        ).order_by(desc(func.count(Document.id))).limit(10)

        result = await db.execute(query)
        rows = result.fetchall()

        return {
            "chart_type": "bar",
            "title": "Documents by Source",
            "labels": [row.name for row in rows],
            "datasets": [{
                "label": "Documents",
                "data": [row.count for row in rows],
                "backgroundColor": "#4BC0C0",
            }],
        }

    async def get_documents_timeline_chart(
        self,
        db: AsyncSession,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get document creation timeline for line chart."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = select(
            func.date(Document.created_at).label('date'),
            func.count(Document.id).label('count')
        ).where(
            Document.created_at >= cutoff
        ).group_by(func.date(Document.created_at)).order_by('date')

        result = await db.execute(query)
        rows = result.fetchall()

        # Fill in missing dates
        date_counts = {str(row.date): row.count for row in rows}
        all_dates = []
        all_counts = []

        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days - i - 1)).date()
            date_str = str(date)
            all_dates.append(date_str)
            all_counts.append(date_counts.get(date_str, 0))

        return {
            "chart_type": "line",
            "title": f"Documents Added (Last {days} Days)",
            "labels": all_dates,
            "datasets": [{
                "label": "Documents",
                "data": all_counts,
                "borderColor": "#36A2EB",
                "fill": True,
                "backgroundColor": "rgba(54, 162, 235, 0.1)",
            }],
        }

    async def get_source_health(
        self,
        db: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """Get health status for all document sources."""
        query = select(
            DocumentSource.id,
            DocumentSource.name,
            DocumentSource.source_type,
            DocumentSource.is_active,
            DocumentSource.last_sync,
            func.count(Document.id).label('doc_count'),
            func.count(Document.id).filter(Document.is_processed == True).label('processed'),
            func.count(Document.id).filter(Document.processing_error != None).label('errors'),
        ).outerjoin(Document).group_by(
            DocumentSource.id,
            DocumentSource.name,
            DocumentSource.source_type,
            DocumentSource.is_active,
            DocumentSource.last_sync
        ).order_by(DocumentSource.name)

        result = await db.execute(query)
        rows = result.fetchall()

        sources = []
        for row in rows:
            # Determine health status
            if not row.is_active:
                status = "inactive"
                status_color = "gray"
            elif row.errors > 0:
                status = "warning"
                status_color = "yellow"
            elif row.last_sync and (datetime.utcnow() - row.last_sync).days > 7:
                status = "stale"
                status_color = "orange"
            else:
                status = "healthy"
                status_color = "green"

            sources.append({
                "id": str(row.id),
                "name": row.name,
                "type": row.source_type,
                "is_active": row.is_active,
                "last_sync": row.last_sync.isoformat() if row.last_sync else None,
                "document_count": row.doc_count or 0,
                "processed_count": row.processed or 0,
                "error_count": row.errors or 0,
                "status": status,
                "status_color": status_color,
            })

        return sources

    async def get_trending_tags(
        self,
        db: AsyncSession,
        days: int = 7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get trending tags from recent documents."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = select(Document.tags).where(
            and_(
                Document.created_at >= cutoff,
                Document.tags != None
            )
        )
        result = await db.execute(query)

        all_tags = []
        for row in result.fetchall():
            if row.tags:
                all_tags.extend(row.tags)

        tag_counts = Counter(all_tags).most_common(limit)

        return [
            {"tag": tag, "count": count, "trend": "up"}
            for tag, count in tag_counts
        ]

    async def get_quick_actions(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> List[Dict[str, Any]]:
        """Get suggested quick actions based on current state."""
        actions = []

        # Check for pending documents
        pending_query = select(func.count(Document.id)).where(
            Document.is_processed == False
        )
        pending_result = await db.execute(pending_query)
        pending_count = pending_result.scalar() or 0

        if pending_count > 0:
            actions.append({
                "id": "process_pending",
                "title": f"Process {pending_count} pending documents",
                "description": "Documents awaiting processing",
                "icon": "refresh",
                "action_type": "link",
                "action_data": {"url": "/documents?status=pending"},
                "priority": "high",
            })

        # Check for documents without summaries
        unsummarized_query = select(func.count(Document.id)).where(
            and_(
                Document.is_processed == True,
                Document.summary == None
            )
        )
        unsummarized_result = await db.execute(unsummarized_query)
        unsummarized_count = unsummarized_result.scalar() or 0

        if unsummarized_count > 5:
            actions.append({
                "id": "summarize_docs",
                "title": f"Summarize {unsummarized_count} documents",
                "description": "Generate AI summaries for documents",
                "icon": "sparkles",
                "action_type": "workflow",
                "action_data": {"template": "batch_summarize"},
                "priority": "medium",
            })

        # Check for inactive sources
        inactive_query = select(func.count(DocumentSource.id)).where(
            DocumentSource.is_active == False
        )
        inactive_result = await db.execute(inactive_query)
        inactive_count = inactive_result.scalar() or 0

        if inactive_count > 0:
            actions.append({
                "id": "review_sources",
                "title": f"Review {inactive_count} inactive sources",
                "description": "Some data sources are disabled",
                "icon": "alert",
                "action_type": "link",
                "action_data": {"url": "/admin/sources"},
                "priority": "low",
            })

        # Always show upload action
        actions.append({
            "id": "upload_docs",
            "title": "Upload documents",
            "description": "Add new documents to the knowledge base",
            "icon": "upload",
            "action_type": "modal",
            "action_data": {"modal": "upload"},
            "priority": "normal",
        })

        return actions

    async def get_agent_usage_summary(
        self,
        db: AsyncSession,
        user_id: Optional[UUID] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Get agent usage summary for the dashboard."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Total conversations
        conv_query = select(func.count(AgentConversation.id)).where(
            AgentConversation.created_at >= cutoff
        )
        if user_id:
            conv_query = conv_query.where(AgentConversation.user_id == user_id)
        conv_result = await db.execute(conv_query)
        total_conversations = conv_result.scalar() or 0

        # Agent usage breakdown
        agent_query = select(
            AgentDefinition.display_name,
            func.count(AgentConversationContext.id).label('turns')
        ).join(AgentConversationContext).where(
            AgentConversationContext.created_at >= cutoff
        ).group_by(AgentDefinition.id, AgentDefinition.display_name).order_by(
            desc(func.count(AgentConversationContext.id))
        ).limit(5)

        agent_result = await db.execute(agent_query)
        top_agents = [
            {"name": row.display_name, "turns": row.turns}
            for row in agent_result.fetchall()
        ]

        return {
            "period_days": days,
            "total_conversations": total_conversations,
            "top_agents": top_agents,
        }

    async def get_workflow_summary(
        self,
        db: AsyncSession,
        user_id: UUID,
        days: int = 7,
    ) -> Dict[str, Any]:
        """Get workflow execution summary."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Execution counts by status
        status_query = select(
            WorkflowExecution.status,
            func.count(WorkflowExecution.id)
        ).join(Workflow).where(
            and_(
                Workflow.user_id == user_id,
                WorkflowExecution.created_at >= cutoff
            )
        ).group_by(WorkflowExecution.status)

        status_result = await db.execute(status_query)
        status_counts = dict(status_result.fetchall())

        # Recent executions
        recent_query = select(WorkflowExecution).join(Workflow).where(
            Workflow.user_id == user_id
        ).order_by(desc(WorkflowExecution.created_at)).limit(5)

        recent_result = await db.execute(recent_query)
        recent_executions = [
            {
                "id": str(ex.id),
                "workflow_id": str(ex.workflow_id),
                "status": ex.status,
                "created_at": ex.created_at.isoformat() if ex.created_at else None,
            }
            for ex in recent_result.scalars()
        ]

        return {
            "period_days": days,
            "status_counts": status_counts,
            "total_executions": sum(status_counts.values()),
            "recent_executions": recent_executions,
        }


# Singleton instance
dashboard_service = DashboardService()
