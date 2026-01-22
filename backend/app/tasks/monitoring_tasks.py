"""
Monitoring and maintenance tasks.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document, DocumentSource
from app.models.chat import ChatSession, ChatMessage
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

# Note: cleanup_old_data has been moved to app.tasks.maintenance_tasks


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


@celery_app.task(name="app.tasks.monitoring_tasks.health_check")
def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check of all services."""
    return _run_async(_async_health_check())


@celery_app.task(name="app.tasks.monitoring_tasks.generate_stats")
def generate_stats() -> Dict[str, Any]:
    """Generate system statistics."""
    return _run_async(_async_generate_stats())


async def _async_health_check() -> Dict[str, Any]:
    """Async implementation of health check."""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "services": {}
    }
    
    # Check database connectivity
    try:
        async with create_celery_session()() as db:
            result = await db.execute(select(func.count(DocumentSource.id)))
            source_count = result.scalar()
            
            health_status["services"]["database"] = {
                "status": "healthy",
                "message": f"Connected successfully, {source_count} sources configured"
            }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "unhealthy"
    
    # Check vector store
    try:
        vector_store = VectorStoreService()
        if not vector_store._initialized:
            await vector_store.initialize()
        
        stats = await vector_store.get_collection_stats()
        health_status["services"]["vector_store"] = {
            "status": "healthy",
            "message": f"ChromaDB operational, {stats.get('total_chunks', 0)} chunks indexed"
        }
    except Exception as e:
        health_status["services"]["vector_store"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "unhealthy"
    
    # Check LLM service
    try:
        llm_service = LLMService()
        is_healthy = await llm_service.health_check()
        
        if is_healthy:
            models = await llm_service.list_available_models()
            health_status["services"]["llm"] = {
                "status": "healthy",
                "message": f"Ollama operational, {len(models)} models available"
            }
        else:
            health_status["services"]["llm"] = {
                "status": "unhealthy",
                "message": "Ollama service unavailable"
            }
            health_status["overall_status"] = "degraded"
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    # Check disk space
    try:
        import shutil
        data_dir = "./data"
        total, used, free = shutil.disk_usage(data_dir)
        
        # Convert to GB
        total_gb = total // (1024**3)
        used_gb = used // (1024**3)
        free_gb = free // (1024**3)
        usage_percent = (used / total) * 100
        
        status = "healthy"
        if usage_percent > 90:
            status = "critical"
            health_status["overall_status"] = "unhealthy"
        elif usage_percent > 80:
            status = "warning"
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"
        
        health_status["services"]["disk_space"] = {
            "status": status,
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "usage_percent": round(usage_percent, 2)
        }
    except Exception as e:
        health_status["services"]["disk_space"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    logger.info(f"Health check completed: {health_status['overall_status']}")
    return health_status


async def _async_generate_stats() -> Dict[str, Any]:
    """Generate comprehensive system statistics."""
    async with create_celery_session()() as db:
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "documents": {},
                "chat": {},
                "sources": {},
                "processing": {}
            }
            
            # Document statistics
            total_docs_result = await db.execute(select(func.count(Document.id)))
            total_docs = total_docs_result.scalar()
            
            processed_docs_result = await db.execute(
                select(func.count(Document.id)).where(Document.is_processed == True)
            )
            processed_docs = processed_docs_result.scalar()
            
            failed_docs_result = await db.execute(
                select(func.count(Document.id)).where(
                    and_(Document.is_processed == False, Document.processing_error.isnot(None))
                )
            )
            failed_docs = failed_docs_result.scalar()
            
            stats["documents"] = {
                "total": total_docs,
                "processed": processed_docs,
                "failed": failed_docs,
                "pending": total_docs - processed_docs - failed_docs,
                "success_rate": round((processed_docs / total_docs * 100) if total_docs > 0 else 0, 2)
            }

            # Documents without summary (processed but missing or empty summary)
            try:
                without_summary_result = await db.execute(
                    select(func.count(Document.id)).where(
                        and_(
                            Document.is_processed == True,
                            (Document.summary.is_(None)) | (Document.summary == "")
                        )
                    )
                )
                stats["documents"]["without_summary"] = int(without_summary_result.scalar() or 0)
            except Exception:
                stats["documents"]["without_summary"] = 0
            
            # Chat statistics
            total_sessions_result = await db.execute(select(func.count(ChatSession.id)))
            total_sessions = total_sessions_result.scalar()
            
            total_messages_result = await db.execute(select(func.count(ChatMessage.id)))
            total_messages = total_messages_result.scalar()
            
            # Active sessions (with messages in last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            active_sessions_result = await db.execute(
                select(func.count(ChatSession.id.distinct())).where(
                    ChatSession.last_message_at >= yesterday
                )
            )
            active_sessions = active_sessions_result.scalar()
            
            stats["chat"] = {
                "total_sessions": total_sessions,
                "active_sessions_24h": active_sessions,
                "total_messages": total_messages,
                "avg_messages_per_session": round(
                    (total_messages / total_sessions) if total_sessions > 0 else 0, 2
                )
            }
            
            # Source statistics
            sources_by_type_result = await db.execute(
                select(DocumentSource.source_type, func.count(DocumentSource.id))
                .group_by(DocumentSource.source_type)
            )
            sources_by_type = dict(sources_by_type_result.all())
            
            active_sources_result = await db.execute(
                select(func.count(DocumentSource.id)).where(DocumentSource.is_active == True)
            )
            active_sources = active_sources_result.scalar()
            
            stats["sources"] = {
                "total": sum(sources_by_type.values()),
                "active": active_sources,
                "by_type": sources_by_type
            }
            
            # Vector store statistics
            try:
                vector_store = VectorStoreService()
                if not vector_store._initialized:
                    await vector_store.initialize()
                
                vector_stats = await vector_store.get_collection_stats()
                stats["vector_store"] = vector_stats
            except Exception as e:
                stats["vector_store"] = {"error": str(e)}
            
            # Processing statistics (documents by date)
            last_week = datetime.utcnow() - timedelta(days=7)
            recent_docs_result = await db.execute(
                select(
                    func.date(Document.created_at).label('date'),
                    func.count(Document.id).label('count')
                )
                .where(Document.created_at >= last_week)
                .group_by(func.date(Document.created_at))
                .order_by(func.date(Document.created_at))
            )
            recent_docs = [{"date": str(row.date), "count": row.count} 
                          for row in recent_docs_result.all()]
            
            stats["processing"] = {
                "documents_last_7_days": recent_docs,
                "total_documents_last_7_days": sum(item["count"] for item in recent_docs)
            }
            
            logger.info("System statistics generated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating system statistics: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
