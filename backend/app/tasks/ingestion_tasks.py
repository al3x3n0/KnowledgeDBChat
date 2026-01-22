"""
Background tasks for document ingestion and processing.
"""

import asyncio
from typing import List, Dict, Any, Optional
from uuid import UUID
from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document, DocumentSource, DocumentSourceSyncLog
from app.services.document_service import DocumentService
from app.services.persona_service import persona_service
from app.services.connectors.gitlab_connector import GitLabConnector
from app.services.connectors.github_connector import GitHubConnector
from app.services.connectors.confluence_connector import ConfluenceConnector
from app.services.connectors.web_connector import WebConnector
from app.services.connectors.arxiv_connector import ArxivConnector
import json
import redis
from app.core.config import settings


def _run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        # Fallback for unexpected contexts
        return asyncio.run(coroutine)
    return loop.run_until_complete(coroutine)


@celery_app.task(bind=True, name="app.tasks.ingestion_tasks.ingest_from_source")
def ingest_from_source(self, source_id: str) -> Dict[str, Any]:
    """
    Ingest documents from a specific data source.
    
    Args:
        source_id: UUID of the document source
        
    Returns:
        Dict with ingestion results
    """
    return _run_async(_async_ingest_from_source(self, source_id))


async def _async_ingest_from_source(task, source_id: str) -> Dict[str, Any]:
    """Async implementation of source ingestion."""
    async with create_celery_session()() as db:
        try:
            logger.info(f"Starting ingestion from source {source_id}")
            
            # Get the document source
            source = await db.get(DocumentSource, UUID(source_id))
            if not source:
                raise ValueError(f"Document source {source_id} not found")
            _publish_ing_status(source_id, {"is_syncing": True})
            # Persist is_syncing flag
            try:
                source.is_syncing = True
                source.last_error = None
                await db.commit()
            except Exception:
                pass
            # Create sync log
            sync_log = DocumentSourceSyncLog(
                source_id=source.id,
                task_id=current_task.request.id if current_task else None,
                status='running'
            )
            try:
                db.add(sync_log)
                await db.commit()
                await db.refresh(sync_log)
            except Exception:
                sync_log = None
            
            # Update task state
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 0,
                    "total": 0,
                    "status": f"Connecting to {source.source_type} source: {source.name}"
                }
            )
            
            # Get appropriate connector
            connector = _get_connector(source)
            if not connector:
                raise ValueError(f"No connector available for source type: {source.source_type}")
            
            # Initialize connector
            await connector.initialize(source.config)
            
            # Get document list (incremental when possible)
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 0,
                    "total": 0,
                    "status": "Fetching document list..."
                }
            )
            
            from datetime import datetime, timedelta
            # Determine incremental vs full based on force_full flag and last_sync
            since = None
            try:
                rc = _get_redis_client()
                force_full = bool(rc.get(f"ingestion:force_full:{source_id}")) if rc else False
            except Exception:
                force_full = False
            if not force_full:
                try:
                    since = source.last_sync if source.last_sync else None
                except Exception:
                    since = None
            # Clear force_full flag if set
            try:
                rc = _get_redis_client()
                if rc:
                    rc.delete(f"ingestion:force_full:{source_id}")
            except Exception:
                pass

            if since:
                try:
                    documents_info = await connector.list_changed_documents(since)
                except Exception:
                    documents_info = await connector.list_documents()
            else:
                documents_info = await connector.list_documents()
            total_docs = len(documents_info)
            
            logger.info(f"Found {total_docs} documents in source {source.name}")
            _publish_ing_progress(source_id, {"stage": "listing", "total": total_docs, "current": 0, "progress": 0, "status": "Fetched list"})
            
            # Process documents
            import time
            start_time = time.time()
            processed = 0
            created = 0
            updated = 0
            errors = 0
            
            for i, doc_info in enumerate(documents_info):
                try:
                    # Cancellation check
                    try:
                        rc = _get_redis_client()
                        if rc and rc.get(f"ingestion:cancel:{source_id}"):
                            _publish_ing_status(source_id, {"is_syncing": False, "canceled": True})
                            # persist state
                            try:
                                src = await db.get(DocumentSource, UUID(source_id))
                                if src:
                                    src.is_syncing = False
                                    await db.commit()
                            except Exception:
                                pass
                            # update sync log
                            try:
                                if sync_log:
                                    from datetime import datetime as _dt
                                    sync_log.status = 'canceled'
                                    sync_log.finished_at = _dt.utcnow()
                                    sync_log.total_documents = total_docs
                                    sync_log.processed = processed
                                    sync_log.created = created
                                    sync_log.updated = updated
                                    sync_log.errors = errors
                                    await db.commit()
                            except Exception:
                                pass
                            return {
                                "source_id": source_id,
                                "source_name": source.name,
                                "total_documents": total_docs,
                                "processed": processed,
                                "created": created,
                                "updated": updated,
                                "errors": errors,
                                "success": False,
                                "canceled": True,
                            }
                    except Exception:
                        pass
                    task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": i + 1,
                            "total": total_docs,
                            "status": f"Processing: {doc_info.get('title', 'Unknown')}"
                        }
                    )
                    
                    # Check if document exists
                    existing_doc = await _find_existing_document(
                        db, source, doc_info["identifier"]
                    )
                    
                    # Get document content
                    content = await connector.get_document_content(doc_info["identifier"])
                    
                    if existing_doc:
                        # Check if content has changed
                        if await _content_changed(existing_doc, content):
                            await _update_document(db, source, existing_doc, doc_info, content)
                            updated += 1
                            logger.info(f"Updated document: {doc_info['title']}")
                        else:
                            logger.debug(f"No changes in document: {doc_info['title']}")
                    else:
                        # Create new document (skip duplicates by hash)
                        _, created_flag = await _create_document(db, source, doc_info, content)
                        if created_flag:
                            created += 1
                            logger.info(f"Created document: {doc_info['title']}")
                        else:
                            logger.info(f"Skipped duplicate document: {doc_info['title']}")
                    
                    processed += 1
                    pct = int((processed / max(1, total_docs)) * 100)
                    elapsed = max(0.0, time.time() - start_time)
                    remaining_seconds = None
                    if processed > 0 and total_docs:
                        rate = processed / elapsed if elapsed > 0 else None
                        if rate and rate > 0:
                            remaining = max(0, total_docs - processed)
                            remaining_seconds = int(remaining / rate)
                    _publish_ing_progress(source_id, {
                        "stage": "processing",
                        "total": total_docs,
                        "current": processed,
                        "progress": pct,
                        "status": doc_info.get('title', 'Processing'),
                        "elapsed": int(elapsed),
                        "remaining_seconds": remaining_seconds,
                        "remaining_formatted": _format_seconds(remaining_seconds) if remaining_seconds is not None else None,
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_info.get('title', 'Unknown')}: {e}")
                    errors += 1
            
            # Update source sync timestamp
            from datetime import datetime
            source.last_sync = datetime.utcnow()
            await db.commit()
            
            result = {
                "source_id": source_id,
                "source_name": source.name,
                "total_documents": total_docs,
                "processed": processed,
                "created": created,
                "updated": updated,
                "errors": errors,
                "success": True
            }
            
            logger.info(f"Ingestion completed for source {source.name}: {result}")
            _publish_ing_complete(source_id, result)
            _publish_ing_status(source_id, {"is_syncing": False})
            try:
                source.is_syncing = False
                await db.commit()
            except Exception:
                pass
            # Update sync log
            try:
                if sync_log:
                    sync_log.status = 'success'
                    from datetime import datetime as _dt
                    sync_log.finished_at = _dt.utcnow()
                    sync_log.total_documents = total_docs
                    sync_log.processed = processed
                    sync_log.created = created
                    sync_log.updated = updated
                    sync_log.errors = errors
                    await db.commit()
            except Exception:
                pass
            # Cleanup mapping and cancel flag
            try:
                client = _get_redis_client()
                if client:
                    client.delete(f"ingestion:task:{source_id}")
                    client.delete(f"ingestion:cancel:{source_id}")
            except Exception:
                pass
            return result
            
        except Exception as e:
            logger.error(f"Error in ingestion task for source {source_id}: {e}")
            _publish_ing_error(source_id, str(e))
            _publish_ing_status(source_id, {"is_syncing": False, "failed": True})
            try:
                source = await db.get(DocumentSource, UUID(source_id))
                if source:
                    source.is_syncing = False
                    source.last_error = str(e)
                    await db.commit()
            except Exception:
                pass
            # Update failure in sync log
            try:
                if sync_log:
                    from datetime import datetime as _dt
                    sync_log.status = 'failed'
                    sync_log.finished_at = _dt.utcnow()
                    sync_log.error_message = str(e)
                    await db.commit()
            except Exception:
                pass
            return {
                "source_id": source_id,
                "error": str(e),
                "success": False
            }

        finally:
            # Ensure mapping is cleared on any exit
            try:
                client = _get_redis_client()
                if client:
                    client.delete(f"ingestion:task:{source_id}")
            except Exception:
                pass


def _get_redis_client():
    try:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for ingestion progress: {e}")
        return None


def _publish_ing_progress(source_id: str, progress: dict):
    try:
        client = _get_redis_client()
        if client:
            channel = f"ingestion_progress:{source_id}"
            msg = json.dumps({
                "type": "progress",
                "document_id": source_id,
                "progress": progress,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish ingestion progress: {e}")


def _publish_ing_complete(source_id: str, result: dict):
    try:
        client = _get_redis_client()
        if client:
            channel = f"ingestion_progress:{source_id}"
            msg = json.dumps({
                "type": "complete",
                "document_id": source_id,
                "result": result,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish ingestion complete: {e}")


def _publish_ing_error(source_id: str, error: str):
    try:
        client = _get_redis_client()
        if client:
            channel = f"ingestion_progress:{source_id}"
            msg = json.dumps({
                "type": "error",
                "document_id": source_id,
                "error": error,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish ingestion error: {e}")


def _publish_ing_status(source_id: str, status: dict):
    try:
        client = _get_redis_client()
        if client:
            channel = f"ingestion_progress:{source_id}"
            msg = json.dumps({
                "type": "status",
                "document_id": source_id,
                "status": status,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish ingestion status: {e}")


def _format_seconds(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"
    except Exception:
        return None


@celery_app.task(bind=True, name="app.tasks.ingestion_tasks.process_uploaded_document")
def process_uploaded_document(self, document_id: str) -> Dict[str, Any]:
    """
    Process an uploaded document for indexing.
    
    Args:
        document_id: UUID of the document to process
        
    Returns:
        Dict with processing results
    """
    return _run_async(_async_process_uploaded_document(self, document_id))


async def _async_process_uploaded_document(task, document_id: str) -> Dict[str, Any]:
    """Async implementation of document processing."""
    async with create_celery_session()() as db:
        try:
            logger.info(f"Starting processing of document {document_id}")
            
            # Get the document
            document = await db.get(Document, UUID(document_id))
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 1,
                    "total": 4,
                    "status": f"Processing document: {document.title}"
                }
            )
            
            document_service = DocumentService()
            
            # Process the document
            await document_service._process_document_async(document, db)
            
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 4,
                    "total": 4,
                    "status": "Processing completed"
                }
            )
            
            result = {
                "document_id": document_id,
                "title": document.title,
                "success": True,
                "processed": True
            }
            
            logger.info(f"Document processing completed: {document.title}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            
            # Update document with error
            try:
                document = await db.get(Document, UUID(document_id))
                if document:
                    document.is_processed = False
                    document.processing_error = str(e)
                    await db.commit()
            except Exception as db_error:
                logger.error(f"Error updating document with processing error: {db_error}")
            
            return {
                "document_id": document_id,
                "error": str(e),
                "success": False
            }


def _get_connector(source: DocumentSource):
    """Get the appropriate connector for a document source."""
    if source.source_type == "gitlab":
        return GitLabConnector()
    elif source.source_type == "github":
        return GitHubConnector()
    elif source.source_type == "confluence":
        return ConfluenceConnector()
    elif source.source_type == "web":
        return WebConnector()
    elif source.source_type == "arxiv":
        return ArxivConnector()
    else:
        return None


@celery_app.task(bind=True, name="app.tasks.ingestion_tasks.dry_run_source")
def dry_run_source(self, source_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dry-run ingestion for a source: no writes, just report counts and sample documents.

    Args:
        source_id: the source UUID string
        options: optional overrides like include_files/include_issues/include_merge_requests/include_pull_requests/include_wiki
    """
    return asyncio.run(_async_dry_run_source(self, source_id, options or {}))


async def _async_dry_run_source(task, source_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            source = await db.get(DocumentSource, UUID(source_id))
            if not source:
                raise ValueError("Source not found")
            connector = _get_connector(source)
            if not connector:
                raise ValueError(f"No connector for type {source.source_type}")
            await connector.initialize(source.config)
            # Apply temporary include_* overrides for dry-run if provided
            try:
                for key in [
                    'include_files', 'include_issues', 'include_merge_requests', 'include_pull_requests', 'include_wiki', 'include_wikis'
                ]:
                    if key in options and hasattr(connector, key):
                        setattr(connector, key, bool(options[key]))
                # aliases
                if 'include_prs' in options and hasattr(connector, 'include_pull_requests'):
                    connector.include_pull_requests = bool(options['include_prs'])
                if 'include_mrs' in options and hasattr(connector, 'include_merge_requests'):
                    connector.include_merge_requests = bool(options['include_mrs'])
            except Exception:
                pass
            # Prefer incremental
            since = source.last_sync
            mode = 'full'
            if since:
                try:
                    docs = await connector.list_changed_documents(since)
                    mode = 'incremental'
                except Exception:
                    docs = await connector.list_documents()
            else:
                docs = await connector.list_documents()
            total = len(docs)
            # type counts
            by_type: Dict[str, int] = {}
            for d in docs:
                t = (d.get('metadata') or {}).get('type') or 'unknown'
                by_type[t] = by_type.get(t, 0) + 1
            # Estimate existing vs new by identifier
            from sqlalchemy import select
            existing = 0
            for d in docs[:500]:  # scan sample for speed
                res = await db.execute(
                    select(Document).where(
                        Document.source_id == source.id,
                        Document.source_identifier == d.get('identifier')
                    )
                )
                if res.scalar_one_or_none():
                    existing += 1
            est_existing = int((existing / max(1, min(500, total))) * total) if total > 500 else existing
            est_new = max(0, total - est_existing)
            sample = [{"title": d.get("title"), "identifier": d.get("identifier"), "type": d.get("metadata", {}).get("type")} for d in docs[:20]]
            return {
                "source_id": source_id,
                "source_name": source.name,
                "total": total,
                "mode": mode,
                "by_type": by_type,
                "estimated_existing": est_existing,
                "estimated_new": est_new,
                "sample": sample,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Dry-run failed: {e}")
            return {"success": False, "error": str(e), "source_id": source_id}


async def _find_existing_document(
    db: AsyncSession, 
    source: DocumentSource, 
    identifier: str
) -> Optional[Document]:
    """Find existing document by source and identifier."""
    from sqlalchemy import select
    
    result = await db.execute(
        select(Document).where(
            Document.source_id == source.id,
            Document.source_identifier == identifier
        )
    )
    return result.scalar_one_or_none()


async def _content_changed(document: Document, new_content: str) -> bool:
    """Check if document content has changed."""
    import hashlib
    
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()
    return document.content_hash != new_hash


async def _update_document(
    db: AsyncSession,
    source: DocumentSource,
    document: Document,
    doc_info: Dict[str, Any],
    content: str
):
    """Update existing document with new content."""
    import hashlib
    from datetime import datetime
    
    # Update document fields
    document.content = content
    document.content_hash = hashlib.sha256(content.encode()).hexdigest()
    document.title = doc_info.get("title", document.title)
    document.url = doc_info.get("url", document.url)
    document.author = doc_info.get("author", document.author)
    document.last_modified = doc_info.get("last_modified")
    document.updated_at = datetime.utcnow()
    document.is_processed = False  # Will be processed again
    document.processing_error = None
    
    # Update metadata
    if doc_info.get("metadata"):
        document.extra_metadata = {**(document.extra_metadata or {}), **doc_info["metadata"]}

    await _assign_document_owner(db, document, source, doc_info)
    await db.commit()
    
    # Trigger reprocessing
    process_uploaded_document.delay(str(document.id))


async def _create_document(
    db: AsyncSession,
    source: DocumentSource,
    doc_info: Dict[str, Any],
    content: str
):
    """Create new document from source information."""
    import hashlib
    from sqlalchemy import select

    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Skip duplicates (same repo + hash)
    existing = await db.execute(
        select(Document).where(
            Document.source_id == source.id,
            Document.content_hash == content_hash
        )
    )
    duplicate = existing.scalar_one_or_none()
    if duplicate:
        logger.info(
            f"Skipping duplicate document for source {source.name}: {doc_info.get('title', duplicate.title)}"
        )
        return duplicate, False
    
    document = Document(
        title=doc_info.get("title", "Untitled"),
        content=content,
        content_hash=content_hash,
        url=doc_info.get("url"),
        file_type=doc_info.get("file_type"),
        source_id=source.id,
        source_identifier=doc_info["identifier"],
        author=doc_info.get("author"),
        extra_metadata=doc_info.get("metadata", {}),
        last_modified=doc_info.get("last_modified"),
        is_processed=False
    )
    
    db.add(document)
    await db.flush()
    await _assign_document_owner(db, document, source, doc_info)
    await db.commit()
    await db.refresh(document)
    
    # Trigger processing
    process_uploaded_document.delay(str(document.id))
    return document, True


async def _assign_document_owner(
    db: AsyncSession,
    document: Document,
    source: Optional[DocumentSource],
    doc_info: Dict[str, Any]
) -> None:
    """Assign persona ownership for a document based on source metadata."""
    try:
        metadata = doc_info.get("metadata") or {}
        platform_identifier = metadata.get("author_id") or metadata.get("user_id") or metadata.get("creator_id")
        await persona_service.assign_document_owner(
            db,
            document,
            author_name=doc_info.get("author"),
            platform_scope=(source.source_type if source else None),
            platform_identifier=str(platform_identifier) if platform_identifier else None,
        )
    except Exception as exc:
        logger.debug(f"Persona assignment skipped for document {document.id}: {exc}")
