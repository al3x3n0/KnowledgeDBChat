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
from app.core.database import AsyncSessionLocal
from app.models.document import Document, DocumentSource
from app.services.document_service import DocumentService
from app.services.connectors.gitlab_connector import GitLabConnector
from app.services.connectors.confluence_connector import ConfluenceConnector
from app.services.connectors.web_connector import WebConnector


@celery_app.task(bind=True, name="app.tasks.ingestion_tasks.ingest_from_source")
def ingest_from_source(self, source_id: str) -> Dict[str, Any]:
    """
    Ingest documents from a specific data source.
    
    Args:
        source_id: UUID of the document source
        
    Returns:
        Dict with ingestion results
    """
    return asyncio.run(_async_ingest_from_source(self, source_id))


async def _async_ingest_from_source(task, source_id: str) -> Dict[str, Any]:
    """Async implementation of source ingestion."""
    async with AsyncSessionLocal() as db:
        try:
            logger.info(f"Starting ingestion from source {source_id}")
            
            # Get the document source
            source = await db.get(DocumentSource, UUID(source_id))
            if not source:
                raise ValueError(f"Document source {source_id} not found")
            
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
            
            # Get document list
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 0,
                    "total": 0,
                    "status": "Fetching document list..."
                }
            )
            
            documents_info = await connector.list_documents()
            total_docs = len(documents_info)
            
            logger.info(f"Found {total_docs} documents in source {source.name}")
            
            # Process documents
            processed = 0
            created = 0
            updated = 0
            errors = 0
            
            for i, doc_info in enumerate(documents_info):
                try:
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
                            await _update_document(db, existing_doc, doc_info, content)
                            updated += 1
                            logger.info(f"Updated document: {doc_info['title']}")
                        else:
                            logger.debug(f"No changes in document: {doc_info['title']}")
                    else:
                        # Create new document
                        await _create_document(db, source, doc_info, content)
                        created += 1
                        logger.info(f"Created document: {doc_info['title']}")
                    
                    processed += 1
                    
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
            return result
            
        except Exception as e:
            logger.error(f"Error in ingestion task for source {source_id}: {e}")
            return {
                "source_id": source_id,
                "error": str(e),
                "success": False
            }


@celery_app.task(bind=True, name="app.tasks.ingestion_tasks.process_uploaded_document")
def process_uploaded_document(self, document_id: str) -> Dict[str, Any]:
    """
    Process an uploaded document for indexing.
    
    Args:
        document_id: UUID of the document to process
        
    Returns:
        Dict with processing results
    """
    return asyncio.run(_async_process_uploaded_document(self, document_id))


async def _async_process_uploaded_document(task, document_id: str) -> Dict[str, Any]:
    """Async implementation of document processing."""
    async with AsyncSessionLocal() as db:
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
    elif source.source_type == "confluence":
        return ConfluenceConnector()
    elif source.source_type == "web":
        return WebConnector()
    else:
        return None


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
    
    document = Document(
        title=doc_info.get("title", "Untitled"),
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
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
    await db.commit()
    await db.refresh(document)
    
    # Trigger processing
    process_uploaded_document.delay(str(document.id))


