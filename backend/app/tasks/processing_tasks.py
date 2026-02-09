"""
Background tasks for document processing (text extraction, chunking, embedding generation).
"""

import asyncio
from typing import Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document
from app.services.document_service import DocumentService


@celery_app.task(bind=True, name="app.tasks.processing_tasks.process_document")
def process_document(self, document_id: str) -> Dict[str, Any]:
    """
    Process a document for indexing (text extraction, chunking, embedding generation).
    
    Args:
        document_id: UUID of the document to process
        
    Returns:
        Dict with processing results
    """
    return asyncio.run(_async_process_document(self, document_id))


async def _async_process_document(task, document_id: str) -> Dict[str, Any]:
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
            
            # Step 1: Text extraction (if needed)
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 1,
                    "total": 4,
                    "status": "Extracting text content"
                }
            )
            
            # Step 2: Text chunking
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 2,
                    "total": 4,
                    "status": "Splitting text into chunks"
                }
            )
            
            # Step 3: Generate embeddings and add to vector store
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 3,
                    "total": 4,
                    "status": "Generating embeddings and indexing"
                }
            )
            
            # Process the document (includes all steps above)
            await document_service._process_document_async(document, db)
            
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 4,
                    "total": 4,
                    "status": "Processing completed"
                }
            )
            
            # Refresh document to get updated state
            await db.refresh(document)
            
            result = {
                "document_id": document_id,
                "title": document.title,
                "success": True,
                "processed": document.is_processed,
                "chunks_count": len(document.chunks) if hasattr(document, 'chunks') else 0
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


@celery_app.task(bind=True, name="app.tasks.processing_tasks.reprocess_document")
def reprocess_document(self, document_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Reprocess a document (delete old chunks and re-index).

    Args:
        document_id: UUID of the document to reprocess
        user_id: Optional user ID for LLM settings

    Returns:
        Dict with processing results
    """
    return asyncio.run(_async_reprocess_document(self, document_id, user_id))


async def _async_reprocess_document(task, document_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Async implementation of document reprocessing."""
    async with create_celery_session()() as db:
        try:
            logger.info(f"Starting reprocessing of document {document_id}")

            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 1,
                    "total": 3,
                    "status": "Deleting existing chunks"
                }
            )

            document_service = DocumentService()

            # Reprocess document (deletes old chunks and re-indexes)
            success = await document_service.reprocess_document(UUID(document_id), db, user_id=UUID(user_id) if user_id else None)
            
            if not success:
                raise ValueError(f"Document {document_id} not found or reprocessing failed")
            
            task.update_state(
                state="PROGRESS",
                meta={
                    "current": 3,
                    "total": 3,
                    "status": "Reprocessing completed"
                }
            )
            
            # Get updated document
            document = await document_service.get_document(UUID(document_id), db)
            
            result = {
                "document_id": document_id,
                "title": document.title if document else "Unknown",
                "success": True,
                "processed": document.is_processed if document else False,
                "chunks_count": len(document.chunks) if document and hasattr(document, 'chunks') else 0
            }
            
            logger.info(f"Document reprocessing completed: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {e}")
            
            return {
                "document_id": document_id,
                "error": str(e),
                "success": False
            }

