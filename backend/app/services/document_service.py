"""
Document service for managing documents and document sources.
"""

import os
import hashlib
import tempfile
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload, noload
from loguru import logger

from app.models.document import Document, DocumentChunk, DocumentSource
from app.services.vector_store import VectorStoreService
from app.services.text_processor import TextProcessor
from app.services.storage_service import storage_service
from app.core.config import settings
from app.core.cache import cache_service


class DocumentService:
    """Service for managing documents and document processing."""
    
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.text_processor = TextProcessor()
        self._vector_store_initialized = False
    
    async def _ensure_vector_store_initialized(self):
        """Ensure vector store is initialized."""
        if not self._vector_store_initialized:
            await self.vector_store.initialize()
            self._vector_store_initialized = True
    
    async def get_documents(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        source_id: Optional[UUID] = None,
        search: Optional[str] = None,
        order_by: Optional[str] = "updated_at",
        order: Optional[str] = "desc"
    ) -> tuple[List[Document], int]:
        """
        Get paginated list of documents with total count.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            source_id: Optional source ID filter
            search: Optional search term
            order_by: Field to order by (default: updated_at)
            order: Sort order - 'asc' or 'desc' (default: desc)
            db: Database session
            
        Returns:
            Tuple of (documents list, total count)
        """
        from sqlalchemy import func, desc, asc
        
        # Build base query for filtering
        base_query = select(Document)
        
        # Apply filters
        if source_id:
            base_query = base_query.where(Document.source_id == source_id)
        
        if search:
            search_term = f"%{search}%"
            base_query = base_query.where(
                or_(
                    Document.title.ilike(search_term),
                    Document.content.ilike(search_term),
                    Document.author.ilike(search_term)
                )
            )
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results with ordering
        # Use noload for chunks to prevent lazy loading (chunks not needed in list view)
        query = base_query.options(
            selectinload(Document.source),
            noload(Document.chunks)
        )
        
        # Apply ordering
        order_field_map = {
            "updated_at": Document.updated_at,
            "created_at": Document.created_at,
            "title": Document.title,
            "author": Document.author,
        }
        
        order_field = order_field_map.get(order_by, Document.updated_at)
        if order.lower() == "asc":
            query = query.order_by(asc(order_field))
        else:
            query = query.order_by(desc(order_field))
        
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return documents, total
    
    async def get_document(self, document_id: UUID, db: AsyncSession) -> Optional[Document]:
        """
        Get a document by ID with caching.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            Document object or None if not found
        """
        # Try cache first
        cache_key = f"document:{document_id}"
        cached_doc = await cache_service.get(cache_key)
        if cached_doc is not None:
            return cached_doc
        
        # Get from database
        result = await db.execute(
            select(Document)
            .options(
                selectinload(Document.source),
                selectinload(Document.chunks)
            )
            .where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        # Cache result if found (TTL: 1 hour)
        if document:
            await cache_service.set(cache_key, document, ttl=3600)
        
        return document
    
    async def upload_file(
        self,
        file: UploadFile,
        db: AsyncSession,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Document:
        """Upload and process a file."""
        try:
            logger.info(f"Starting upload for file: {file.filename}, content_type: {file.content_type}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
            # Create upload source if it doesn't exist
            upload_source = await self._get_or_create_upload_source(db)
            
            # Read file content
            content = await file.read()
            
            # Generate file hash
            content_hash = hashlib.sha256(content).hexdigest()
            
            # Check if document already exists
            existing_doc = await db.execute(
                select(Document).where(
                    and_(
                        Document.content_hash == content_hash,
                        Document.source_id == upload_source.id
                    )
                )
            )
            existing_doc = existing_doc.scalar_one_or_none()
            
            if existing_doc:
                logger.info(f"Document already exists: {existing_doc.id}")
                return existing_doc
            
            # Create document record first to get ID
            # Use content_type if available, otherwise derive from filename
            file_type = file.content_type or self._get_file_extension(file.filename)
            if not file_type and file.filename:
                # Fallback: try to determine from extension
                ext = os.path.splitext(file.filename)[1].lower()
                from app.utils.validators import ALLOWED_FILE_TYPES
                for mime_type, extensions in ALLOWED_FILE_TYPES.items():
                    if ext in extensions:
                        file_type = mime_type
                        break
            
            document = Document(
                title=title or file.filename or "Uploaded Document",
                content="",  # Will be filled after text extraction
                content_hash=content_hash,
                file_path="",  # Will be filled after MinIO upload
                file_type=file_type,
                file_size=len(content),
                source_id=upload_source.id,
                source_identifier=file.filename or f"upload_{content_hash[:8]}",
                tags=tags,
                extra_metadata={
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            db.add(document)
            await db.commit()
            await db.refresh(document)
            
            # Extract text content from temporary file
            # Save to temp file for text extraction (text processor needs file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or "")[1]) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Upload to MinIO first
                logger.info(f"Uploading file to MinIO: {file.filename}")
                object_path = await self._save_uploaded_file(document.id, file, content)
                document.file_path = object_path
                logger.info(f"File uploaded to MinIO: {object_path}")
                
                # Check if file is video/audio that needs transcription
                from app.services.transcription_service import get_transcription_service
                transcription_service = get_transcription_service()
                file_path_obj = Path(temp_file_path)
                
                if transcription_service and transcription_service.is_supported_format(file_path_obj):
                    # Video/audio file
                    logger.info(f"Video/audio file detected: {file.filename}.")
                    document.content = ""  # Will be filled by background task
                    document.extra_metadata = document.extra_metadata or {}
                    
                    # If non-MP4 video -> transcode first, then transcribe
                    try:
                        from pathlib import Path as _Path
                        suffix = _Path(file.filename or "").suffix.lower()
                        is_video = (file.content_type or "").startswith("video/")
                        is_mp4 = suffix == ".mp4" or (file.content_type == "video/mp4")
                    except Exception:
                        is_video = True
                        is_mp4 = False

                    if is_video and not is_mp4:
                        document.extra_metadata["is_transcoding"] = True
                        document.extra_metadata.pop("is_transcribing", None)
                        try:
                            from app.tasks.transcode_tasks import transcode_to_mp4, _publish_status as publish_status
                            publish_status(str(document.id), {
                                "is_transcoding": True,
                                "is_transcribing": False,
                                "is_transcribed": False,
                            })
                            transcode_to_mp4.delay(str(document.id))
                            logger.info(f"Triggered transcode (then transcribe) for document {document.id}")
                        except Exception as e:
                            logger.error(f"Failed to dispatch transcode task: {e}")
                            document.extra_metadata["transcode_error"] = str(e)
                    else:
                        # Audio or already MP4 -> transcribe now
                        document.extra_metadata["is_transcribing"] = True
                        try:
                            from app.tasks.transcription_tasks import transcribe_document, _publish_status as publish_status
                            publish_status(str(document.id), {
                                "is_transcoding": False,
                                "is_transcribing": True,
                                "is_transcribed": False,
                            })
                            transcribe_document.delay(str(document.id))
                            logger.info(f"Triggered transcription task for document {document.id}")
                        except Exception as e:
                            logger.error(f"Failed to dispatch transcription task: {e}")
                else:
                    # Regular document - extract text content
                    logger.info(f"Extracting text from regular document: {file.filename}")
                    try:
                        text_content = await self.text_processor.extract_text(temp_file_path, file.content_type)
                        document.content = text_content
                        logger.info(f"Extracted {len(text_content)} characters from document")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from {file.filename}: {e}. Continuing with empty content.")
                        document.content = ""
                    
                    # Process document asynchronously
                    await self._process_document_async(document, db)
                
                await db.commit()
                await db.refresh(document)
                logger.info(f"Successfully uploaded and processed document: {document.id}")
            except Exception as e:
                logger.error(f"Error during file processing for {file.filename}: {e}", exc_info=True)
                # Rollback document creation if processing fails
                await db.rollback()
                raise
            finally:
                # Clean up temp file
                try:
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        logger.debug(f"Cleaned up temp file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
            
            # Cache the newly created document
            cache_key = f"document:{document.id}"
            await cache_service.set(cache_key, document, ttl=3600)
            
            logger.info(f"Uploaded document: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def _save_uploaded_file(
        self,
        document_id: UUID,
        file: UploadFile,
        content: bytes
    ) -> str:
        """
        Save uploaded file to MinIO.
        
        Returns:
            Object path in MinIO (e.g., "documents/{document_id}/{filename}")
        """
        # Upload to MinIO
        object_path = await storage_service.upload_file(
            document_id=document_id,
            filename=file.filename or "uploaded_file",
            content=content,
            content_type=file.content_type
        )
        
        return object_path
    
    def _get_file_extension(self, filename: Optional[str]) -> Optional[str]:
        """Get file extension from filename."""
        if not filename:
            return None
        return os.path.splitext(filename)[1].lower().lstrip('.')
    
    async def _get_or_create_upload_source(self, db: AsyncSession) -> DocumentSource:
        """Get or create the upload document source."""
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.name == "File Upload")
        )
        source = result.scalar_one_or_none()
        
        if not source:
            source = DocumentSource(
                name="File Upload",
                source_type="file",
                config={"type": "upload", "description": "Manually uploaded files"}
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)
        
        return source
    
    async def _process_document_async(self, document: Document, db: AsyncSession):
        """Process document for indexing (should be moved to background task)."""
        try:
            # Ensure vector store is initialized
            await self._ensure_vector_store_initialized()
            
            # Split text into chunks with metadata
            chunks_with_metadata = await self.text_processor.split_text_with_metadata(
                text=document.content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                strategy=settings.RAG_CHUNKING_STRATEGY
            )
            
            # Create document chunks with enhanced metadata
            document_chunks = []
            for chunk_data in chunks_with_metadata:
                chunk_content = chunk_data["content"]
                chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()
                
                # Build metadata dictionary
                chunk_metadata = {
                    "section_title": chunk_data.get("section_title"),
                    "paragraph_index": chunk_data.get("paragraph_index"),
                    "semantic_score": chunk_data.get("semantic_score", 1.0)
                }
                
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_content,
                    content_hash=chunk_hash,
                    chunk_index=chunk_data["chunk_index"],
                    start_pos=chunk_data.get("start_pos"),
                    end_pos=chunk_data.get("end_pos"),
                    metadata=chunk_metadata
                )
                
                document_chunks.append(chunk)
                db.add(chunk)
            
            await db.commit()
            
            # Add to vector store
            await self.vector_store.add_document_chunks(document, document_chunks)
            
            # Update document as processed
            document.is_processed = True
            document.processing_error = None
            await db.commit()
            
            # Invalidate and refresh cache with updated document
            cache_key = f"document:{document.id}"
            await cache_service.delete(cache_key)
            await cache_service.set(cache_key, document, ttl=3600)
            
            logger.info(f"Processed document {document.id} with {len(document_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            
            # Update document with error
            document.is_processed = False
            document.processing_error = str(e)
            await db.commit()
    
    async def delete_document(self, document_id: UUID, db: AsyncSession) -> bool:
        """
        Delete a document and all associated data.
        
        This method deletes:
        1. File from MinIO storage
        2. Chunks from vector store (ChromaDB)
        3. Chunks from database (explicitly, in addition to cascade)
        4. Document from database
        5. Cache entries
        
        Args:
            document_id: UUID of the document to delete
            db: Database session
            
        Returns:
            True if deletion was successful, False otherwise
        """
        # Try to get document directly from database (bypass cache)
        # This ensures we get the latest state even if cache is stale
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            # Also try cache in case it exists there but not in DB (shouldn't happen, but for debugging)
            cache_key = f"document:{document_id}"
            cached_doc = await cache_service.get(cache_key)
            if cached_doc:
                logger.warning(f"Document {document_id} found in cache but not in database - possible race condition")
            else:
                logger.warning(f"Document {document_id} not found in database or cache for deletion")
            return False
        
        try:
            logger.info(f"Starting deletion of document {document_id}: {document.title}")
            
            # Ensure vector store is initialized
            await self._ensure_vector_store_initialized()
            
            # Step 1: Delete from MinIO if file_path exists
            if document.file_path:
                try:
                    logger.info(f"Deleting file from MinIO: {document.file_path}")
                    deleted = await storage_service.delete_file(document.file_path)
                    if deleted:
                        logger.info(f"Successfully deleted file from MinIO: {document.file_path}")
                    else:
                        logger.warning(f"File deletion returned False for: {document.file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file from MinIO {document.file_path}: {e}", exc_info=True)
                    # Continue with deletion even if MinIO delete fails
            
            # Step 2: Delete from vector store (ChromaDB)
            try:
                logger.info(f"Deleting chunks from vector store for document {document_id}")
                await self.vector_store.delete_document_chunks(document_id)
                logger.info(f"Successfully deleted chunks from vector store for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks from vector store for document {document_id}: {e}")
                # Continue with deletion even if vector store delete fails
            
            # Step 3: Delete chunks from database explicitly (in addition to cascade)
            try:
                from app.models.document import DocumentChunk
                from sqlalchemy import delete
                
                delete_chunks_stmt = delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                )
                result = await db.execute(delete_chunks_stmt)
                deleted_chunks = result.rowcount
                logger.info(f"Deleted {deleted_chunks} chunks from database for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks from database for document {document_id}: {e}")
                # Continue with deletion even if chunk delete fails
            
            # Step 3.5: Delete or update upload sessions that reference this document
            try:
                from app.models.upload_session import UploadSession
                
                # Find upload sessions that reference this document
                upload_sessions_result = await db.execute(
                    select(UploadSession).where(UploadSession.document_id == document_id)
                )
                upload_sessions = upload_sessions_result.scalars().all()
                
                if upload_sessions:
                    logger.info(f"Found {len(upload_sessions)} upload session(s) referencing document {document_id}")
                    # Delete the upload sessions (they're just tracking records)
                    for session in upload_sessions:
                        await db.delete(session)
                    logger.info(f"Deleted {len(upload_sessions)} upload session(s) for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to delete upload sessions for document {document_id}: {e}")
                # Continue with deletion even if upload session delete fails
            
            # Step 4: Delete document from database
            # Use delete() method which is async in SQLAlchemy 2.0
            await db.delete(document)
            await db.flush()  # Flush before commit to ensure all deletes are processed
            await db.commit()
            logger.info(f"Successfully deleted document {document_id} from database")
            
            # Step 5: Invalidate cache
            try:
                cache_key = f"document:{document_id}"
                await cache_service.delete(cache_key)
                # Also invalidate documents list cache
                await cache_service.delete_pattern("documents:*")
                logger.info(f"Invalidated cache for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for document {document_id}: {e}")
            
            logger.info(f"Successfully deleted document {document_id}: {document.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
            await db.rollback()
            return False
    
    async def reprocess_document(self, document_id: UUID, db: AsyncSession) -> bool:
        """Reprocess a document for indexing."""
        document = await self.get_document(document_id, db)
        if not document:
            return False
        
        try:
            # Ensure vector store is initialized
            await self._ensure_vector_store_initialized()
            
            # Delete existing chunks from vector store
            await self.vector_store.delete_document_chunks(document_id)
            
            # Delete existing chunks from database
            for chunk in document.chunks:
                await db.delete(chunk)
            await db.commit()
            
            # Reprocess document
            await self._process_document_async(document, db)
            
            # Invalidate cache (already done in _process_document_async, but ensure it's cleared)
            cache_key = f"document:{document_id}"
            await cache_service.delete(cache_key)
            
            logger.info(f"Reprocessed document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {e}")
            return False
    
    # Document Source methods
    async def get_document_sources(self, db: AsyncSession) -> List[DocumentSource]:
        """Get all document sources."""
        result = await db.execute(select(DocumentSource).order_by(DocumentSource.name))
        return result.scalars().all()
    
    async def create_document_source(
        self,
        name: str,
        source_type: str,
        config: Dict[str, Any],
        db: AsyncSession
    ) -> DocumentSource:
        """Create a new document source."""
        source = DocumentSource(
            name=name,
            source_type=source_type,
            config=config
        )
        
        db.add(source)
        await db.commit()
        await db.refresh(source)
        
        logger.info(f"Created document source: {name}")
        return source
    
    async def update_document_source(
        self,
        source_id: UUID,
        name: str,
        source_type: str,
        config: Dict[str, Any],
        db: AsyncSession
    ) -> Optional[DocumentSource]:
        """Update a document source."""
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return None
        
        source.name = name
        source.source_type = source_type
        source.config = config
        source.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Updated document source: {source_id}")
        return source
    
    async def delete_document_source(self, source_id: UUID, db: AsyncSession) -> bool:
        """Delete a document source and all its documents."""
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return False
        
        try:
            # This will cascade delete all documents and chunks
            await db.delete(source)
            await db.commit()
            
            logger.info(f"Deleted document source: {source_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document source {source_id}: {e}")
            return False
    
    async def sync_document_source(self, source_id: UUID, db: AsyncSession) -> bool:
        """Trigger synchronization for a document source."""
        # This would be implemented based on the specific source type
        # For now, just update the last_sync timestamp
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        
        if not source:
            return False
        
        source.last_sync = datetime.utcnow()
        await db.commit()
        
        logger.info(f"Triggered sync for document source: {source_id}")
        return True

