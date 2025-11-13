"""
Document service for managing documents and document sources.
"""

import os
import hashlib
import aiofiles
from datetime import datetime
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
            
            # Save file to disk
            file_path = await self._save_uploaded_file(file, content)
            
            # Extract text content
            text_content = await self.text_processor.extract_text(file_path, file.content_type)
            
            # Create document record
            document = Document(
                title=title or file.filename or "Uploaded Document",
                content=text_content,
                content_hash=content_hash,
                file_path=file_path,
                file_type=self._get_file_extension(file.filename),
                file_size=len(content),
                source_id=upload_source.id,
                source_identifier=file.filename or f"upload_{content_hash[:8]}",
                tags=tags,
                metadata={
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            db.add(document)
            await db.commit()
            await db.refresh(document)
            
            # Process document asynchronously
            await self._process_document_async(document, db)
            
            # Cache the newly created document
            cache_key = f"document:{document.id}"
            await cache_service.set(cache_key, document, ttl=3600)
            
            logger.info(f"Uploaded document: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def _save_uploaded_file(self, file: UploadFile, content: bytes) -> str:
        """Save uploaded file to disk."""
        # Create documents directory if it doesn't exist
        docs_dir = "./data/documents"
        os.makedirs(docs_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(docs_dir, filename)
        
        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        
        return file_path
    
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
        """Delete a document and its chunks."""
        document = await self.get_document(document_id, db)
        if not document:
            return False
        
        try:
            # Ensure vector store is initialized
            await self._ensure_vector_store_initialized()
            
            # Delete from vector store
            await self.vector_store.delete_document_chunks(document_id)
            
            # Delete file if it exists
            if document.file_path and os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            # Delete from database
            await db.delete(document)
            await db.commit()
            
            # Invalidate cache
            cache_key = f"document:{document_id}"
            await cache_service.delete(cache_key)
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
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


