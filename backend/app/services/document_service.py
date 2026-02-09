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
from app.models.persona import DocumentPersonaDetection
from app.models.user import User
from app.services.vector_store import vector_store_service
from app.services.text_processor import TextProcessor
from app.services.storage_service import storage_service
from app.core.config import settings
from app.core.cache import cache_service
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.persona_service import persona_service
import json
import redis


class DocumentService:
    """Service for managing documents and document processing."""
    
    def __init__(self):
        self.vector_store = vector_store_service
        self.text_processor = TextProcessor()
        self._vector_store_initialized = False
        self.llm = LLMService()
    
    async def _ensure_vector_store_initialized(self):
        """Ensure vector store is initialized."""
        if not self._vector_store_initialized:
            await self.vector_store.initialize(background=True)
            self._vector_store_initialized = True
    
    async def get_documents(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        source_id: Optional[UUID] = None,
        search: Optional[str] = None,
        order_by: Optional[str] = "updated_at",
        order: Optional[str] = "desc",
        owner_persona_id: Optional[UUID] = None,
        persona_id: Optional[UUID] = None,
        persona_role: Optional[str] = None,
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

        if owner_persona_id:
            base_query = base_query.where(Document.owner_persona_id == owner_persona_id)

        if persona_id or persona_role:
            base_query = base_query.join(Document.persona_detections)
            if persona_id:
                base_query = base_query.where(DocumentPersonaDetection.persona_id == persona_id)
            if persona_role:
                base_query = base_query.where(DocumentPersonaDetection.role == persona_role)
            base_query = base_query.distinct()
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results with ordering
        # Use noload for chunks and persona_detections to prevent lazy loading (not needed in list view)
        query = base_query.options(
            selectinload(Document.source),
            noload(Document.chunks),
            noload(Document.persona_detections),
            selectinload(Document.owner_persona),
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
        Get a document by ID.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            Document object or None if not found
        """
        # Get from database
        result = await db.execute(
            select(Document)
            .options(
                selectinload(Document.source),
                selectinload(Document.chunks),
                selectinload(Document.owner_persona),
                selectinload(Document.persona_detections).selectinload(DocumentPersonaDetection.persona),
            )
            .where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        return document
    
    async def upload_file(
        self,
        file: UploadFile,
        db: AsyncSession,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        owner_user: Optional[User] = None
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
            
            owner_display_name = None
            if owner_user:
                owner_display_name = owner_user.full_name or owner_user.username or owner_user.email

            document = Document(
                title=title or file.filename or "Uploaded Document",
                content="",  # Will be filled after text extraction
                content_hash=content_hash,
                file_path="",  # Will be filled after MinIO upload
                file_type=file_type,
                file_size=len(content),
                source_id=upload_source.id,
                source_identifier=file.filename or f"upload_{content_hash[:8]}",
                author=owner_display_name,
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

            if owner_user:
                await persona_service.assign_document_owner(
                    db,
                    document,
                    user=owner_user,
                    platform_scope="file-upload"
                )
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
                        text_content, extraction_metadata = await self.text_processor.extract_text(temp_file_path, file.content_type)
                        document.content = text_content
                        if extraction_metadata:
                            document.extra_metadata = document.extra_metadata or {}
                            document.extra_metadata.update(extraction_metadata)
                        logger.info(f"Extracted {len(text_content)} characters from document")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from {file.filename}: {e}. Continuing with empty content.")
                        document.content = ""
                    
                    # Process document asynchronously
                    await self._process_document_async(document, db, user_id=owner_user.id if owner_user else None)
                
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

    async def _get_or_create_agent_notes_source(self, db: AsyncSession) -> DocumentSource:
        """Get or create the agent-created notes document source."""
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.name == "Agent Notes")
        )
        source = result.scalar_one_or_none()

        if not source:
            source = DocumentSource(
                name="Agent Notes",
                source_type="file",
                config={"type": "agent_notes", "description": "Notes created by the in-app agent/tools"},
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)

        return source

    async def _get_or_create_latex_projects_source(self, db: AsyncSession) -> DocumentSource:
        """Get or create the LaTeX Studio projects document source."""
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.name == "LaTeX Projects")
        )
        source = result.scalar_one_or_none()

        if not source:
            source = DocumentSource(
                name="LaTeX Projects",
                source_type="file",
                config={"type": "latex_projects", "description": "LaTeX Studio projects published into the knowledge base"},
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)

        return source

    async def _get_or_create_url_ingest_source(self, db: AsyncSession) -> DocumentSource:
        """Get or create the URL ingestion source."""
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.name == "URL Ingest")
        )
        source = result.scalar_one_or_none()

        if not source:
            source = DocumentSource(
                name="URL Ingest",
                source_type="web",
                # Keep this source inactive so periodic web source sync jobs don't try to crawl it.
                is_active=False,
                config={"type": "url_ingest", "description": "Ad-hoc URL ingestion into the knowledge base"},
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)
        elif source.is_active:
            # Safety: if it exists and is active, disable to avoid scheduled web syncs.
            source.is_active = False
            await db.commit()

        return source
    
    async def _process_document_async(self, document: Document, db: AsyncSession, user_id: Optional[UUID] = None):
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

            # Extract knowledge graph (entities and relations) from chunks
            from app.core.feature_flags import get_flag as _get_flag
            if await _get_flag("knowledge_graph_enabled"):
                try:
                    from app.services.knowledge_extraction import extractor as rule_extractor, llm_extractor
                    from app.models.memory import UserPreferences
                    from app.services.llm_service import UserLLMSettings

                    # Best-effort: use per-user LLM settings for KG ingestion when a user_id is available.
                    effective_user_settings = None
                    if user_id is not None:
                        try:
                            prefs_result = await db.execute(
                                select(UserPreferences).where(UserPreferences.user_id == user_id)
                            )
                            user_prefs = prefs_result.scalar_one_or_none()
                            effective_user_settings = UserLLMSettings.from_preferences(user_prefs)
                        except Exception as _pref_err:
                            logger.warning(f"Failed to load user LLM settings for KG ingestion: {_pref_err}")

                    use_llm = bool(getattr(settings, "KG_LLM_EXTRACTION_ENABLED", False))
                    total_mentions, total_relations = 0, 0
                    for ch in document_chunks:
                        if use_llm:
                            m, r = await llm_extractor.index_chunk(
                                db, document, ch, rule_extractor=rule_extractor, user_settings=effective_user_settings
                            )
                        else:
                            m, r = await rule_extractor.index_chunk(db, document, ch)
                        total_mentions += m
                        total_relations += r
                    await db.commit()
                    logger.info(f"KG extraction for document {document.id}: mentions={total_mentions}, relations={total_relations}")
                except Exception as e:
                    logger.warning(f"Knowledge extraction failed for document {document.id}: {e}")
            
            # Update document as processed
            document.is_processed = True
            document.processing_error = None
            await db.commit()
            
            # Invalidate and refresh cache with updated document
            cache_key = f"document:{document.id}"
            await cache_service.delete(cache_key)
            await cache_service.set(cache_key, document, ttl=3600)
            
            logger.info(f"Processed document {document.id} with {len(document_chunks)} chunks")

            # Optionally auto-summarize
            if await _get_flag("summarization_enabled") and await _get_flag("auto_summarize_on_process"):
                try:
                    from app.tasks.summarization_tasks import summarize_document as _summ_task
                    _summ_task.delay(str(document.id), False, user_id=str(user_id) if user_id else None)
                    logger.info(f"Scheduled auto-summarization for document {document.id}")
                except Exception as _e:
                    logger.warning(f"Failed to schedule auto-summarization for {document.id}: {_e}")
            
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

    async def delete_documents_by_source(self, source_id: UUID, db: AsyncSession) -> int:
        """Delete all documents for a specific source."""
        result = await db.execute(
            select(Document.id).where(Document.source_id == source_id)
        )
        document_ids = [row[0] for row in result.fetchall()]
        deleted = 0
        for doc_id in document_ids:
            try:
                success = await self.delete_document(doc_id, db)
                if success:
                    deleted += 1
            except Exception as exc:
                logger.warning(f"Failed to delete document {doc_id} for source {source_id}: {exc}")
        return deleted
    
    async def reprocess_document(self, document_id: UUID, db: AsyncSession, user_id: Optional[UUID] = None) -> bool:
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
            await self._process_document_async(document, db, user_id=user_id)

            # Invalidate cache (already done in _process_document_async, but ensure it's cleared)
            cache_key = f"document:{document_id}"
            await cache_service.delete(cache_key)
            
            logger.info(f"Reprocessed document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {e}")
            return False

    async def summarize_document(
        self,
        document_id: UUID,
        db: AsyncSession,
        force: bool = False,
        model: Optional[str] = None,
        user_settings: Optional[UserLLMSettings] = None,
    ) -> Optional[str]:
        """Generate and store a summary for a document using the LLM.

        Args:
            document_id: UUID of document
            db: session
            force: if True, regenerates even if summary exists
            model: optional model override
            user_settings: optional user LLM settings for provider preference

        Returns:
            The generated summary text, or None if document not found.
        """
        document = await self.get_document(document_id, db)
        if not document:
            return None
        if document.summary and not force:
            return document.summary
        text = document.content or ""
        if not text.strip():
            # Fall back to concatenated chunks
            if hasattr(document, 'chunks') and document.chunks:
                text = "\n\n".join([c.content for c in document.chunks])

        # Heuristics for heavy jobs and chunking
        from app.core.config import settings as _settings
        heavy_threshold = getattr(_settings, 'SUMMARIZATION_HEAVY_THRESHOLD_CHARS', 30000)
        prefer_deepseek = bool(getattr(_settings, 'DEEPSEEK_API_KEY', None)) and len(text) > heavy_threshold

        chunk_size = getattr(_settings, 'SUMMARIZATION_CHUNK_SIZE_CHARS', 12000)
        chunk_overlap = getattr(_settings, 'SUMMARIZATION_CHUNK_OVERLAP_CHARS', 800)

        def _chunk_text(t: str) -> List[str]:
            if len(t) <= chunk_size:
                return [t]
            chunks: List[str] = []
            start = 0
            while start < len(t):
                end = min(len(t), start + chunk_size)
                chunk = t[start:end]
                chunks.append(chunk)
                if end == len(t):
                    break
                start = max(end - chunk_overlap, start + 1)
            return chunks

        chunks = _chunk_text(text)

        system = (
            "Summarize the text with: (1) a 3-5 sentence abstract, "
            "(2) 5-10 bullet key takeaways, and (3) any dates or action items."
        )

        # Helper to publish summarization progress to Redis (for WebSocket bridge)
        def _publish_sum_progress(progress: dict):
            try:
                client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                channel = f"summarization_progress:{str(document_id)}"
                msg = json.dumps({
                    "type": "progress",
                    "document_id": str(document_id),
                    "progress": progress,
                })
                client.publish(channel, msg)
            except Exception as e:
                logger.debug(f"Failed to publish summarization progress to Redis: {e}")

        # Track which model was actually used
        actual_model_used = None
        
        # Helper function to generate response with fallback to local model
        async def _generate_with_fallback(query: str, max_tokens: int, use_deepseek: bool = False) -> str:
            """Generate response, falling back to local model if remote fails."""
            nonlocal actual_model_used
            first_error = None
            try:
                result = await self.llm.generate_response(
                    query=query,
                    context=None,
                    model=model or None,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    prefer_deepseek=use_deepseek,
                    task_type="summarization",
                    user_settings=user_settings,
                )
                if use_deepseek:
                    actual_model_used = getattr(_settings, 'DEEPSEEK_MODEL', 'deepseek-chat')
                else:
                    actual_model_used = model or self.llm.default_model
                return result
            except Exception as e:
                first_error = e
                if use_deepseek:
                    # If DeepSeek failed, try with local model (Ollama)
                    logger.warning(f"Remote model (DeepSeek) failed: {e}. Falling back to local model (Ollama).")
                    try:
                        result = await self.llm.generate_response(
                            query=query,
                            context=None,
                            model=model or None,
                            temperature=0.2,
                            max_tokens=max_tokens,
                            prefer_deepseek=False,  # Force local model
                            task_type="summarization",
                            user_settings=user_settings,
                        )
                        actual_model_used = model or self.llm.default_model
                        logger.info(f"Successfully used local model (Ollama) after DeepSeek failure")
                        return result
                    except Exception as local_error:
                        error_msg = (
                            f"Both remote (DeepSeek) and local (Ollama) models failed. "
                            f"Remote error: {str(first_error)}. "
                            f"Local error: {str(local_error)}. "
                            f"Please ensure Ollama is running and accessible at {self.llm.base_url}"
                        )
                        logger.error(error_msg)
                        from app.utils.exceptions import LLMServiceError
                        raise LLMServiceError(error_msg) from local_error
                else:
                    # Already using local model, provide helpful error message
                    error_msg = (
                        f"Local model (Ollama) failed: {str(e)}. "
                        f"Please ensure Ollama is running and accessible at {self.llm.base_url}"
                    )
                    logger.error(error_msg)
                    from app.utils.exceptions import LLMServiceError
                    raise LLMServiceError(error_msg) from e

        if len(chunks) == 1:
            query = f"{system}\n\nDocument:\n{chunks[0]}"
            summary = await _generate_with_fallback(query, max_tokens=800, use_deepseek=prefer_deepseek)
        else:
            # Summarize each chunk first
            chunk_summaries: List[str] = []
            for i, ch in enumerate(chunks, 1):
                q = f"{system}\n\nChunk {i}/{len(chunks)}:\n{ch}"
                s = await _generate_with_fallback(q, max_tokens=600, use_deepseek=prefer_deepseek)
                chunk_summaries.append(s)
                # Publish incremental progress (up to 90%)
                pct = int((i / max(1, len(chunks))) * 90)
                _publish_sum_progress({
                    "stage": "chunk",
                    "current_chunk": i,
                    "total_chunks": len(chunks),
                    "progress": pct,
                })

            # Combine chunk summaries into a final cohesive summary
            combine_prompt = (
                "You are synthesizing a final document summary from chunk summaries.\n"
                "Combine, deduplicate, and produce: (1) a cohesive abstract;"
                " (2) 7-12 bullet key takeaways grouped by theme;"
                " (3) important dates/action items. Keep under 600 words.\n\n"
                "Chunk summaries:\n" + "\n\n---\n\n".join(chunk_summaries)
            )
            _publish_sum_progress({
                "stage": "combining",
                "progress": 95,
            })
            summary = await _generate_with_fallback(combine_prompt, max_tokens=1000, use_deepseek=prefer_deepseek)

        # Track which model was actually used (will be set by the fallback function)
        from datetime import datetime as _dt
        document.summary = summary
        # Set the model that was actually used (may differ from preferred if fallback occurred)
        document.summary_model = actual_model_used or (
            getattr(_settings, 'DEEPSEEK_MODEL', 'deepseek-chat') if prefer_deepseek and bool(getattr(_settings, 'DEEPSEEK_API_KEY', None))
            else (model or self.llm.default_model)
        )
        document.summary_generated_at = _dt.utcnow()
        await db.commit()
        # Invalidate cache so next fetch includes summary
        await cache_service.delete(f"document:{document_id}")
        return summary
    
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
