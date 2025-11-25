"""
Chunked upload endpoints for restartable file uploads.
"""

import os
import math
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.models.upload_session import UploadSession
from app.services.auth_service import get_current_user
from app.services.storage_service import storage_service
from app.services.document_service import DocumentService
from app.utils.validators import validate_file_type
from app.utils.exceptions import ValidationError


router = APIRouter()
document_service = DocumentService()

# Chunk size: 5MB (good balance between network efficiency and memory usage)
CHUNK_SIZE = 5 * 1024 * 1024  # 5MB


@router.post("/init")
async def init_upload(
    filename: str = Form(...),
    file_size: int = Form(...),
    file_type: Optional[str] = Form(None),
    content_type: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Initialize a chunked upload session.
    
    Returns:
        Upload session ID and chunk information
    """
    try:
        # Validate file type
        if not validate_file_type(filename, content_type):
            raise ValidationError(
                f"File type not allowed: {content_type}",
                field="file"
            )
        
        # Validate file size
        from app.core.config import settings
        file_ext = os.path.splitext(filename)[1].lower()
        is_video_audio = file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', 
                                      '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        max_size = settings.MAX_VIDEO_SIZE if is_video_audio else settings.MAX_FILE_SIZE
        
        if file_size > max_size:
            size_mb = file_size / (1024*1024)
            max_mb = max_size / (1024*1024)
            raise ValidationError(
                f"File size ({size_mb:.2f}MB) exceeds maximum allowed size of {max_mb:.0f}MB",
                field="file_size"
            )
        
        # Parse tags
        tag_list = []
        if tags:
            from app.utils.validators import validate_tags
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if not validate_tags(tag_list):
                raise ValidationError("Invalid tag format", field="tags")
        
        # Calculate chunk information
        total_chunks = math.ceil(file_size / CHUNK_SIZE)
        
        # Create upload session
        session = UploadSession(
            user_id=current_user.id,
            filename=filename,
            file_size=file_size,
            file_type=file_type or content_type,
            content_type=content_type,
            chunk_size=CHUNK_SIZE,
            total_chunks=total_chunks,
            uploaded_chunks=[],
            uploaded_bytes=0,
            title=title,
            tags=tag_list,
            status="pending",
            expires_at=datetime.utcnow() + timedelta(hours=24)  # 24 hour expiration
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        logger.info(f"Created upload session {session.id} for file {filename} ({file_size} bytes, {total_chunks} chunks)")
        
        return {
            "session_id": str(session.id),
            "chunk_size": CHUNK_SIZE,
            "total_chunks": total_chunks,
            "uploaded_chunks": []
        }
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error initializing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize upload: {str(e)}")


@router.get("/{session_id}/status")
async def get_upload_status(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get upload session status for resume capability."""
    result = await db.execute(
        select(UploadSession).where(
            UploadSession.id == session_id,
            UploadSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    return {
        "session_id": str(session.id),
        "filename": session.filename,
        "file_size": session.file_size,
        "total_chunks": session.total_chunks,
        "uploaded_chunks": session.uploaded_chunks,
        "uploaded_bytes": session.uploaded_bytes,
        "progress": session.progress_percentage,
        "status": session.status,
        "can_resume": session.can_resume
    }


@router.post("/{session_id}/chunk")
async def upload_chunk(
    session_id: UUID,
    chunk_number: int = Form(...),
    chunk: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a single chunk.
    
    Args:
        session_id: Upload session ID
        chunk_number: Chunk number (0-indexed)
        chunk: Chunk file data
    """
    try:
        # Get upload session
        result = await db.execute(
            select(UploadSession).where(
                UploadSession.id == session_id,
                UploadSession.user_id == current_user.id
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        if session.status == "completed":
            raise HTTPException(status_code=400, detail="Upload already completed")
        
        if session.status == "failed":
            raise HTTPException(status_code=400, detail="Upload session failed")
        
        # Check if chunk already uploaded
        if chunk_number in session.uploaded_chunks:
            logger.info(f"Chunk {chunk_number} already uploaded for session {session_id}")
            return {
                "session_id": str(session.id),
                "chunk_number": chunk_number,
                "uploaded_bytes": session.uploaded_bytes,
                "progress": session.progress_percentage,
                "status": "already_uploaded"
            }
        
        # Read chunk data
        chunk_data = await chunk.read()
        chunk_size = len(chunk_data)
        
        # Validate chunk size (last chunk can be smaller)
        if chunk_number < session.total_chunks - 1:
            if chunk_size != CHUNK_SIZE:
                raise ValidationError(
                    f"Chunk size mismatch: expected {CHUNK_SIZE}, got {chunk_size}",
                    field="chunk"
                )
        else:
            # Last chunk
            expected_size = session.file_size - (session.total_chunks - 1) * CHUNK_SIZE
            if chunk_size > expected_size:
                raise ValidationError(
                    f"Last chunk too large: expected max {expected_size}, got {chunk_size}",
                    field="chunk"
                )
        
        # Update session status
        session.status = "uploading"
        
        # Initialize multipart upload if not already done
        if not session.minio_upload_id:
            # Generate object path (we'll use a temporary path, then move it)
            from uuid import uuid4
            temp_doc_id = uuid4()
            object_path = storage_service._get_object_path(temp_doc_id, session.filename)
            
            # Create multipart upload (or get placeholder for manual reassembly)
            upload_id = await storage_service.create_multipart_upload(
                object_path,
                session.content_type
            )
            session.minio_upload_id = upload_id
            session.extra_metadata = session.extra_metadata or {}
            session.extra_metadata["temp_object_path"] = object_path
            session.extra_metadata["temp_doc_id"] = str(temp_doc_id)
            session.extra_metadata["chunk_paths"] = {}  # Store individual chunk paths for manual reassembly
            session.extra_metadata["use_multipart"] = not upload_id.startswith("upload_")  # Store detection result
            flag_modified(session, "extra_metadata")
        
        object_path = session.extra_metadata.get("temp_object_path")
        use_multipart = session.extra_metadata.get("use_multipart", False)
        
        # Check if we're using MinIO multipart upload or manual reassembly
        # If upload_id doesn't start with "upload_", it's a real MinIO multipart upload
        if use_multipart and not session.minio_upload_id.startswith("upload_"):
            # Use MinIO multipart upload
            part_number = chunk_number + 1  # MinIO uses 1-indexed part numbers
            etag = await storage_service.upload_part(
                object_path,
                session.minio_upload_id,
                part_number,
                chunk_data
            )
        else:
            # Manual reassembly: store each chunk as a separate object
            from uuid import uuid4
            temp_doc_id = UUID(session.extra_metadata.get("temp_doc_id"))
            chunk_path = await storage_service.upload_file(
                temp_doc_id,
                f"{session.filename}.chunk_{chunk_number}",
                chunk_data,
                session.content_type
            )
            if "chunk_paths" not in session.extra_metadata:
                session.extra_metadata["chunk_paths"] = {}
            session.extra_metadata["chunk_paths"][str(chunk_number)] = chunk_path
            flag_modified(session, "extra_metadata")
            etag = f"chunk_{chunk_number}"  # Placeholder ETag for manual reassembly
        
        # Update session
        if chunk_number not in session.uploaded_chunks:
            session.uploaded_chunks.append(chunk_number)
            # Flag JSON column as modified so SQLAlchemy detects the change
            flag_modified(session, "uploaded_chunks")
            session.uploaded_bytes += chunk_size
        
        # Store part ETag (for MinIO multipart) or chunk info (for manual reassembly)
        if not session.minio_part_etags:
            session.minio_part_etags = {}
        if not session.minio_upload_id.startswith("upload_"):
            # MinIO multipart upload
            part_number = chunk_number + 1
            session.minio_part_etags[str(part_number)] = etag
        else:
            # Manual reassembly - store chunk info in part_etags for consistency
            session.minio_part_etags[str(chunk_number)] = etag
        
        # Flag JSON columns as modified
        flag_modified(session, "minio_part_etags")
        if session.extra_metadata:
            flag_modified(session, "extra_metadata")
        
        await db.commit()
        await db.refresh(session)
        
        logger.info(f"Uploaded chunk {chunk_number}/{session.total_chunks - 1} for session {session_id} ({session.uploaded_bytes}/{session.file_size} bytes)")
        
        return {
            "session_id": str(session.id),
            "chunk_number": chunk_number,
            "uploaded_bytes": session.uploaded_bytes,
            "progress": session.progress_percentage,
            "status": "uploaded"
        }
    
    except ValidationError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading chunk: {e}", exc_info=True)
        # Mark session as failed (only if session was successfully retrieved)
        try:
            # Check if session exists before trying to update it
            if 'session' in locals() and session is not None:
                session.status = "failed"
                session.error_message = str(e)
                await db.commit()
        except Exception as db_error:
            logger.warning(f"Failed to update session status: {db_error}")
        raise HTTPException(status_code=500, detail=f"Failed to upload chunk: {str(e)}")


@router.post("/{session_id}/complete")
async def complete_upload(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete the chunked upload and create the document.
    """
    try:
        # Get upload session
        result = await db.execute(
            select(UploadSession).where(
                UploadSession.id == session_id,
                UploadSession.user_id == current_user.id
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        if session.status == "completed":
            if session.document_id:
                return {
                    "message": "Upload already completed",
                    "document_id": str(session.document_id),
                    "session_id": str(session.id)
                }
            else:
                raise HTTPException(status_code=400, detail="Upload marked as completed but no document ID found")
        
        # Verify all chunks are uploaded
        if len(session.uploaded_chunks) != session.total_chunks:
            missing = set(range(session.total_chunks)) - set(session.uploaded_chunks)
            raise HTTPException(
                status_code=400,
                detail=f"Not all chunks uploaded. Missing chunks: {sorted(missing)}"
            )
        
        # Complete multipart upload or reassemble chunks
        object_path = session.extra_metadata.get("temp_object_path")
        if not object_path or not session.minio_upload_id:
            raise HTTPException(status_code=400, detail="Multipart upload not initialized")
        
        # Check if we're using MinIO multipart upload or manual reassembly
        use_multipart = session.extra_metadata.get("use_multipart", False)
        if use_multipart and not session.minio_upload_id.startswith("upload_"):
            # Use MinIO multipart upload (automatic reassembly)
            # Prepare parts list
            parts = [
                (int(part_num), etag)
                for part_num, etag in session.minio_part_etags.items()
            ]
            parts.sort(key=lambda x: x[0])  # Sort by part number
            
            await storage_service.complete_multipart_upload(
                object_path,
                session.minio_upload_id,
                parts
            )
            
            logger.info(f"Completed MinIO multipart upload for session {session_id} - file automatically reassembled")
        else:
            # Manual reassembly: download all chunks and concatenate them
            logger.info(f"Reassembling file manually from chunks for session {session_id}")
            import tempfile
            import asyncio
            
            temp_file_path = tempfile.mktemp()
            try:
                # Download all chunks in order and concatenate
                with open(temp_file_path, 'wb') as final_file:
                    for chunk_idx in range(session.total_chunks):
                        chunk_path = session.extra_metadata.get("chunk_paths", {}).get(str(chunk_idx))
                        if not chunk_path:
                            raise HTTPException(status_code=400, detail=f"Chunk {chunk_idx} path not found")
                        
                        # Download chunk to temp file
                        chunk_temp_path = tempfile.mktemp()
                        try:
                            await storage_service.download_file(chunk_path, chunk_temp_path)
                            # Append to final file
                            with open(chunk_temp_path, 'rb') as chunk_file:
                                final_file.write(chunk_file.read())
                        finally:
                            if os.path.exists(chunk_temp_path):
                                os.unlink(chunk_temp_path)
                
                # Upload the reassembled file
                with open(temp_file_path, 'rb') as f:
                    reassembled_data = f.read()
                
                # Upload to final location
                await storage_service.upload_file(
                    UUID(session.extra_metadata.get("temp_doc_id")),
                    session.filename,
                    reassembled_data,
                    session.content_type
                )
                
                # Clean up chunk files
                for chunk_path in session.extra_metadata.get("chunk_paths", {}).values():
                    try:
                        await storage_service.delete_file(chunk_path)
                    except:
                        pass  # Ignore errors when cleaning up chunks
                
                logger.info(f"Manually reassembled file for session {session_id}")
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Now create the document
        # We need to create a document record and move the file to the final location
        from app.services.document_service import DocumentService
        from app.models.document import Document
        from app.models.document import DocumentSource
        from sqlalchemy import select as sql_select, and_
        import hashlib
        
        # Get or create upload source
        upload_source_result = await db.execute(
            sql_select(DocumentSource).where(DocumentSource.name == "File Upload")
        )
        upload_source = upload_source_result.scalar_one_or_none()
        
        if not upload_source:
            upload_source = DocumentSource(
                name="File Upload",
                source_type="file",
                config={},
                is_active=True
            )
            db.add(upload_source)
            await db.commit()
            await db.refresh(upload_source)
        
        # Download the file to calculate hash (we need content for hash)
        # Actually, we can't easily get the hash without downloading, so we'll use a placeholder
        # Or we can download it temporarily
        import tempfile
        temp_file_path = tempfile.mktemp()
        try:
            await storage_service.download_file(object_path, temp_file_path)
            with open(temp_file_path, 'rb') as f:
                content = f.read()
            content_hash = hashlib.sha256(content).hexdigest()
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Create document
        document = Document(
            title=session.title or session.filename,
            content="",  # Will be filled by transcription or text extraction
            content_hash=content_hash,
            file_path=object_path,  # Use the temp path for now
            file_type=session.file_type,
            file_size=session.file_size,
            source_id=upload_source.id,
            source_identifier=session.filename,
            tags=session.tags,
            extra_metadata={
                "original_filename": session.filename,
                "content_type": session.content_type,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "upload_session_id": str(session.id)
            }
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Move file to final location
        final_object_path = storage_service._get_object_path(document.id, session.filename)
        if object_path != final_object_path:
            # Copy file to final location (MinIO doesn't have move, so we copy and delete)
            logger.info(f"Copying file from temp path {object_path} to final path {final_object_path}")
            copy_success = await storage_service.copy_file(
                source_path=object_path,
                dest_path=final_object_path,
                content_type=session.content_type
            )
            if not copy_success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to copy file from temporary location to final location"
                )
            
            # Delete the temporary file
            try:
                await storage_service.delete_file(object_path)
                logger.info(f"Deleted temporary file at {object_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file at {object_path}: {e}")
            
            # Update document with final path
            document.file_path = final_object_path
            await db.commit()
            logger.info(f"Updated document file_path to {final_object_path}")
        
        # Update session
        session.status = "completed"
        session.document_id = document.id
        await db.commit()
        
        # Check if file needs transcription
        from app.services.transcription_service import get_transcription_service
        transcription_service = get_transcription_service()
        from pathlib import Path
        
        if transcription_service:
            # Download file temporarily to check format (preserve extension for detection)
            # Use the final stored filename's extension so is_supported_format works correctly
            final_ext = Path(document.file_path).suffix or (Path(session.filename).suffix if 'session' in locals() else '')
            try:
                temp_named = tempfile.NamedTemporaryFile(delete=False, suffix=final_ext or '.tmp')
                temp_named.close()
                temp_check_path = temp_named.name
                # Use the final stored path on the document
                await storage_service.download_file(document.file_path, temp_check_path)
                file_path_obj = Path(temp_check_path)
                
                if transcription_service.is_supported_format(file_path_obj):
                    # Video/audio file
                    logger.info(f"Video/audio file detected: {session.filename}. Scheduling tasks.")
                    document.content = ""
                    document.extra_metadata = document.extra_metadata or {}

                    # Determine if we should transcode first (non-MP4 video)
                    try:
                        ext = (Path(session.filename).suffix or "").lower()
                    except Exception:
                        ext = ""
                    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
                    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
                    is_mp4 = ext == ".mp4" or (session.content_type == "video/mp4")
                    is_video = ext in video_exts or (session.file_type or "").startswith("video/")
                    is_audio = ext in audio_exts or (session.file_type or "").startswith("audio/")

                    if is_video and not is_mp4:
                        # Transcode first, then transcribe (done in transcode task)
                        document.extra_metadata["is_transcoding"] = True
                        document.extra_metadata.pop("is_transcribing", None)
                        # Publish status for UI via WebSocket
                        try:
                            from app.tasks.transcode_tasks import _publish_status as publish_status
                            publish_status(str(document.id), {
                                "is_transcoding": True,
                                "is_transcribing": False,
                                "is_transcribed": False,
                            })
                        except Exception:
                            pass
                        from app.tasks.transcode_tasks import transcode_to_mp4
                        transcode_to_mp4.delay(str(document.id))
                        logger.info(f"Triggered transcode-then-transcribe for document {document.id}")
                    else:
                        # No transcode needed (mp4 video or audio) -> transcribe now
                        document.extra_metadata["is_transcribing"] = True
                        try:
                            from app.tasks.transcription_tasks import _publish_status as publish_status
                            publish_status(str(document.id), {
                                "is_transcoding": False,
                                "is_transcribing": True,
                                "is_transcribed": False,
                            })
                        except Exception:
                            pass
                        from app.tasks.transcription_tasks import transcribe_document
                        transcribe_document.delay(str(document.id))
                        logger.info(f"Triggered transcription task for document {document.id}")
                else:
                    # Regular document - extract text
                    from app.services.text_processor import TextProcessor
                    text_processor = TextProcessor()
                    text_content = await text_processor.extract_text(temp_check_path, session.content_type)
                    document.content = text_content
                    
                    # Process document
                    await document_service._process_document_async(document, db)
                
                await db.commit()
            finally:
                if os.path.exists(temp_check_path):
                    os.unlink(temp_check_path)
        
        logger.info(f"Completed upload session {session_id}, created document {document.id}")
        
        return {
            "message": "Upload completed successfully",
            "document_id": str(document.id),
            "session_id": str(session.id)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing upload: {e}", exc_info=True)
        # Mark session as failed (only if session was successfully retrieved)
        try:
            # Check if session exists before trying to update it
            if 'session' in locals() and session is not None:
                session.status = "failed"
                session.error_message = str(e)
                await db.commit()
        except Exception as db_error:
            logger.warning(f"Failed to update session status: {db_error}")
        raise HTTPException(status_code=500, detail=f"Failed to complete upload: {str(e)}")


@router.delete("/{session_id}")
async def cancel_upload(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Cancel/abort an upload session."""
    result = await db.execute(
        select(UploadSession).where(
            UploadSession.id == session_id,
            UploadSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    # Abort multipart upload if in progress
    if session.minio_upload_id and session.status == "uploading":
        try:
            object_path = session.extra_metadata.get("temp_object_path")
            if object_path:
                await storage_service.abort_multipart_upload(
                    object_path,
                    session.minio_upload_id
                )
        except Exception as e:
            logger.warning(f"Failed to abort multipart upload: {e}")
    
    # Delete session
    await db.delete(session)
    await db.commit()
    
    return {"message": "Upload session cancelled"}
