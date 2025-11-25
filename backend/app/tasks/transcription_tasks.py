"""
Background tasks for video/audio transcription.
"""

import asyncio
import json
from typing import Dict, Any, Callable
import traceback
from uuid import UUID
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.celery import celery_app
from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.models.document import Document
from app.services.transcription_service import get_transcription_service
from app.services.storage_service import storage_service
from sqlalchemy import select
import tempfile
import os
import redis


@celery_app.task(bind=True, name="app.tasks.transcription_tasks.transcribe_document")
def transcribe_document(self, document_id: str) -> Dict[str, Any]:
    """
    Transcribe a video/audio document and update its content.
    
    Args:
        document_id: UUID of the document to transcribe
        
    Returns:
        Dict with transcription results
    """
    return asyncio.run(_async_transcribe_document(self, document_id))


async def _async_transcribe_document(task, document_id: str) -> Dict[str, Any]:
    """Async implementation of document transcription."""
    async with AsyncSessionLocal() as db:
        try:
            logger.info(f"Starting transcription for document {document_id}")
            
            # Get the document
            result = await db.execute(
                select(Document).where(Document.id == UUID(document_id))
            )
            document = result.scalar_one_or_none()
            
            if not document:
                logger.warning(f"Document {document_id} not found for transcription")
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "Document not found"
                }
            
            # Check if document is a video/audio file
            transcription_service = get_transcription_service()
            if not transcription_service:
                logger.warning("Transcription service not available")
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "Transcription service not available"
                }

            # If it's a non-MP4 video and not yet transcoded, reroute to transcode first
            try:
                from pathlib import Path as _Path
                ext = (_Path(document.file_path or "").suffix or "").lower()
                video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
                is_video = (document.file_type or "").startswith("video/") or ext in video_exts
                is_mp4 = (ext == ".mp4") or (document.file_type == "video/mp4")
                is_transcoded = bool((document.extra_metadata or {}).get("is_transcoded"))
            except Exception:
                is_video, is_mp4, is_transcoded = True, False, False
            if is_video and not is_mp4 and not is_transcoded:
                logger.info(f"Document {document_id} is non-MP4 video; scheduling transcode before transcription")
                # Flip flags and publish status
                document.extra_metadata = document.extra_metadata or {}
                document.extra_metadata["is_transcribing"] = False
                document.extra_metadata["is_transcoding"] = True
                await db.commit()
                _publish_status(document_id, {
                    "is_transcribing": False,
                    "is_transcoding": True,
                })
                try:
                    from app.tasks.transcode_tasks import transcode_to_mp4
                    transcode_to_mp4.delay(str(document.id))
                except Exception as e:
                    logger.error(f"Failed to dispatch transcode from transcription guard: {e}")
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "transcode_required_first"
                }
            
            # Get file from MinIO
            if not document.file_path:
                logger.warning(f"Document {document_id} has no file_path")
                return {
                    "document_id": document_id,
                    "success": False,
                    "error": "Document has no file path"
                }
            
            # Download file from MinIO to temporary location
            temp_file_path = None
            try:
                # Create temporary file
                file_ext = Path(document.file_path).suffix or ".tmp"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                temp_file_path = Path(temp_file.name)
                temp_file.close()
                
                # Download from MinIO
                logger.info(f"Downloading file from MinIO: {document.file_path}")
                
                # Publish download progress
                _publish_progress(document_id, {
                    "stage": "downloading",
                    "message": "Downloading file from storage...",
                    "progress": 5
                })
                
                await storage_service.download_file(document.file_path, str(temp_file_path))
                
                # Check if file format is supported
                if not transcription_service.is_supported_format(temp_file_path):
                    logger.warning(f"File format not supported for transcription: {document.file_path}")
                    _publish_progress(document_id, {
                        "stage": "error",
                        "message": "File format not supported",
                        "progress": 0,
                        "error": "File format not supported for transcription"
                    })
                    return {
                        "document_id": document_id,
                        "success": False,
                        "error": "File format not supported for transcription"
                    }
                
                # Create progress callback
                def progress_callback(progress_dict: dict):
                    """Callback to publish transcription progress or segments."""
                    if progress_dict and progress_dict.get('type') == 'segment':
                        _publish_segment(document_id, {
                            'start': progress_dict.get('start'),
                            'text': progress_dict.get('text'),
                        })
                    else:
                        _publish_progress(document_id, progress_dict)
                
                # Transcribe the file
                logger.info(f"Transcribing file: {temp_file_path}")
                transcript_text, metadata = transcription_service.transcribe_file(
                    temp_file_path,
                    language=settings.TRANSCRIPTION_LANGUAGE,
                    progress_callback=progress_callback
                )
                
                if not transcript_text:
                    logger.warning(f"No transcript generated for document {document_id}")
                    return {
                        "document_id": document_id,
                        "success": False,
                        "error": "No transcript generated"
                    }
                
                # Publish processing progress
                _publish_progress(document_id, {
                    "stage": "saving",
                    "message": "Saving transcript...",
                    "progress": 98
                })
                
                # Format transcript with time codes
                formatted_transcript = _format_transcript_with_timecodes(metadata.get('segments', []), transcript_text)
                
                # Update document with formatted transcript
                document.content = formatted_transcript
                document.extra_metadata = document.extra_metadata or {}
                document.extra_metadata.update({
                    "transcription_metadata": metadata,
                    "is_transcribed": True,
                    "is_transcribing": False
                })
                
                await db.commit()
                await db.refresh(document)
                # Publish status update for UI
                _publish_status(document_id, {
                    "is_transcribing": False,
                    "is_transcribed": True,
                })
                
                logger.info(f"Transcription completed for document {document_id}. Text length: {len(transcript_text)} chars")
                
                # Publish indexing progress
                _publish_progress(document_id, {
                    "stage": "indexing",
                    "message": "Indexing document...",
                    "progress": 99
                })
                
                # Now process the document for indexing (chunking, embedding, etc.)
                from app.services.document_service import DocumentService
                document_service = DocumentService()
                await document_service._process_document_async(document, db)
                
                # Publish completion
                result = {
                    "document_id": document_id,
                    "success": True,
                    "transcript_length": len(transcript_text),
                    "duration": metadata.get("duration", 0)
                }
                _publish_complete(document_id, result)
                
                return result
                
            finally:
                # Clean up temporary file
                if temp_file_path and temp_file_path.exists():
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"Cleaned up temp file: {temp_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error transcribing document {document_id}: {e}", exc_info=True)
            # Persist error on document
            try:
                result = await db.execute(
                    select(Document).where(Document.id == UUID(document_id))
                )
                doc = result.scalar_one_or_none()
                if doc:
                    tb = traceback.format_exc()
                    doc.extra_metadata = (doc.extra_metadata or {})
                    doc.extra_metadata.update({
                        "transcription_error": str(e),
                        "transcription_traceback": tb,
                        "is_transcribing": False,
                        "is_transcribed": False,
                    })
                    # Also set generic processing_error for UI
                    doc.processing_error = f"Transcription failed: {e}"
                    await db.commit()
                    _publish_status(document_id, {
                        "is_transcribing": False,
                        "is_transcoded": doc.extra_metadata.get("is_transcoded", False),
                    })
            except Exception as save_err:
                logger.warning(f"Failed to persist transcription error for {document_id}: {save_err}")
            _publish_error(document_id, str(e))
            return {
                "document_id": document_id,
                "success": False,
                "error": str(e)
            }


def _get_redis_client():
    """Get Redis client for pub/sub."""
    try:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for progress updates: {e}")
        return None


def _publish_progress(document_id: str, progress: dict):
    """Publish transcription progress to Redis."""
    try:
        redis_client = _get_redis_client()
        if redis_client:
            channel = f"transcription_progress:{document_id}"
            message = json.dumps({
                "type": "progress",
                "document_id": document_id,
                "progress": progress
            })
            redis_client.publish(channel, message)
    except Exception as e:
        logger.warning(f"Failed to publish progress: {e}")


def _publish_complete(document_id: str, result: dict):
    """Publish transcription completion to Redis."""
    try:
        redis_client = _get_redis_client()
        if redis_client:
            channel = f"transcription_progress:{document_id}"
            message = json.dumps({
                "type": "complete",
                "document_id": document_id,
                "result": result
            })
            redis_client.publish(channel, message)
    except Exception as e:
        logger.warning(f"Failed to publish completion: {e}")


def _publish_status(document_id: str, status: dict):
    """Publish document status flags via Redis (for WebSocket)."""
    try:
        redis_client = _get_redis_client()
        if redis_client:
            channel = f"transcription_progress:{document_id}"
            message = json.dumps({
                "type": "status",
                "document_id": document_id,
                "status": status,
            })
            redis_client.publish(channel, message)
    except Exception as e:
        logger.warning(f"Failed to publish status: {e}")


def _publish_segment(document_id: str, segment: dict):
    """Publish a partial transcription segment to Redis."""
    try:
        redis_client = _get_redis_client()
        if redis_client:
            channel = f"transcription_progress:{document_id}"
            message = json.dumps({
                "type": "segment",
                "document_id": document_id,
                "segment": segment,
            })
            redis_client.publish(channel, message)
    except Exception as e:
        logger.warning(f"Failed to publish segment: {e}")


def _publish_error(document_id: str, error: str):
    """Publish transcription error to Redis."""
    try:
        redis_client = _get_redis_client()
        if redis_client:
            channel = f"transcription_progress:{document_id}"
            message = json.dumps({
                "type": "error",
                "document_id": document_id,
                "error": error
            })
            redis_client.publish(channel, message)
    except Exception as e:
        logger.warning(f"Failed to publish error: {e}")


def _format_timecode(seconds: float) -> str:
    """Format seconds to timecode string (HH:MM:SS or MM:SS)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def _format_transcript_with_timecodes(segments: list, plain_text: str) -> str:
    """
    Format transcript text with time codes.
    
    Args:
        segments: List of segment dictionaries with 'start', 'end', 'text'
        plain_text: Plain transcript text (fallback if segments unavailable)
        
    Returns:
        Formatted transcript with time codes
    """
    if not segments:
        # Fallback to plain text if no segments
        return plain_text
    
    formatted_lines = []
    for segment in segments:
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '').strip()
        
        if not text:
            continue
        
        # Format timecode
        start_tc = _format_timecode(start_time)
        end_tc = _format_timecode(end_time)
        
        # Add speaker label if available
        speaker = segment.get('speaker')
        speaker_label = f"[{speaker}] " if speaker else ""
        
        # Format: [00:15 - 00:23] [Speaker] Text here...
        formatted_line = f"[{start_tc} - {end_tc}] {speaker_label}{text}"
        formatted_lines.append(formatted_line)
    
    return "\n\n".join(formatted_lines)
