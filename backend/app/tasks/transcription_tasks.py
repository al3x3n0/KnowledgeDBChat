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
from celery import Task
from app.core.database import create_celery_session
from app.core.config import settings
import hashlib
from app.models.document import Document
from app.services.transcription_service import get_transcription_service
from app.services.storage_service import storage_service
from app.services.persona_service import persona_service
from sqlalchemy import select
import tempfile
import os
import redis
from datetime import timedelta


def _format_timecode(seconds: float) -> str:
    try:
        if seconds is None:
            seconds = 0
        seconds = max(0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"


def _format_transcript_with_timecodes(segments: list, fallback_text: str) -> str:
    """Create a human-friendly transcript with one timecoded line per segment."""
    lines = []
    try:
        for seg in segments or []:
            start = seg.get('start', 0)
            text = (seg.get('text') or '').strip()
            if not text:
                continue
            tc = _format_timecode(start)
            lines.append(f"[{tc}] {text}")
    except Exception:
        pass
    # Ensure each timecode is on its own line
    formatted = "\n".join(lines).strip()
    return formatted if formatted else (fallback_text or '')


def _format_transcript_from_sentences(sentences: list, fallback_text: str) -> str:
    """Format transcript from sentence-level items with speakers and start times.

    sentences: list of {'start': seconds, 'text': str, 'speaker': optional}
    """
    try:
        lines = []
        for item in sentences or []:
            start = item.get('start', 0)
            text = (item.get('text') or '').strip()
            if not text:
                continue
            tc = _format_timecode(start)
            speaker = item.get('speaker')
            end = item.get('end')
            if end is not None:
                tc_end = _format_timecode(end)
                prefix = f"[{tc} - {tc_end}]"
            else:
                prefix = f"[{tc}]"
            if speaker:
                lines.append(f"{prefix} [{speaker}] {text}")
            else:
                lines.append(f"{prefix} {text}")
        formatted = "\n".join(lines).strip()
        return formatted if formatted else (fallback_text or '')
    except Exception:
        return fallback_text or ''


class TranscribeTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo=None):
        try:
            document_id = args[0] if args else None
            if not document_id:
                return
            async def _mark_failed(doc_id: str, error_text: str):
                async with create_celery_session()() as db:
                    try:
                        from sqlalchemy import select as _select
                        result = await db.execute(_select(Document).where(Document.id == UUID(doc_id)))
                        doc = result.scalar_one_or_none()
                        if not doc:
                            return
                        doc.extra_metadata = (doc.extra_metadata or {})
                        doc.extra_metadata.update({
                            "transcription_error": error_text or "Transcription failed",
                            "is_transcribing": False,
                            "is_transcribed": False,
                        })
                        doc.processing_error = f"Transcription failed: {error_text}"
                        await db.commit()
                    except Exception as _e:
                        logger.warning(f"Failed to mark transcription failure for {doc_id}: {_e}")
            import asyncio
            error_text = str(exc) if exc else "Worker lost"
            asyncio.run(_mark_failed(document_id, error_text))
            # Publish websocket notifications
            try:
                _publish_error(str(document_id), error_text)
                _publish_status(str(document_id), {"is_transcribing": False})
            except Exception:
                pass
        except Exception as ee:
            logger.warning(f"on_failure handler error: {ee}")


@celery_app.task(bind=True, base=TranscribeTask, name="app.tasks.transcription_tasks.transcribe_document")
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
    async with create_celery_session()() as db:
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
                        seg = {
                            'start': progress_dict.get('start'),
                            'text': progress_dict.get('text'),
                        }
                        if 'speaker' in progress_dict and progress_dict.get('speaker'):
                            seg['speaker'] = progress_dict.get('speaker')
                        _publish_segment(document_id, seg)
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
                
                # Prefer diarization-driven sentence segments if available
                sentence_segs = metadata.get('sentence_segments') or []
                if sentence_segs:
                    formatted_transcript = _format_transcript_from_sentences(sentence_segs, transcript_text)
                else:
                    formatted_transcript = _format_transcript_with_timecodes(metadata.get('segments', []), transcript_text)
                
                # Create a separate transcript document linked to this video/audio
                # Keep metadata on original for UI (segments, flags), but index a new text document
                base_title = (document.title or '').rsplit('.', 1)[0] or (document.title or '')
                transcript_title = f"{base_title} (Transcript)"
                content_bytes = formatted_transcript.encode('utf-8')
                content_hash = hashlib.sha256(content_bytes).hexdigest()

                transcript_doc = Document(
                    title=transcript_title,
                    content=formatted_transcript,
                    content_hash=content_hash,
                    url=None,
                    file_path=None,
                    file_type="text/plain",
                    file_size=len(content_bytes),
                    source_id=document.source_id,
                    source_identifier=f"{document.source_identifier}:transcript",
                    author=document.author,
                    owner_persona_id=document.owner_persona_id,
                    tags=document.tags,
                    extra_metadata={
                        "doc_type": "transcript",
                        "parent_document_id": str(document.id),
                        "transcription_metadata": metadata,
                    },
                )
                db.add(transcript_doc)
                
                # Update original document flags and link to transcript
                document.extra_metadata = document.extra_metadata or {}
                document.extra_metadata.update({
                    "transcription_metadata": metadata,
                    "is_transcribed": True,
                    "is_transcribing": False,
                    "transcript_document_id": None,  # set after flush
                })

                await db.commit()
                await db.refresh(transcript_doc)
                
                # Now set the link id
                document.extra_metadata["transcript_document_id"] = str(transcript_doc.id)
                await db.commit()
                await db.refresh(document)

                try:
                    await persona_service.record_sentence_speakers(
                        db,
                        document=document,
                        sentence_segments=sentence_segs,
                        base_document_id=document.id,
                    )
                    await persona_service.record_sentence_speakers(
                        db,
                        document=transcript_doc,
                        sentence_segments=sentence_segs,
                        base_document_id=document.id,
                    )
                    await db.commit()
                except Exception as persona_err:
                    logger.debug(f"Failed to persist diarization personas for document {document_id}: {persona_err}")
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
                
                # Now process the transcript document for indexing (chunking, embedding, etc.)
                from app.services.document_service import DocumentService
                document_service = DocumentService()
                await document_service._process_document_async(transcript_doc, db)
                
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


