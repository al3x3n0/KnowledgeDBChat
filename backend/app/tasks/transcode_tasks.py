"""
Background task to transcode uploaded videos to MP4 (H.264/AAC) for broad browser support.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from uuid import UUID

from loguru import logger
import redis
import json
from app.core.config import settings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery import celery_app
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.storage_service import storage_service


@celery_app.task(bind=True, name="app.tasks.transcode_tasks.transcode_to_mp4")
def transcode_to_mp4(self, document_id: str) -> Dict[str, Any]:
    """Celery entrypoint: transcode document's video to MP4.

    Args:
        document_id: Document UUID string
    """
    return asyncio.run(_async_transcode_to_mp4(self, document_id))


async def _async_transcode_to_mp4(task, document_id: str) -> Dict[str, Any]:
    async with AsyncSessionLocal() as db:
        # Fetch document
        result = await db.execute(select(Document).where(Document.id == UUID(document_id)))
        document = result.scalar_one_or_none()
        if not document:
            logger.warning(f"Transcode: document {document_id} not found")
            return {"success": False, "error": "document_not_found", "document_id": document_id}

        if not document.file_path:
            logger.warning(f"Transcode: document {document_id} has no file_path")
            return {"success": False, "error": "no_file_path", "document_id": document_id}

        # Skip if already transcoded
        meta = document.extra_metadata or {}
        if meta.get("is_transcoded") and meta.get("stream_file_path"):
            logger.info(f"Transcode: document {document_id} already transcoded -> {meta.get('stream_file_path')}")
            return {"success": True, "document_id": document_id, "skipped": True}

        # Determine if conversion is needed (non-mp4 video)
        ext = Path(document.file_path).suffix.lower()
        is_video = ext in {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
        if not is_video:
            logger.info(f"Transcode: document {document_id} is not a video (ext={ext}), skipping")
            return {"success": True, "document_id": document_id, "skipped": True}
        needs_transcode = ext != ".mp4"
        if not needs_transcode:
            logger.info(f"Transcode: document {document_id} already mp4, skipping")
            # Mark metadata accordingly
            document.extra_metadata = {**meta, "is_transcoded": True, "stream_file_path": document.file_path}
            await db.commit()
            return {"success": True, "document_id": document_id, "skipped": True}

        # Download original
        temp_src = None
        temp_dst = None
        try:
            temp_src = Path(tempfile.NamedTemporaryFile(delete=False, suffix=ext).name)
            temp_dst = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
            logger.info(f"Transcode: downloading {document.file_path} -> {temp_src}")
            _publish_progress(document_id, {"stage": "transcoding", "message": "Preparing...", "progress": 5})
            await storage_service.download_file(document.file_path, str(temp_src))

            # Run ffmpeg to H.264/AAC with browser-friendly settings
            # Use subprocess for reliable binary invocation
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(temp_src),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "128k",
                str(temp_dst),
            ]
            logger.info(f"Transcode: running {' '.join(cmd)}")
            _publish_progress(document_id, {"stage": "transcoding", "message": "Converting to MP4...", "progress": 15})
            subprocess.run(cmd, check=True)
            _publish_progress(document_id, {"stage": "transcoding", "message": "Uploading MP4...", "progress": 85})

            # Upload mp4
            new_filename = f"{Path(document.file_path).stem}.mp4"
            with open(temp_dst, "rb") as f:
                data = f.read()
            new_object_path = await storage_service.upload_file(UUID(str(document.id)), new_filename, data, "video/mp4")
            logger.info(f"Transcode: uploaded MP4 to {new_object_path}")

            # Update document to point to MP4 for streaming, preserve original
            document.extra_metadata = {
                **meta,
                "original_file_path": meta.get("original_file_path") or document.file_path,
                "stream_file_path": new_object_path,
                "is_transcoding": False,
                "is_transcoded": True,
            }
            document.file_path = new_object_path
            document.file_type = "video/mp4"
            # Mark and start transcription now that MP4 is ready
            # Update and publish status flags
            document.extra_metadata["is_transcribing"] = True
            await db.commit()
            await db.refresh(document)
            _publish_progress(document_id, {"stage": "transcoding", "message": "Transcode complete. Starting transcription...", "progress": 100})
            _publish_status(document_id, {
                "is_transcoding": False,
                "is_transcoded": True,
                "is_transcribing": True,
            })

            # Dispatch transcription task
            try:
                from app.tasks.transcription_tasks import transcribe_document
                transcribe_document.delay(str(document.id))
                logger.info(f"Transcode: dispatched transcription for document {document_id}")
            except Exception as e:
                logger.error(f"Transcode: failed to dispatch transcription for {document_id}: {e}")
                _publish_error(document_id, f"dispatch_transcription_failed: {e}")

            return {"success": True, "document_id": document_id, "stream_file_path": new_object_path}

        except subprocess.CalledProcessError as e:  # type: ignore[name-defined]
            logger.error(f"Transcode failed for {document_id}: {e}")
            document.extra_metadata = {**meta, "is_transcoding": False, "is_transcoded": False, "transcode_error": str(e)}
            await db.commit()
            _publish_error(document_id, "ffmpeg_failed")
            return {"success": False, "document_id": document_id, "error": "ffmpeg_failed"}
        except Exception as e:
            logger.error(f"Transcode error for {document_id}: {e}")
            document.extra_metadata = {**meta, "is_transcoding": False, "is_transcoded": False, "transcode_error": str(e)}
            await db.commit()
            _publish_error(document_id, str(e))
            _publish_status(document_id, {
                "is_transcoding": False,
                "is_transcoded": False,
                "is_transcribing": False,
            })
            return {"success": False, "document_id": document_id, "error": str(e)}
        finally:
            for p in (temp_src, temp_dst):
                try:
                    if p and Path(p).exists():
                        os.unlink(p)
                except Exception:
                    pass


def _get_redis_client():
    try:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Transcode: Failed to connect to Redis for progress: {e}")
        return None


def _publish_progress(document_id: str, progress: dict):
    try:
        rc = _get_redis_client()
        if not rc:
            return
        channel = f"transcription_progress:{document_id}"
        message = json.dumps({"type": "progress", "document_id": document_id, "progress": progress})
        rc.publish(channel, message)
    except Exception as e:
        logger.warning(f"Transcode: failed to publish progress: {e}")


def _publish_complete(document_id: str, result: dict):
    try:
        rc = _get_redis_client()
        if not rc:
            return
        channel = f"transcription_progress:{document_id}"
        message = json.dumps({"type": "complete", "document_id": document_id, "result": result})
        rc.publish(channel, message)
    except Exception as e:
        logger.warning(f"Transcode: failed to publish complete: {e}")


def _publish_error(document_id: str, error: str):
    try:
        rc = _get_redis_client()
        if not rc:
            return
        channel = f"transcription_progress:{document_id}"
        message = json.dumps({"type": "error", "document_id": document_id, "error": error})
        rc.publish(channel, message)
    except Exception as e:
        logger.warning(f"Transcode: failed to publish error: {e}")


def _publish_status(document_id: str, status: dict):
    try:
        rc = _get_redis_client()
        if not rc:
            return
        channel = f"transcription_progress:{document_id}"
        message = json.dumps({"type": "status", "document_id": document_id, "status": status})
        rc.publish(channel, message)
    except Exception as e:
        logger.warning(f"Transcode: failed to publish status: {e}")
