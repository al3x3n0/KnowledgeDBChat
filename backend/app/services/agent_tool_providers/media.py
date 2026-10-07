"""Autonomous-job tools: the ``media`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_media_provider(executor: Any) -> FunctionToolProvider:
    """Media ingestion and analysis tools for AutonomousAgentExecutor."""

    async def _transcribe_document(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.document import Document as DocModel
        from app.services.job_dispatch import send_now

        doc_id = (params.get("document_id") or "").strip()
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        try:
            doc_result = await ctx.db.execute(
                # Documents have no owner column: the knowledge base is
                # shared. Filtering on one raised AttributeError on every call.
                select(DocModel).where(DocModel.id == _UUID(doc_id))
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return {"error": f"Document {doc_id} not found"}
            if not doc.file_path:
                return {"error": "Document has no associated file"}
            meta = doc.extra_metadata or {}
            if meta.get("is_transcribed"):
                return {
                    "success": True,
                    "data": {
                        "document_id": doc_id,
                        "status": "already_transcribed",
                        "transcript_document_id": meta.get("transcript_document_id"),
                    },
                }
            if meta.get("is_transcribing"):
                return {
                    "success": True,
                    "data": {"document_id": doc_id, "status": "in_progress"},
                }
            ft = (doc.file_type or "").lower()
            from pathlib import Path as _Path

            ext = _Path(doc.file_path).suffix.lower()
            av_exts = {
                ".mp3",
                ".mp4",
                ".wav",
                ".m4a",
                ".ogg",
                ".flac",
                ".aac",
                ".avi",
                ".mkv",
                ".mov",
                ".webm",
                ".flv",
                ".wmv",
            }
            is_av = (
                any(ft.startswith(p) for p in ("audio/", "video/")) or ext in av_exts
            )
            if not is_av:
                return {"error": f"Document is not audio/video (type={ft}, ext={ext})"}
            # Queue first. With the flag committed before the enqueue, a
            # broker that was down left the document "in progress" for ever
            # and every retry was told so.
            celery_result = send_now(
                "app.tasks.transcription_tasks.transcribe_document", str(doc.id)
            )
            doc.extra_metadata = {**meta, "is_transcribing": True}
            flag_modified(doc, "extra_metadata")
            await ctx.db.commit()
            return {
                "success": True,
                "data": {
                    "document_id": doc_id,
                    "status": "dispatched",
                    "task_id": celery_result.id,
                    "title": doc.title,
                },
                "findings": [
                    {
                        "type": "transcription_started",
                        "title": f"Transcription started for {doc.title}",
                        "document_id": doc_id,
                        "task_id": celery_result.id,
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Failed to transcribe document: {exc}"}

    async def _analyze_image(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import base64
        from pathlib import Path as _Path
        from uuid import UUID as _UUID

        import httpx

        from app.core.config import settings as _settings
        from app.models.document import Document as DocModel
        from app.services.storage_service import storage_service as _storage

        doc_id = (params.get("document_id") or "").strip()
        prompt_text = (
            params.get("prompt") or ""
        ).strip() or "Describe this image in detail, including any text, diagrams, charts, or notable visual elements."
        vision_model = (params.get("model") or "").strip() or (
            getattr(_settings, "VISION_MODEL", "llava") or "llava"
        )
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        try:
            doc_result = await ctx.db.execute(
                # Documents have no owner column: the knowledge base is
                # shared. Filtering on one raised AttributeError on every call.
                select(DocModel).where(DocModel.id == _UUID(doc_id))
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return {"error": f"Document {doc_id} not found"}
            if not doc.file_path:
                return {"error": "Document has no associated file"}
            ft = (doc.file_type or "").lower()
            ext = _Path(doc.file_path).suffix.lower()
            image_types = {
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/gif",
                "image/webp",
                "image/bmp",
                "image/tiff",
            }
            image_exts = {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".tiff",
                ".tif",
            }
            if ft not in image_types and ext not in image_exts:
                return {"error": f"Document is not an image (type={ft}, ext={ext})"}
            try:
                image_bytes = await _storage.get_file_content(doc.file_path)
            except Exception as storage_exc:
                return {
                    "error": f"Failed to download image {doc.file_path}: {storage_exc}"
                }
            if not image_bytes:
                return {"error": "Failed to download image: empty content"}
            if len(image_bytes) > 20 * 1024 * 1024:
                return {
                    "error": f"Image too large ({len(image_bytes) // (1024*1024)}MB). Max 20MB."
                }
            payload = {
                "model": vision_model,
                "prompt": prompt_text[:2000],
                "images": [base64.b64encode(image_bytes).decode("utf-8")],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048},
            }
            response = await executor.llm_service.client.post(
                f"{executor.llm_service.base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            analysis_text = (response.json().get("response") or "").strip()
            if not analysis_text:
                return {"error": "Vision model returned empty response"}
            return {
                "success": True,
                "data": {
                    "document_id": doc_id,
                    "title": doc.title,
                    "analysis": analysis_text[:5000],
                    "model": vision_model,
                    "prompt": prompt_text[:200],
                },
                "findings": [
                    {
                        "type": "image_analysis",
                        "title": f"Image analysis: {doc.title}",
                        "document_id": doc_id,
                        "content": analysis_text[:2000],
                        "model": vision_model,
                    }
                ],
            }
        except Exception as exc:
            error_msg = str(exc)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return {
                    "error": f"Vision model '{vision_model}' not available. Pull it with: ollama pull {vision_model}"
                }
            if isinstance(exc, httpx.ConnectError):
                # Image analysis is Ollama-only whatever LLM_PROVIDER says,
                # and the stack does not bundle Ollama. Say so: a bare
                # connection error reads as a transient fault worth retrying.
                return {
                    "error": (
                        "Image analysis needs an Ollama instance with a vision "
                        f"model, and none is reachable at "
                        f"{executor.llm_service.base_url} (OLLAMA_BASE_URL): "
                        f"{error_msg}"
                    )
                }
            return {"error": f"Failed to analyze image: {error_msg}"}

    async def _get_media_info(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from pathlib import Path as _Path
        from uuid import UUID as _UUID

        from app.models.document import Document as DocModel

        doc_id = (params.get("document_id") or "").strip()
        if not doc_id:
            return {"error": "Missing required parameter: document_id"}
        try:
            doc_result = await ctx.db.execute(
                # Documents have no owner column: the knowledge base is
                # shared. Filtering on one raised AttributeError on every call.
                select(DocModel).where(DocModel.id == _UUID(doc_id))
            )
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return {"error": f"Document {doc_id} not found"}
            if not doc.file_path:
                return {"error": "Document has no associated file"}
            ft = (doc.file_type or "").lower()
            ext = _Path(doc.file_path).suffix.lower()
            media_info = {
                "document_id": doc_id,
                "title": doc.title,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
            }
            av_exts = {
                ".mp3",
                ".mp4",
                ".wav",
                ".m4a",
                ".ogg",
                ".flac",
                ".aac",
                ".avi",
                ".mkv",
                ".mov",
                ".webm",
                ".flv",
                ".wmv",
            }
            image_exts = {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".bmp",
                ".tiff",
                ".tif",
            }
            is_av = (
                any(ft.startswith(p) for p in ("audio/", "video/")) or ext in av_exts
            )
            is_image = ft.startswith("image/") or ext in image_exts
            if is_av:
                import json
                import os
                import subprocess
                import tempfile

                from app.services.storage_service import storage_service as _storage

                temp_path = None
                try:
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=ext or ".tmp"
                    )
                    temp_path = tmp.name
                    tmp.close()
                    if not await _storage.download_file(doc.file_path, temp_path):
                        raise FileNotFoundError(
                            f"{doc.file_path} was not found in storage"
                        )
                    probe_result = await asyncio.to_thread(
                        subprocess.run,
                        [
                            "ffprobe",
                            "-v",
                            "quiet",
                            "-print_format",
                            "json",
                            "-show_format",
                            "-show_streams",
                            temp_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if probe_result.returncode == 0:
                        probe_data = json.loads(probe_result.stdout)
                        fmt = probe_data.get("format", {})
                        media_info["duration_seconds"] = float(fmt.get("duration", 0))
                        media_info["format_name"] = fmt.get("format_name")
                        media_info["bit_rate"] = int(fmt.get("bit_rate", 0) or 0)
                        for stream in probe_data.get("streams", []):
                            codec_type = stream.get("codec_type")
                            if codec_type == "video":
                                media_info["video_codec"] = stream.get("codec_name")
                                media_info["width"] = stream.get("width")
                                media_info["height"] = stream.get("height")
                                media_info["fps"] = stream.get("r_frame_rate")
                            elif codec_type == "audio":
                                media_info["audio_codec"] = stream.get("codec_name")
                                media_info["sample_rate"] = stream.get("sample_rate")
                                media_info["channels"] = stream.get("channels")
                    else:
                        media_info["probe_error"] = "ffprobe failed or not installed"
                    media_info["media_category"] = "audio_video"
                except Exception as probe_err:
                    media_info["probe_error"] = str(probe_err)
                    media_info["media_category"] = "audio_video"
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            elif is_image:
                try:
                    from io import BytesIO

                    from PIL import Image

                    from app.services.storage_service import storage_service as _storage

                    image_bytes = await _storage.get_file_content(doc.file_path)
                    img = Image.open(BytesIO(image_bytes))
                    media_info["width"] = img.width
                    media_info["height"] = img.height
                    media_info["image_format"] = img.format
                    media_info["color_mode"] = img.mode
                    media_info["media_category"] = "image"
                except ImportError:
                    media_info["probe_error"] = "Pillow not installed"
                    media_info["media_category"] = "image"
                except Exception as img_err:
                    media_info["probe_error"] = str(img_err)
                    media_info["media_category"] = "image"
            else:
                media_info["media_category"] = "other"
            meta = doc.extra_metadata or {}
            media_info["is_transcribed"] = bool(meta.get("is_transcribed"))
            media_info["is_transcribing"] = bool(meta.get("is_transcribing"))
            media_info["transcript_document_id"] = meta.get("transcript_document_id")
            return {"success": True, "data": media_info}
        except Exception as exc:
            return {"error": f"Failed to get media info: {exc}"}

    return FunctionToolProvider(
        name="autonomous_media_tools",
        modes={"autonomous"},
        handlers={
            "transcribe_document": _transcribe_document,
            "analyze_image": _analyze_image,
            "get_media_info": _get_media_info,
        },
    )
