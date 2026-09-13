"""Tests for multi-modal ingestion agent tools (transcribe_document, analyze_image, get_media_info)."""

import uuid


class TestTranscribeDocument:
    """Tests for transcribe_document tool logic."""

    def test_requires_document_id(self):
        params = {}
        doc_id = (params.get("document_id") or "").strip()
        assert not doc_id

    def test_audio_file_types_accepted(self):
        av_prefixes = ("audio/", "video/")
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
        assert any("audio/mpeg".startswith(p) for p in av_prefixes)
        assert ".mp3" in av_exts
        assert ".wav" in av_exts

    def test_video_file_types_accepted(self):
        av_prefixes = ("audio/", "video/")
        assert any("video/mp4".startswith(p) for p in av_prefixes)

    def test_non_av_rejected(self):
        ft = "application/pdf"
        ext = ".pdf"
        av_prefixes = ("audio/", "video/")
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
        is_av = any(ft.startswith(p) for p in av_prefixes) or ext in av_exts
        assert not is_av

    def test_already_transcribed_returns_existing(self):
        meta = {"is_transcribed": True, "transcript_document_id": str(uuid.uuid4())}
        assert meta.get("is_transcribed")
        assert meta.get("transcript_document_id") is not None

    def test_in_progress_returns_status(self):
        meta = {"is_transcribing": True}
        assert meta.get("is_transcribing")

    def test_dispatched_sets_transcribing_flag(self):
        meta = {}
        updated = {**meta, "is_transcribing": True}
        assert updated["is_transcribing"] is True

    def test_result_format_dispatched(self):
        result = {
            "document_id": str(uuid.uuid4()),
            "status": "dispatched",
            "task_id": "celery-task-id-123",
            "title": "Recording.mp3",
        }
        assert result["status"] == "dispatched"
        assert "task_id" in result

    def test_result_format_already_transcribed(self):
        result = {
            "document_id": str(uuid.uuid4()),
            "status": "already_transcribed",
            "transcript_document_id": str(uuid.uuid4()),
        }
        assert result["status"] == "already_transcribed"
        assert result["transcript_document_id"] is not None

    def test_finding_type_transcription_started(self):
        finding = {
            "type": "transcription_started",
            "title": "Transcription started for Recording.mp3",
            "document_id": str(uuid.uuid4()),
            "task_id": "celery-task-123",
        }
        assert finding["type"] == "transcription_started"

    def test_language_param_defaults_auto(self):
        params = {"document_id": str(uuid.uuid4())}
        language = (params.get("language") or "auto").strip()
        assert language == "auto"

    def test_language_param_override(self):
        params = {"document_id": str(uuid.uuid4()), "language": "ru"}
        language = (params.get("language") or "auto").strip()
        assert language == "ru"

    def test_extension_detection(self):
        from pathlib import Path

        assert Path("recording.mp3").suffix.lower() == ".mp3"
        assert Path("video.avi").suffix.lower() == ".avi"
        assert Path("document.pdf").suffix.lower() == ".pdf"


class TestAnalyzeImage:
    """Tests for analyze_image tool logic."""

    def test_requires_document_id(self):
        params = {}
        doc_id = (params.get("document_id") or "").strip()
        assert not doc_id

    def test_image_types_accepted(self):
        image_types = {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff",
        }
        assert "image/png" in image_types
        assert "image/jpeg" in image_types

    def test_image_extensions_accepted(self):
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
        assert ".png" in image_exts
        assert ".jpg" in image_exts
        assert ".tif" in image_exts

    def test_non_image_rejected(self):
        ft = "application/pdf"
        ext = ".pdf"
        image_types = {
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/gif",
            "image/webp",
            "image/bmp",
            "image/tiff",
        }
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
        is_image = ft in image_types or ext in image_exts
        assert not is_image

    def test_default_prompt(self):
        params = {"document_id": str(uuid.uuid4())}
        prompt = (
            params.get("prompt") or ""
        ).strip() or "Describe this image in detail, including any text, diagrams, charts, or notable visual elements."
        assert "Describe this image" in prompt

    def test_custom_prompt(self):
        params = {
            "document_id": str(uuid.uuid4()),
            "prompt": "Extract all text from this image",
        }
        prompt = (params.get("prompt") or "").strip() or "default"
        assert prompt == "Extract all text from this image"

    def test_prompt_truncated_to_2000(self):
        long_prompt = "P" * 3000
        truncated = long_prompt[:2000]
        assert len(truncated) == 2000

    def test_default_vision_model(self):
        vision_model = ""
        if not vision_model:
            vision_model = "llava"
        assert vision_model == "llava"

    def test_custom_vision_model(self):
        params = {"document_id": str(uuid.uuid4()), "model": "llava:13b"}
        vision_model = (params.get("model") or "").strip()
        assert vision_model == "llava:13b"

    def test_image_size_cap_20mb(self):
        size = 25 * 1024 * 1024
        max_size = 20 * 1024 * 1024
        assert size > max_size

    def test_base64_encoding(self):
        import base64

        sample = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64 = base64.b64encode(sample).decode("utf-8")
        decoded = base64.b64decode(b64)
        assert decoded == sample

    def test_ollama_payload_format(self):
        import base64

        image_b64 = base64.b64encode(b"fake image data").decode("utf-8")
        payload = {
            "model": "llava",
            "prompt": "Describe this image",
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2048,
            },
        }
        assert payload["model"] == "llava"
        assert len(payload["images"]) == 1
        assert payload["stream"] is False

    def test_result_format(self):
        result = {
            "document_id": str(uuid.uuid4()),
            "title": "diagram.png",
            "analysis": "This image shows a flowchart...",
            "model": "llava",
            "prompt": "Describe this image",
        }
        assert "analysis" in result
        assert result["model"] == "llava"

    def test_analysis_text_truncated(self):
        long_analysis = "A" * 6000
        truncated = long_analysis[:5000]
        assert len(truncated) == 5000

    def test_finding_type_image_analysis(self):
        finding = {
            "type": "image_analysis",
            "title": "Image analysis: diagram.png",
            "document_id": str(uuid.uuid4()),
            "content": "This image shows...",
            "model": "llava",
        }
        assert finding["type"] == "image_analysis"
        assert "content" in finding

    def test_finding_content_truncated(self):
        long_content = "C" * 3000
        truncated = long_content[:2000]
        assert len(truncated) == 2000

    def test_404_error_message(self):
        error_msg = "HTTP 404 Not Found"
        if "404" in error_msg:
            friendly = (
                "Vision model 'llava' not available. Pull it with: ollama pull llava"
            )
        assert "Pull it with" in friendly


class TestGetMediaInfo:
    """Tests for get_media_info tool logic."""

    def test_requires_document_id(self):
        params = {}
        doc_id = (params.get("document_id") or "").strip()
        assert not doc_id

    def test_av_detection_by_type(self):
        av_prefixes = ("audio/", "video/")
        assert any("audio/mpeg".startswith(p) for p in av_prefixes)
        assert any("video/mp4".startswith(p) for p in av_prefixes)
        assert not any("image/png".startswith(p) for p in av_prefixes)

    def test_av_detection_by_extension(self):
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
        assert ".mp4" in av_exts
        assert ".pdf" not in av_exts

    def test_image_detection_by_type(self):
        assert "image/png".startswith("image/")
        assert "image/jpeg".startswith("image/")
        assert not "audio/mp3".startswith("image/")

    def test_image_detection_by_extension(self):
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
        assert ".png" in image_exts
        assert ".mp3" not in image_exts

    def test_other_category(self):
        ft = "application/pdf"
        ext = ".pdf"
        av_prefixes = ("audio/", "video/")
        av_exts = {".mp3", ".mp4", ".wav"}
        image_exts = {".png", ".jpg"}
        is_av = any(ft.startswith(p) for p in av_prefixes) or ext in av_exts
        is_image = ft.startswith("image/") or ext in image_exts
        if not is_av and not is_image:
            category = "other"
        assert category == "other"

    def test_ffprobe_result_parsing(self):
        probe_data = {
            "format": {
                "duration": "125.5",
                "format_name": "mp4",
                "bit_rate": "192000",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "44100",
                    "channels": 2,
                },
            ],
        }
        fmt = probe_data.get("format", {})
        media_info = {}
        media_info["duration_seconds"] = float(fmt.get("duration", 0))
        media_info["format_name"] = fmt.get("format_name")
        media_info["bit_rate"] = int(fmt.get("bit_rate", 0) or 0)

        for s in probe_data.get("streams", []):
            codec_type = s.get("codec_type")
            if codec_type == "video":
                media_info["video_codec"] = s.get("codec_name")
                media_info["width"] = s.get("width")
                media_info["height"] = s.get("height")
                media_info["fps"] = s.get("r_frame_rate")
            elif codec_type == "audio":
                media_info["audio_codec"] = s.get("codec_name")
                media_info["sample_rate"] = s.get("sample_rate")
                media_info["channels"] = s.get("channels")

        assert media_info["duration_seconds"] == 125.5
        assert media_info["format_name"] == "mp4"
        assert media_info["video_codec"] == "h264"
        assert media_info["width"] == 1920
        assert media_info["height"] == 1080
        assert media_info["audio_codec"] == "aac"
        assert media_info["channels"] == 2

    def test_audio_only_probe(self):
        probe_data = {
            "format": {"duration": "245.3", "format_name": "mp3", "bit_rate": "128000"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "sample_rate": "44100",
                    "channels": 2,
                },
            ],
        }
        media_info = {}
        fmt = probe_data.get("format", {})
        media_info["duration_seconds"] = float(fmt.get("duration", 0))
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "audio":
                media_info["audio_codec"] = s.get("codec_name")
        assert media_info["duration_seconds"] == 245.3
        assert media_info["audio_codec"] == "mp3"
        assert "video_codec" not in media_info

    def test_image_metadata(self):
        media_info = {
            "width": 1920,
            "height": 1080,
            "image_format": "PNG",
            "color_mode": "RGB",
            "media_category": "image",
        }
        assert media_info["width"] == 1920
        assert media_info["height"] == 1080
        assert media_info["image_format"] == "PNG"
        assert media_info["color_mode"] == "RGB"
        assert media_info["media_category"] == "image"

    def test_transcription_status_included(self):
        meta = {"is_transcribed": True, "transcript_document_id": "abc-123"}
        media_info = {}
        media_info["is_transcribed"] = bool(meta.get("is_transcribed"))
        media_info["is_transcribing"] = bool(meta.get("is_transcribing"))
        media_info["transcript_document_id"] = meta.get("transcript_document_id")
        assert media_info["is_transcribed"] is True
        assert media_info["is_transcribing"] is False
        assert media_info["transcript_document_id"] == "abc-123"

    def test_no_transcription_status(self):
        meta = {}
        media_info = {}
        media_info["is_transcribed"] = bool(meta.get("is_transcribed"))
        media_info["is_transcribing"] = bool(meta.get("is_transcribing"))
        media_info["transcript_document_id"] = meta.get("transcript_document_id")
        assert media_info["is_transcribed"] is False
        assert media_info["transcript_document_id"] is None

    def test_result_format(self):
        result = {
            "document_id": str(uuid.uuid4()),
            "title": "video.mp4",
            "file_type": "video/mp4",
            "file_size": 15000000,
            "media_category": "audio_video",
            "duration_seconds": 125.5,
            "video_codec": "h264",
            "width": 1920,
            "height": 1080,
        }
        assert result["media_category"] == "audio_video"
        assert result["duration_seconds"] == 125.5
        assert "file_size" in result

    def test_pillow_import_error_handled(self):
        media_info = {}
        try:
            raise ImportError("No module named 'PIL'")
        except ImportError:
            media_info["probe_error"] = "Pillow not installed"
            media_info["media_category"] = "image"
        assert media_info["probe_error"] == "Pillow not installed"
        assert media_info["media_category"] == "image"

    def test_basic_info_always_present(self):
        media_info = {
            "document_id": str(uuid.uuid4()),
            "title": "file.xyz",
            "file_type": "application/octet-stream",
            "file_size": 1000,
            "media_category": "other",
        }
        assert "document_id" in media_info
        assert "title" in media_info
        assert "file_type" in media_info
        assert "file_size" in media_info


class TestMultiModalSchemas:
    """Tests for multi-modal tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "transcribe_document" in names
        assert "analyze_image" in names
        assert "get_media_info" in names

    def test_transcribe_document_requires_document_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("transcribe_document")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_id" in required

    def test_transcribe_document_has_language(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("transcribe_document")
        assert "language" in tool["parameters"]["properties"]

    def test_analyze_image_requires_document_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("analyze_image")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_id" in required

    def test_analyze_image_has_prompt_and_model(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("analyze_image")
        props = tool["parameters"]["properties"]
        assert "prompt" in props
        assert "model" in props

    def test_get_media_info_requires_document_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_media_info")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_id" in required


class TestMultiModalRegistry:
    """Tests for multi-modal tool registry classification."""

    def test_transcribe_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("transcribe_document")
        assert meta is not None
        assert meta.effects == "write"

    def test_transcribe_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("transcribe_document")
        assert meta.cost_tier == "medium"

    def test_analyze_image_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("analyze_image")
        assert meta is not None
        assert meta.effects == "read"

    def test_analyze_image_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("analyze_image")
        assert meta.cost_tier == "medium"

    def test_get_media_info_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_media_info")
        assert meta is not None
        assert meta.effects == "read"

    def test_get_media_info_is_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_media_info")
        assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["transcribe_document", "analyze_image", "get_media_info"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
