"""The media tools: transcribe_document, analyze_image, get_media_info.

Every test here calls the real handler from `build_autonomous_media_provider`
against the real in-memory database. Only the edges that leave the process are
replaced: Celery's `.delay`, object storage, the vision model's HTTP client,
`ffprobe` and Pillow.
"""

import base64
import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.core.config import settings
from app.models.document import Document, DocumentSource
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_media_provider,
)
from app.services.storage_service import storage_service
from app.tasks.transcription_tasks import transcribe_document as transcribe_task

pytestmark = pytest.mark.unit

TOOLS = ("transcribe_document", "analyze_image", "get_media_info")

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


# ---------------------------------------------------------------- fakes


class FakeResponse:
    def __init__(self, body=None, error=None):
        self._body = body if body is not None else {"response": "A flowchart."}
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._body


class FakeVisionClient:
    """Stands in for `LLMService.client` (an httpx.AsyncClient)."""

    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    async def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self.response


class TaskRecorder:
    """Stands in for the Celery task's `.delay`."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(id="celery-task-1")


# ------------------------------------------------------------- fixtures


@pytest.fixture
def queued(monkeypatch):
    recorder = TaskRecorder()
    monkeypatch.setattr(transcribe_task, "delay", recorder)
    return recorder


@pytest.fixture
def vision():
    return FakeVisionClient()


@pytest.fixture
def storage(monkeypatch):
    """Object storage holding whatever the test puts in `files`."""
    state = SimpleNamespace(files={}, reads=[], downloads=[])

    async def get_file_content(path):
        state.reads.append(path)
        if path not in state.files:
            raise FileNotFoundError(f"File not found: {path}")
        return state.files[path]

    async def download_file(path, local_path):
        state.downloads.append((path, local_path))
        if path not in state.files:
            return False
        with open(local_path, "wb") as fh:
            fh.write(state.files[path])
        return True

    monkeypatch.setattr(storage_service, "get_file_content", get_file_content)
    monkeypatch.setattr(storage_service, "download_file", download_file)
    return state


@pytest.fixture
def call(db_session, test_user, vision):
    executor = SimpleNamespace(
        llm_service=SimpleNamespace(client=vision, base_url="http://ollama.test")
    )
    provider = build_autonomous_media_provider(executor)
    job = SimpleNamespace(id=uuid4(), user_id=test_user.id, config={})

    async def _call(tool, params):
        return await provider._handlers[tool](
            params,
            AgentToolExecutionContext(
                mode="autonomous",
                db=db_session,
                service=None,
                user_id=str(test_user.id),
                job=job,
                state={},
            ),
        )

    return _call


@pytest.fixture
def make_document(db_session):
    async def _make(
        *,
        title="Recording",
        file_path="uploads/recording.mp3",
        file_type="audio/mpeg",
        file_size=1234,
        extra_metadata=None,
    ):
        source = DocumentSource(
            name=f"uploads-{uuid4().hex[:8]}", source_type="file", config={}
        )
        db_session.add(source)
        await db_session.flush()
        doc = Document(
            title=title,
            content="",
            content_hash=uuid4().hex,
            source_id=source.id,
            source_identifier=uuid4().hex,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            extra_metadata=extra_metadata,
        )
        db_session.add(doc)
        await db_session.commit()
        return doc

    return _make


async def _stored_metadata(db_session, doc):
    await db_session.refresh(doc)
    return doc.extra_metadata or {}


# ======================================================================
# The handlers as shipped
# ======================================================================


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("params", [{}, {"document_id": ""}, {"document_id": "  "}])
async def test_a_call_without_a_document_id_is_refused(tool, params, call, queued):
    result = await call(tool, params)

    assert "document_id" in result["error"]
    assert "success" not in result
    assert queued.calls == []


@pytest.mark.parametrize("tool", TOOLS)
async def test_a_malformed_document_id_is_refused(tool, call, queued, vision):
    result = await call(tool, {"document_id": "not-a-uuid"})

    assert result.get("error")
    assert "success" not in result
    assert queued.calls == []
    assert vision.calls == []


@pytest.mark.parametrize("tool", TOOLS)
async def test_an_unknown_document_is_reported_as_not_found(tool, call):
    missing = str(uuid4())

    result = await call(tool, {"document_id": missing})

    assert result["error"] == f"Document {missing} not found"


async def test_transcribing_an_audio_document_queues_the_task(
    call, make_document, queued, db_session
):
    doc = await make_document()

    result = await call("transcribe_document", {"document_id": str(doc.id)})

    assert result.get("success") is True, result
    assert queued.calls == [((str(doc.id),), {})]
    assert (await _stored_metadata(db_session, doc))["is_transcribing"] is True


async def test_analyzing_an_image_asks_the_vision_model(
    call, make_document, storage, vision
):
    doc = await make_document(
        title="diagram.png", file_path="uploads/diagram.png", file_type="image/png"
    )
    storage.files["uploads/diagram.png"] = PNG

    result = await call("analyze_image", {"document_id": str(doc.id)})

    assert result.get("success") is True, result
    assert result["data"]["analysis"] == "A flowchart."
    assert len(vision.calls) == 1


async def test_media_info_describes_a_stored_document(call, make_document):
    doc = await make_document(
        title="paper.pdf", file_path="uploads/paper.pdf", file_type="application/pdf"
    )

    result = await call("get_media_info", {"document_id": str(doc.id)})

    assert result.get("success") is True, result
    assert result["data"]["title"] == "paper.pdf"


# ======================================================================
# transcribe_document, once the lookup works
# ======================================================================


class TestTranscribeDocument:
    async def test_an_unknown_document_is_not_found(self, call, queued):
        missing = str(uuid4())

        result = await call("transcribe_document", {"document_id": missing})

        assert result["error"] == f"Document {missing} not found"
        assert queued.calls == []

    async def test_dispatch_queues_the_task_and_marks_the_document(
        self, call, make_document, queued, db_session
    ):
        doc = await make_document(
            title="Standup", extra_metadata={"uploaded_by": "someone"}
        )

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert result.get("success") is True, result
        # The task takes exactly one argument: the document id, as a string.
        assert queued.calls == [((str(doc.id),), {})]
        assert result["data"] == {
            "document_id": str(doc.id),
            "status": "dispatched",
            "task_id": "celery-task-1",
            "title": "Standup",
        }
        finding = result["findings"][0]
        assert finding["type"] == "transcription_started"
        assert finding["document_id"] == str(doc.id)
        assert finding["task_id"] == "celery-task-1"
        # Persisted, and without discarding what was already there.
        assert await _stored_metadata(db_session, doc) == {
            "uploaded_by": "someone",
            "is_transcribing": True,
        }

    @pytest.mark.parametrize(
        "file_path,file_type",
        [
            ("uploads/blob", "audio/mpeg"),
            ("uploads/blob", "video/mp4"),
            ("uploads/talk.WAV", "application/octet-stream"),
            ("uploads/talk.mkv", None),
        ],
    )
    async def test_audio_and_video_are_accepted_by_type_or_extension(
        self, file_path, file_type, call, make_document, queued
    ):
        doc = await make_document(file_path=file_path, file_type=file_type)

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert result.get("success") is True, result
        assert len(queued.calls) == 1

    @pytest.mark.parametrize(
        "file_path,file_type",
        [
            ("uploads/paper.pdf", "application/pdf"),
            ("uploads/diagram.png", "image/png"),
            ("uploads/notes", None),
        ],
    )
    async def test_anything_else_is_refused_and_left_untouched(
        self, file_path, file_type, call, make_document, queued, db_session
    ):
        doc = await make_document(file_path=file_path, file_type=file_type)

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert "not audio/video" in result["error"]
        assert queued.calls == []
        assert "is_transcribing" not in await _stored_metadata(db_session, doc)

    async def test_a_document_without_a_file_is_refused(
        self, call, make_document, queued
    ):
        doc = await make_document(file_path=None)

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert "no associated file" in result["error"]
        assert queued.calls == []

    async def test_an_already_transcribed_document_is_not_queued_again(
        self, call, make_document, queued
    ):
        transcript_id = str(uuid4())
        doc = await make_document(
            extra_metadata={
                "is_transcribed": True,
                "transcript_document_id": transcript_id,
            }
        )

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert result["data"] == {
            "document_id": str(doc.id),
            "status": "already_transcribed",
            "transcript_document_id": transcript_id,
        }
        assert queued.calls == []

    async def test_a_transcription_in_progress_is_not_queued_again(
        self, call, make_document, queued
    ):
        doc = await make_document(extra_metadata={"is_transcribing": True})

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert result["data"]["status"] == "in_progress"
        assert queued.calls == []

    async def test_a_failed_enqueue_does_not_leave_the_document_in_progress(
        self, call, make_document, monkeypatch, db_session
    ):
        doc = await make_document()
        monkeypatch.setattr(
            transcribe_task, "delay", TaskRecorder(error=RuntimeError("broker down"))
        )

        result = await call("transcribe_document", {"document_id": str(doc.id)})

        assert "broker down" in result["error"]
        assert not (await _stored_metadata(db_session, doc)).get("is_transcribing")


# ======================================================================
# analyze_image, once the lookup works
# ======================================================================


class TestAnalyzeImage:
    @pytest.fixture
    async def image(self, make_document, storage):
        doc = await make_document(
            title="diagram.png",
            file_path="uploads/diagram.png",
            file_type="image/png",
        )
        storage.files["uploads/diagram.png"] = PNG
        return doc

    async def test_the_image_is_sent_to_the_configured_vision_model(
        self, call, image, vision, monkeypatch
    ):
        monkeypatch.setattr(settings, "VISION_MODEL", "llava:13b", raising=False)

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert result.get("success") is True, result
        (sent,) = vision.calls
        assert sent["url"] == "http://ollama.test/api/generate"
        assert sent["json"]["model"] == "llava:13b"
        assert sent["json"]["stream"] is False
        assert [base64.b64decode(i) for i in sent["json"]["images"]] == [PNG]
        assert "Describe this image" in sent["json"]["prompt"]
        assert result["data"]["document_id"] == str(image.id)
        assert result["data"]["title"] == "diagram.png"
        assert result["data"]["analysis"] == "A flowchart."
        assert result["data"]["model"] == "llava:13b"
        finding = result["findings"][0]
        assert finding["type"] == "image_analysis"
        assert finding["document_id"] == str(image.id)
        assert finding["content"] == "A flowchart."
        assert finding["model"] == "llava:13b"

    async def test_prompt_and_model_can_be_overridden(self, call, image, vision):
        result = await call(
            "analyze_image",
            {
                "document_id": str(image.id),
                "prompt": "Extract all text",
                "model": "bakllava",
            },
        )

        assert vision.calls[0]["json"]["prompt"] == "Extract all text"
        assert vision.calls[0]["json"]["model"] == "bakllava"
        assert result["data"]["model"] == "bakllava"

    async def test_long_prompts_and_long_answers_are_capped(self, call, image, vision):
        vision.response = FakeResponse({"response": "A" * 6000})

        result = await call(
            "analyze_image", {"document_id": str(image.id), "prompt": "P" * 3000}
        )

        assert vision.calls[0]["json"]["prompt"] == "P" * 2000
        assert result["data"]["analysis"] == "A" * 5000
        assert result["findings"][0]["content"] == "A" * 2000

    @pytest.mark.parametrize(
        "file_path,file_type",
        [("uploads/scan.TIF", "application/octet-stream"), ("uploads/b", "image/webp")],
    )
    async def test_images_are_accepted_by_type_or_extension(
        self, file_path, file_type, call, make_document, storage, vision
    ):
        doc = await make_document(file_path=file_path, file_type=file_type)
        storage.files[file_path] = PNG

        result = await call("analyze_image", {"document_id": str(doc.id)})

        assert result.get("success") is True, result

    async def test_a_non_image_is_refused_before_anything_is_fetched(
        self, call, make_document, storage, vision
    ):
        doc = await make_document()  # audio

        result = await call("analyze_image", {"document_id": str(doc.id)})

        assert "not an image" in result["error"]
        assert storage.reads == []
        assert vision.calls == []

    async def test_a_document_without_a_file_is_refused(
        self, call, make_document, vision
    ):
        doc = await make_document(file_path=None, file_type="image/png")

        result = await call("analyze_image", {"document_id": str(doc.id)})

        assert "no associated file" in result["error"]
        assert vision.calls == []

    async def test_an_image_over_20mb_is_never_sent(self, call, image, storage, vision):
        storage.files["uploads/diagram.png"] = b"\x00" * (20 * 1024 * 1024 + 1)

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert "too large" in result["error"].lower()
        assert vision.calls == []

    async def test_an_image_of_exactly_20mb_is_sent(self, call, image, storage, vision):
        storage.files["uploads/diagram.png"] = b"\x00" * (20 * 1024 * 1024)

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert result.get("success") is True, result

    async def test_an_empty_file_is_refused(self, call, image, storage, vision):
        storage.files["uploads/diagram.png"] = b""

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert "empty" in result["error"].lower()
        assert vision.calls == []

    async def test_an_empty_answer_is_an_error_not_a_finding(self, call, image, vision):
        vision.response = FakeResponse({"response": "   "})

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert "empty response" in result["error"]
        assert "findings" not in result

    async def test_a_model_the_server_does_not_have_says_how_to_get_it(
        self, call, image, vision
    ):
        request = httpx.Request("POST", "http://ollama.test/api/generate")
        vision.response = FakeResponse(
            error=httpx.HTTPStatusError(
                "Client error '404 Not Found' for url '/api/generate'",
                request=request,
                response=httpx.Response(404, request=request),
            )
        )

        result = await call(
            "analyze_image", {"document_id": str(image.id), "model": "bakllava"}
        )

        assert "bakllava" in result["error"]
        assert "ollama pull bakllava" in result["error"]

    async def test_no_ollama_to_reach_is_named_as_such(self, call, image, vision):
        # Image analysis is Ollama-only and the stack does not bundle Ollama;
        # a bare "connection refused" read as a fault worth retrying.
        async def refused(url, json=None, timeout=None):
            raise httpx.ConnectError("[Errno 61] Connection refused")

        vision.post = refused

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert "needs an Ollama instance" in result["error"]
        assert "OLLAMA_BASE_URL" in result["error"]
        assert "Connection refused" in result["error"]
        assert "findings" not in result

    async def test_any_other_upstream_failure_is_reported_as_itself(
        self, call, image, vision
    ):
        vision.response = FakeResponse(error=RuntimeError("503 Service Unavailable"))

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert "503 Service Unavailable" in result["error"]
        assert "ollama pull" not in result["error"]

    async def test_a_file_missing_from_storage_is_not_blamed_on_the_model(
        self, call, image, storage, vision
    ):
        storage.files.clear()

        result = await call("analyze_image", {"document_id": str(image.id)})

        assert vision.calls == []
        assert "ollama pull" not in result["error"]
        assert "uploads/diagram.png" in result["error"]


# ======================================================================
# get_media_info, once the lookup works
# ======================================================================


PROBE = {
    "format": {"duration": "125.5", "format_name": "mp4", "bit_rate": "192000"},
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


@pytest.fixture
def ffprobe(monkeypatch):
    """Replaces the `ffprobe` subprocess; records what it was asked to probe."""
    state = SimpleNamespace(
        calls=[], returncode=0, stdout=json.dumps(PROBE), existed=[], error=None
    )

    def run(cmd, **kwargs):
        state.calls.append(cmd)
        state.existed.append(os.path.exists(cmd[-1]))
        if state.error:
            raise state.error
        return SimpleNamespace(returncode=state.returncode, stdout=state.stdout)

    monkeypatch.setattr(subprocess, "run", run)
    return state


@pytest.fixture
def pillow(monkeypatch):
    """A Pillow that reports fixed dimensions for whatever it is handed."""
    opened = []

    def _open(fp):
        opened.append(fp.read())
        return SimpleNamespace(width=640, height=480, format="PNG", mode="RGBA")

    image_module = ModuleType("PIL.Image")
    image_module.open = _open
    package = ModuleType("PIL")
    package.Image = image_module
    monkeypatch.setitem(sys.modules, "PIL", package)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)
    return opened


class TestGetMediaInfo:
    async def test_an_unknown_document_is_not_found(self, call):
        missing = str(uuid4())

        result = await call("get_media_info", {"document_id": missing})

        assert result["error"] == f"Document {missing} not found"

    async def test_a_document_without_a_file_is_refused(self, call, make_document):
        doc = await make_document(file_path=None)

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert "no associated file" in result["error"]

    async def test_a_video_reports_what_ffprobe_measured(
        self, call, make_document, storage, ffprobe
    ):
        doc = await make_document(
            title="talk.mp4",
            file_path="uploads/talk.mp4",
            file_type="video/mp4",
            file_size=15_000_000,
        )
        storage.files["uploads/talk.mp4"] = b"mp4-bytes"

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert result.get("success") is True, result
        assert result["data"] == {
            "document_id": str(doc.id),
            "title": "talk.mp4",
            "file_type": "video/mp4",
            "file_size": 15_000_000,
            "media_category": "audio_video",
            "duration_seconds": 125.5,
            "format_name": "mp4",
            "bit_rate": 192000,
            "video_codec": "h264",
            "width": 1920,
            "height": 1080,
            "fps": "30/1",
            "audio_codec": "aac",
            "sample_rate": "44100",
            "channels": 2,
            "is_transcribed": False,
            "is_transcribing": False,
            "transcript_document_id": None,
        }
        # The stored object was downloaded to the file ffprobe was pointed at,
        # which existed while it ran and is gone afterwards.
        ((source, local_path),) = storage.downloads
        assert source == "uploads/talk.mp4"
        assert ffprobe.calls[0][0] == "ffprobe"
        assert ffprobe.calls[0][-1] == local_path
        assert local_path.endswith(".mp4")
        assert ffprobe.existed == [True]
        assert not os.path.exists(local_path)

    async def test_an_audio_file_reports_no_video_fields(
        self, call, make_document, storage, ffprobe
    ):
        doc = await make_document()
        storage.files["uploads/recording.mp3"] = b"mp3-bytes"
        ffprobe.stdout = json.dumps(
            {
                "format": {"duration": "245.3", "format_name": "mp3"},
                "streams": [{"codec_type": "audio", "codec_name": "mp3"}],
            }
        )

        data = (await call("get_media_info", {"document_id": str(doc.id)}))["data"]

        assert data["media_category"] == "audio_video"
        assert data["duration_seconds"] == 245.3
        assert data["audio_codec"] == "mp3"
        assert data["bit_rate"] == 0
        assert "video_codec" not in data
        assert "width" not in data

    async def test_a_failed_probe_is_reported_and_cleans_up(
        self, call, make_document, storage, ffprobe
    ):
        doc = await make_document()
        storage.files["uploads/recording.mp3"] = b"not really audio"
        ffprobe.returncode = 1

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert result.get("success") is True, result
        assert result["data"]["probe_error"]
        assert result["data"]["media_category"] == "audio_video"
        assert "duration_seconds" not in result["data"]
        assert not os.path.exists(storage.downloads[0][1])

    async def test_a_missing_ffprobe_binary_is_reported_and_cleans_up(
        self, call, make_document, storage, ffprobe
    ):
        doc = await make_document()
        storage.files["uploads/recording.mp3"] = b"mp3-bytes"
        ffprobe.error = FileNotFoundError("No such file or directory: 'ffprobe'")

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert "ffprobe" in result["data"]["probe_error"]
        assert result["data"]["media_category"] == "audio_video"
        assert not os.path.exists(storage.downloads[0][1])

    async def test_a_file_missing_from_storage_is_not_probed(
        self, call, make_document, storage, ffprobe
    ):
        doc = await make_document()  # nothing put in storage

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert ffprobe.calls == []
        assert result.get("error") or result["data"].get("probe_error")

    async def test_an_image_reports_its_dimensions(
        self, call, make_document, storage, pillow, ffprobe
    ):
        doc = await make_document(
            title="diagram.png", file_path="uploads/diagram.png", file_type="image/png"
        )
        storage.files["uploads/diagram.png"] = PNG

        data = (await call("get_media_info", {"document_id": str(doc.id)}))["data"]

        assert pillow == [PNG]
        assert ffprobe.calls == []
        assert data["media_category"] == "image"
        assert (data["width"], data["height"]) == (640, 480)
        assert data["image_format"] == "PNG"
        assert data["color_mode"] == "RGBA"

    async def test_an_image_without_pillow_says_so(
        self, call, make_document, storage, monkeypatch
    ):
        doc = await make_document(
            file_path="uploads/diagram.png", file_type="image/png"
        )
        storage.files["uploads/diagram.png"] = PNG
        monkeypatch.setitem(sys.modules, "PIL", None)

        data = (await call("get_media_info", {"document_id": str(doc.id)}))["data"]

        assert data["probe_error"] == "Pillow not installed"
        assert data["media_category"] == "image"
        assert "width" not in data

    async def test_an_unreadable_image_is_reported_not_raised(
        self, call, make_document, storage, pillow
    ):
        doc = await make_document(
            file_path="uploads/diagram.png", file_type="image/png"
        )  # nothing put in storage

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert result.get("success") is True, result
        assert "uploads/diagram.png" in result["data"]["probe_error"]
        assert result["data"]["media_category"] == "image"

    async def test_anything_else_is_other_and_touches_no_storage(
        self, call, make_document, storage, ffprobe
    ):
        doc = await make_document(
            title="paper.pdf",
            file_path="uploads/paper.pdf",
            file_type="application/pdf",
            file_size=1000,
        )

        result = await call("get_media_info", {"document_id": str(doc.id)})

        assert result["data"] == {
            "document_id": str(doc.id),
            "title": "paper.pdf",
            "file_type": "application/pdf",
            "file_size": 1000,
            "media_category": "other",
            "is_transcribed": False,
            "is_transcribing": False,
            "transcript_document_id": None,
        }
        assert storage.reads == [] and storage.downloads == []
        assert ffprobe.calls == []

    async def test_transcription_status_is_included(
        self, call, make_document, storage, ffprobe
    ):
        transcript_id = str(uuid4())
        doc = await make_document(
            extra_metadata={
                "is_transcribed": True,
                "transcript_document_id": transcript_id,
            }
        )
        storage.files["uploads/recording.mp3"] = b"mp3-bytes"

        data = (await call("get_media_info", {"document_id": str(doc.id)}))["data"]

        assert data["is_transcribed"] is True
        assert data["is_transcribing"] is False
        assert data["transcript_document_id"] == transcript_id


# ======================================================================
# Schemas and registry
# ======================================================================


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

    def test_transcribe_document_offers_no_language(self):
        """The task transcribes in the configured language and takes no
        other; the tool used to advertise a parameter nothing read."""
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("transcribe_document")
        assert "language" not in tool["parameters"]["properties"]

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

    def test_the_provider_answers_exactly_these_tools(self):
        provider = build_autonomous_media_provider(SimpleNamespace())
        assert provider.supported_tools == set(TOOLS)


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

    def test_only_image_analysis_goes_out(self):
        # analyze_image posts the image to an Ollama server; the other two
        # stay on this side.
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["transcribe_document", "get_media_info"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
        assert get_tool_metadata("analyze_image").network == "egress"
