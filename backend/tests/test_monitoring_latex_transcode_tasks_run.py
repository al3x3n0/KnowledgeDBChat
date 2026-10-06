"""The monitoring, LaTeX and transcode Celery tasks, run for real.

The five monitoring tasks run from the beat schedule, the LaTeX compile job
from the dedicated LaTeX worker and the transcode from upload. None of their
bodies was named in any test. These run them -- the async bodies against the
in-memory database, and each synchronous Celery function once through ``.run``
against a file database -- and judge on the rows they leave behind.

Only edges that leave the process are replaced, and every replacement binds
its arguments against the real callee, so a call the real thing would refuse
fails here too:

* Redis -- ``job_support.publish_sync`` (and its async siblings);
* MinIO -- ``MinIOStorageService`` at class level, with leftover instance
  attributes lifted from the ``storage_service`` singleton;
* Qdrant -- ``VectorStoreService.initialize`` / ``get_collection_stats``;
* the LLM provider -- ``httpx.AsyncClient.get``, the request
  ``LLMService.health_check`` makes;
* subprocesses -- ``subprocess.run`` (pdflatex, ffmpeg, ffprobe), with the
  binaries' lookup (``shutil.which``) answering for an image that has them;
* the broker -- ``transcribe_document.apply_async`` and the bound task's
  ``update_state`` (the result backend).

A test marked xfail(strict) is a defect in the application, not in the test.
"""

import asyncio
import inspect
import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from celery import Task as CeleryTask
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.database import Base
from app.models.agent_job import AgentJob
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, DocumentSource
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.latex_compile_job import LatexCompileJob
from app.models.latex_project import LatexProject
from app.models.latex_project_file import LatexProjectFile
from app.models.notification import (
    Notification,
    NotificationPreferences,
    NotificationType,
)
from app.models.research_note import ResearchNote
from app.services import latex_compiler_service as latex_compiler_module
from app.services.auth_service import AuthService
from app.services.storage_service import MinIOStorageService
from app.services.vector_store import VectorStoreService, vector_store_service
from app.tasks import job_support, latex_tasks, monitoring_tasks, transcode_tasks

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Edges that leave the process
# --------------------------------------------------------------------------


# The real callees, captured once: a fake installed over another fake must
# still bind against the real signature, not the first fake's (*args, **kwargs).
REAL = {
    "publish_progress": job_support.publish_progress,
    "publish_message": job_support.publish_message,
    "publish_sync": job_support.publish_sync,
    "minio_initialize": MinIOStorageService.initialize,
    "upload_file": MinIOStorageService.upload_file,
    "get_file_content": MinIOStorageService.get_file_content,
    "download_file": MinIOStorageService.download_file,
    "vector_initialize": VectorStoreService.initialize,
    "collection_stats": VectorStoreService.get_collection_stats,
    "http_get": httpx.AsyncClient.get,
    "subprocess_run": subprocess.run,
    "disk_usage": shutil.disk_usage,
}


def _binding(real):
    """A checker that raises TypeError exactly when ``real`` would."""
    signature = inspect.signature(real)

    def bind(*args, **kwargs):
        return signature.bind(*args, **kwargs).arguments

    return bind


class FakeRedis:
    """Every message a task publishes, in order."""

    def __init__(self):
        self.messages = []  # (channel, message)

    def install(self, monkeypatch):
        bind_progress = _binding(REAL["publish_progress"])
        bind_message = _binding(REAL["publish_message"])
        bind_sync = _binding(REAL["publish_sync"])

        async def publish_progress(*args, **kwargs):
            a = bind_progress(*args, **kwargs)
            self.messages.append((a["channel"], {"progress": a["progress"]}))

        async def publish_message(*args, **kwargs):
            a = bind_message(*args, **kwargs)
            json.dumps(dict(a["message"]))
            self.messages.append((a["channel"], dict(a["message"])))

        def publish_sync(*args, **kwargs):
            a = bind_sync(*args, **kwargs)
            json.dumps(dict(a["message"]))
            self.messages.append((a["channel"], dict(a["message"])))

        monkeypatch.setattr(job_support, "publish_progress", publish_progress)
        monkeypatch.setattr(job_support, "publish_message", publish_message)
        monkeypatch.setattr(job_support, "publish_sync", publish_sync)
        return self

    def on(self, channel):
        return [m for c, m in self.messages if c == channel]


class FakeMinio:
    """An object store in a dict, behind MinIOStorageService's own signatures.

    ``download_file`` answers False for a missing object, as the real one does
    (it catches NoSuchKey and every other error and returns False).
    """

    def __init__(self, fail_uploads=None):
        self.objects = {}
        self.content_types = {}
        self.downloads = []
        self.fail_uploads = fail_uploads

    def install(self, monkeypatch):
        cls = MinIOStorageService
        bind_init = _binding(REAL["minio_initialize"])
        bind_upload = _binding(REAL["upload_file"])
        bind_get = _binding(REAL["get_file_content"])
        bind_download = _binding(REAL["download_file"])
        store = self

        async def initialize(*args, **kwargs):
            bind_init(*args, **kwargs)

        async def upload_file(*args, **kwargs):
            a = bind_upload(*args, **kwargs)
            assert isinstance(a["content"], (bytes, bytearray))
            if store.fail_uploads is not None:
                raise store.fail_uploads
            path = f"{a['document_id']}/{a['filename']}"
            store.objects[path] = bytes(a["content"])
            store.content_types[path] = a.get("content_type")
            return path

        async def get_file_content(*args, **kwargs):
            a = bind_get(*args, **kwargs)
            return store.objects[a["object_path"]]

        async def download_file(*args, **kwargs):
            a = bind_download(*args, **kwargs)
            store.downloads.append(a["object_path"])
            if a["object_path"] not in store.objects:
                return False
            Path(a["local_path"]).write_bytes(store.objects[a["object_path"]])
            return True

        monkeypatch.setattr(cls, "initialize", initialize)
        monkeypatch.setattr(cls, "upload_file", upload_file)
        monkeypatch.setattr(cls, "get_file_content", get_file_content)
        monkeypatch.setattr(cls, "download_file", download_file)

        # An earlier test's instance-level monkeypatch on the singleton leaves
        # bound methods behind that shadow these class patches.
        from app.services.storage_service import storage_service as _singleton

        for _name in list(vars(_singleton)):
            if callable(getattr(cls, _name, None)) and _name not in ("_get_client",):
                monkeypatch.delattr(_singleton, _name)
        return self


class FakeQdrant:
    def __init__(self, fail_with=None, chunks=42):
        self.fail_with = fail_with
        self.chunks = chunks
        self.initialized = 0

    def install(self, monkeypatch):
        bind_init = _binding(REAL["vector_initialize"])
        bind_stats = _binding(REAL["collection_stats"])
        fake = self

        async def initialize(*args, **kwargs):
            bind_init(*args, **kwargs)
            if fake.fail_with is not None:
                raise fake.fail_with
            fake.initialized += 1

        async def get_collection_stats(*args, **kwargs):
            bind_stats(*args, **kwargs)
            if fake.fail_with is not None:
                raise fake.fail_with
            return {"total_chunks": fake.chunks, "collection_name": "kdbc"}

        monkeypatch.setattr(VectorStoreService, "initialize", initialize)
        monkeypatch.setattr(
            VectorStoreService, "get_collection_stats", get_collection_stats
        )
        # An earlier test's instance-level monkeypatch on the singleton leaves
        # its methods behind as instance attributes, which shadow the class
        # patches above: in the full run the REAL initialize ran (and tried to
        # load the embedding model). Lift them for this test.
        for name in ("initialize", "get_collection_stats"):
            if name in vars(vector_store_service):
                monkeypatch.delattr(vector_store_service, name)
        monkeypatch.setattr(vector_store_service, "_initialized", False)
        return self


class FakeHttp:
    """``httpx.AsyncClient.get``: hosts that answer, and hosts nothing listens on."""

    def __init__(self, answers):
        self.answers = dict(answers)  # host -> status
        self.requested = []

    def install(self, monkeypatch):
        bind = _binding(REAL["http_get"])
        fake = self

        async def get(*args, **kwargs):
            a = bind(*args, **kwargs)
            url = httpx.URL(str(a["url"]))
            fake.requested.append(str(url))
            status = fake.answers.get(url.host)
            if status is None:
                raise httpx.ConnectError(
                    "[Errno 61] Connection refused",
                    request=httpx.Request("GET", url),
                )
            return httpx.Response(status, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", get)
        return self


class FakeDisk:
    def __init__(self, used_fraction=0.5):
        self.used_fraction = used_fraction

    def install(self, monkeypatch):
        real = REAL["disk_usage"]
        bind = _binding(real)
        fake = self

        def disk_usage(*args, **kwargs):
            a = bind(*args, **kwargs)
            if not Path(a["path"]).exists():
                return real(a["path"])  # FileNotFoundError, as the real one
            total = 500 * 1024**3
            used = int(total * fake.used_fraction)
            return shutil._ntuple_diskusage(total, used, total - used)

        monkeypatch.setattr(shutil, "disk_usage", disk_usage)
        return self


class FakeProcesses:
    """``subprocess.run`` for pdflatex, ffmpeg and ffprobe.

    Each behaves as the binary does on the inputs these tests give it: pdflatex
    writes ``main.pdf`` into its output directory unless the source uses an
    undefined control sequence, ffmpeg writes its output unless the input is
    empty or unreadable, ffprobe reports a duration.
    """

    PDF = b"%PDF-1.5\n% fake pdflatex output\n%%EOF\n"
    MP4 = b"\x00\x00\x00\x18ftypmp42 fake h264"

    def __init__(self, ffmpeg_fails=False, duration="12.5\n"):
        self.calls = []
        self.ffmpeg_fails = ffmpeg_fails
        self.duration = duration
        self.seen_files = {}  # name -> bytes, what pdflatex found beside main.tex
        self.temp_paths = []

    def install(self, monkeypatch):
        bind = _binding(REAL["subprocess_run"])
        fake = self

        def run(*args, **kwargs):
            a = bind(*args, **kwargs)
            cmd = list(a["popenargs"][0])
            opts = dict(a.get("kwargs") or {})
            for key in ("capture_output", "timeout", "check", "input"):
                if key in a:
                    opts[key] = a[key]
            fake.calls.append(cmd)
            name = Path(cmd[0]).name
            if name == "pdflatex":
                return fake._pdflatex(cmd, opts)
            if name == "ffmpeg":
                return fake._ffmpeg(cmd, opts)
            if name == "ffprobe":
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=fake.duration.encode(), stderr=b""
                )
            raise FileNotFoundError(2, "No such file or directory", cmd[0])

        def which(name, *args, **kwargs):
            return f"/usr/bin/{name}" if name in ("pdflatex", "ffmpeg") else None

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(latex_compiler_module.shutil, "which", which)
        return self

    def names(self):
        return [Path(c[0]).name for c in self.calls]

    def _pdflatex(self, cmd, opts):
        cwd = Path(opts["cwd"])
        out_dir = Path(
            next(c for c in cmd if c.startswith("-output-directory=")).split("=", 1)[1]
        )
        source = (cwd / cmd[-1]).read_text()
        for p in cwd.iterdir():
            if p.is_file() and p.name != "main.tex":
                self.seen_files[p.name] = p.read_bytes()
        if "\\undefinedmacro" in source:
            output = "./main.tex:3: Undefined control sequence.\nl.3 \\undefinedmacro"
            (out_dir / "main.log").write_text("! Undefined control sequence.\n")
            return subprocess.CompletedProcess(cmd, 1, stdout=output)
        (out_dir / "main.pdf").write_bytes(self.PDF)
        (out_dir / "main.log").write_text("Output written on main.pdf (1 page).\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="This is pdfTeX\n")

    def _ffmpeg(self, cmd, opts):
        src = Path(cmd[cmd.index("-i") + 1])
        dst = Path(cmd[-1])
        self.temp_paths += [src, dst]
        rc = 0
        if self.ffmpeg_fails or not src.exists() or src.stat().st_size == 0:
            rc = 1  # "Invalid data found when processing input"
        else:
            dst.write_bytes(self.MP4)
        if rc and opts.get("check"):
            raise subprocess.CalledProcessError(rc, cmd)
        return subprocess.CompletedProcess(cmd, rc)


class FakeBroker:
    """``transcribe_document.apply_async``, bound to Celery's signature."""

    def __init__(self, fail_with=None):
        self.sent = []
        self.fail_with = fail_with

    def install(self, monkeypatch):
        from app.tasks.transcription_tasks import transcribe_document

        bind = _binding(CeleryTask.apply_async)
        fake = self

        def apply_async(*args, **kwargs):
            a = bind(transcribe_document, *args, **kwargs)
            if fake.fail_with is not None:
                raise fake.fail_with
            fake.sent.append(a)

        monkeypatch.setattr(transcribe_document, "apply_async", apply_async)
        return self


def _fake_update_state(monkeypatch, task):
    """The bound task's ``update_state`` writes to the result backend."""
    bind = _binding(CeleryTask.update_state)
    states = []

    def update_state(*args, **kwargs):
        a = bind(task, *args, **kwargs)
        states.append((a.get("state"), (a.get("meta") or {}).get("status")))

    monkeypatch.setattr(task, "update_state", update_state)
    return states


@pytest.fixture
def redis_fake(monkeypatch):
    return FakeRedis().install(monkeypatch)


@pytest.fixture
def minio(monkeypatch):
    return FakeMinio().install(monkeypatch)


@pytest.fixture
def processes(monkeypatch):
    return FakeProcesses().install(monkeypatch)


@pytest.fixture
def broker(monkeypatch):
    return FakeBroker().install(monkeypatch)


TASK_MODULES = (monitoring_tasks, latex_tasks, transcode_tasks, job_support)


@pytest.fixture
def task_sessions(db_session, monkeypatch):
    factory = async_sessionmaker(
        db_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    for module in TASK_MODULES:
        monkeypatch.setattr(module, "create_celery_session", lambda: factory)
    return factory


async def _reload(db, model, row_id):
    return (
        await db.execute(
            select(model)
            .where(model.id == row_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def _all(db, model, *where):
    stmt = select(model).execution_options(populate_existing=True)
    for clause in where:
        stmt = stmt.where(clause)
    return list((await db.execute(stmt)).scalars().all())


async def _user(db, name):
    return await AuthService().create_user(
        username=name,
        email=f"{name}@example.com",
        password="testpassword123",
        full_name=name.title(),
        db=db,
    )


async def _source(db, source_type="file", is_active=True):
    source = DocumentSource(
        name=f"src-{uuid4().hex[:8]}",
        source_type=source_type,
        config={},
        is_active=is_active,
    )
    db.add(source)
    await db.flush()
    return source


async def _document(db, source=None, **overrides):
    source = source or await _source(db)
    fields = {
        "title": "Prefetching Survey",
        "content": "Hardware prefetchers predict future misses.",
        "content_hash": uuid4().hex,
        "source_id": source.id,
        "source_identifier": f"doc-{uuid4().hex[:6]}",
        "file_type": "pdf",
    }
    fields.update(overrides)
    doc = Document(**fields)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _prefs(db, user, **fields):
    prefs = NotificationPreferences(user_id=user.id, **fields)
    db.add(prefs)
    await db.commit()
    return prefs


def _ago(**delta):
    return datetime.utcnow() - timedelta(**delta)


# --------------------------------------------------------------------------
# health_check
# --------------------------------------------------------------------------


@pytest.fixture
def healthy_world(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    return {
        "qdrant": FakeQdrant().install(monkeypatch),
        "http": FakeHttp({"api.deepseek.com": 200}).install(monkeypatch),
        "disk": FakeDisk(0.5).install(monkeypatch),
    }


async def test_health_check_reports_every_dependency_healthy(
    task_sessions, db_session, healthy_world
):
    await _source(db_session)
    await _source(db_session)
    await db_session.commit()

    report = await monitoring_tasks._async_health_check()

    assert report["overall_status"] == "healthy", report
    services = report["services"]
    assert services["database"]["status"] == "healthy"
    assert "2 sources configured" in services["database"]["message"]
    # A worker that had served no search is initialised, not reported degraded.
    assert healthy_world["qdrant"].initialized == 1
    assert services["vector_store"] == {
        "status": "healthy",
        "message": "Vector store operational, 42 chunks indexed",
    }
    assert services["llm"]["status"] == "healthy"
    assert services["llm"]["provider"] == "deepseek"
    assert healthy_world["http"].requested == ["https://api.deepseek.com/v1/models"]
    assert services["disk_space"]["status"] == "healthy"
    assert services["disk_space"]["usage_percent"] == 50.0


async def test_health_check_reports_a_database_that_cannot_be_reached(
    monkeypatch, healthy_world, tmp_path
):
    url = f"sqlite+aiosqlite:///{tmp_path / 'missing-dir' / 'db.sqlite'}"
    dead = async_sessionmaker(
        create_async_engine(url, poolclass=NullPool), class_=AsyncSession
    )
    monkeypatch.setattr(monitoring_tasks, "create_celery_session", lambda: dead)

    report = await monitoring_tasks._async_health_check()

    assert report["overall_status"] == "unhealthy"
    assert report["services"]["database"]["status"] == "unhealthy"
    assert "unable to open database file" in report["services"]["database"]["error"]
    # The other checks still ran.
    assert report["services"]["vector_store"]["status"] == "healthy"


async def test_health_check_reports_an_unreachable_vector_store(
    task_sessions, db_session, monkeypatch, healthy_world
):
    FakeQdrant(fail_with=ConnectionError("qdrant:6333 refused")).install(monkeypatch)

    report = await monitoring_tasks._async_health_check()

    assert report["overall_status"] == "unhealthy"
    assert report["services"]["vector_store"] == {
        "status": "unhealthy",
        "error": "qdrant:6333 refused",
    }


async def test_health_check_reports_an_llm_provider_that_answers_an_error(
    task_sessions, db_session, monkeypatch, healthy_world
):
    FakeHttp({"api.deepseek.com": 503}).install(monkeypatch)

    report = await monitoring_tasks._async_health_check()

    assert report["overall_status"] == "degraded"
    assert report["services"]["llm"] == {
        "status": "unhealthy",
        "message": "deepseek provider unavailable",
        "provider": "deepseek",
    }


async def test_health_check_asks_the_configured_provider_not_ollama(
    task_sessions, db_session, monkeypatch, healthy_world
):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    # OpenAI answers; nothing listens on the Ollama address.
    http = FakeHttp({"api.openai.com": 200}).install(monkeypatch)

    report = await monitoring_tasks._async_health_check()

    assert report["services"]["llm"]["status"] == "healthy", (
        report["services"]["llm"],
        http.requested,
    )
    assert report["overall_status"] == "healthy"


async def test_health_check_flags_a_nearly_full_disk(
    task_sessions, db_session, monkeypatch, healthy_world
):
    healthy_world["disk"].used_fraction = 0.95

    report = await monitoring_tasks._async_health_check()

    assert report["services"]["disk_space"]["status"] == "critical"
    assert report["overall_status"] == "unhealthy"


async def test_health_check_without_a_data_directory_says_unknown(
    task_sessions, db_session, monkeypatch, healthy_world, tmp_path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = await monitoring_tasks._async_health_check()

    assert report["services"]["disk_space"]["status"] == "unknown"
    assert report["overall_status"] == "healthy"


# --------------------------------------------------------------------------
# generate_stats
# --------------------------------------------------------------------------


async def test_generate_stats_counts_real_rows(
    task_sessions, db_session, test_user, monkeypatch
):
    monkeypatch.setattr(vector_store_service, "_initialized", False)
    files = await _source(db_session, "file")
    await _source(db_session, "gitlab", is_active=False)
    await _document(db_session, files, is_processed=True, summary="A summary.")
    await _document(db_session, files, is_processed=True, summary="")
    await _document(db_session, files, is_processed=True)
    await _document(db_session, files, is_processed=False, processing_error="boom")
    await _document(db_session, files, is_processed=False)
    await _document(
        db_session, files, is_processed=True, summary="old", created_at=_ago(days=30)
    )

    active = ChatSession(user_id=test_user.id, last_message_at=_ago(hours=2))
    stale = ChatSession(user_id=test_user.id, last_message_at=_ago(days=3))
    db_session.add_all([active, stale])
    await db_session.flush()
    for i in range(3):
        db_session.add(ChatMessage(session_id=active.id, content=f"m{i}", role="user"))
    await db_session.commit()

    stats = await monitoring_tasks._async_generate_stats()

    assert "error" not in stats, stats
    assert stats["documents"] == {
        "total": 6,
        "processed": 4,
        "failed": 1,
        "pending": 1,
        "success_rate": 66.67,
        "without_summary": 2,
    }
    assert stats["chat"] == {
        "total_sessions": 2,
        "active_sessions_24h": 1,
        "total_messages": 3,
        "avg_messages_per_session": 1.5,
    }
    assert stats["sources"] == {
        "total": 2,
        "active": 1,
        "by_type": {"file": 1, "gitlab": 1},
    }
    assert stats["vector_store"] == {"status": "not_initialized"}
    assert stats["processing"]["total_documents_last_7_days"] == 5


async def test_generate_stats_on_an_empty_database(task_sessions, db_session):
    stats = await monitoring_tasks._async_generate_stats()

    assert stats["documents"]["total"] == 0
    assert stats["documents"]["success_rate"] == 0
    assert stats["chat"]["avg_messages_per_session"] == 0
    assert stats["sources"] == {"total": 0, "active": 0, "by_type": {}}
    assert stats["processing"]["documents_last_7_days"] == []


async def test_generate_stats_reports_a_database_it_cannot_read(monkeypatch, tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'missing-dir' / 'db.sqlite'}"
    dead = async_sessionmaker(
        create_async_engine(url, poolclass=NullPool), class_=AsyncSession
    )
    monkeypatch.setattr(monitoring_tasks, "create_celery_session", lambda: dead)

    stats = await monitoring_tasks._async_generate_stats()

    assert "unable to open database file" in stats["error"]
    assert "documents" not in stats


# --------------------------------------------------------------------------
# lint_recent_research_notes_citations
# --------------------------------------------------------------------------


UNDER_CITED = (
    "# Findings\n"
    "\n"
    "Prefetching helps streaming loops [[S1]].\n"
    "Tiling keeps the working set in L1 [[S3]].\n"
    "Branch predictors learn loop exits.\n"
)

WELL_CITED = (
    "# Findings\n"
    "\n"
    "Prefetching helps streaming loops [[S1]].\n"
    "Tiling keeps the working set in L1 [[S2]].\n"
    "\n"
    "## Sources\n"
)


async def _note(db, user, content, sources, updated_at=None, **fields):
    note = ResearchNote(
        user_id=user.id,
        title="Cache study",
        content_markdown=content,
        source_document_ids=[str(d.id) for d in sources],
        updated_at=updated_at or _ago(hours=1),
        **fields,
    )
    db.add(note)
    await db.commit()
    return note


async def test_lint_flags_an_under_cited_note_and_notifies_its_owner(
    task_sessions, db_session, test_user, redis_fake
):
    s1 = await _document(db_session, title="Survey", url="https://a.example/s1")
    s2 = await _document(db_session, title="Tiling")
    note = await _note(db_session, test_user, UNDER_CITED, [s1, s2])

    result = await monitoring_tasks._async_lint_recent_research_notes_citations()

    assert result["processed"] == 1
    assert result["updated"] == 1
    assert result["notified"] == 1
    row = await _reload(db_session, ResearchNote, note.id)
    lint = row.attribution["lint"]
    assert lint["sources"] == [
        {"key": "S1", "doc_id": str(s1.id), "title": "Survey", "url": s1.url},
        {"key": "S2", "doc_id": str(s2.id), "title": "Tiling", "url": None},
    ]
    assert lint["used_citation_keys"] == ["S1", "S3"]
    assert lint["unknown_citation_keys"] == ["S3"]
    assert lint["bibliography_present"] is False
    assert lint["total_citable_lines"] == 3
    assert lint["cited_citable_lines"] == 2
    assert lint["uncited_examples"] == [
        {"line_no": 5, "line": "Branch predictors learn loop exits."}
    ]
    assert lint["notified_reasons"] == [
        "low_coverage<0.7",
        "unknown_citation_keys",
        "missing_bibliography",
    ]

    (notification,) = await _all(db_session, Notification)
    assert notification.user_id == test_user.id
    assert notification.notification_type == (
        NotificationType.RESEARCH_NOTE_CITATION_ISSUE
    )
    assert notification.priority == "high"
    assert notification.related_entity_id == note.id
    assert notification.message == (
        "Cited lines: 67% · Unknown keys: S3 · Missing bibliography (## Sources)"
    )
    assert redis_fake.on(f"notifications:{test_user.id}")


async def test_lint_stores_a_clean_report_without_notifying(
    task_sessions, db_session, test_user, redis_fake
):
    s1 = await _document(db_session)
    s2 = await _document(db_session)
    note = await _note(db_session, test_user, WELL_CITED, [s1, s2])

    result = await monitoring_tasks._async_lint_recent_research_notes_citations()

    assert (result["updated"], result["notified"]) == (1, 0)
    lint = (await _reload(db_session, ResearchNote, note.id)).attribution["lint"]
    assert lint["line_citation_coverage"] == 1.0
    assert lint["unknown_citation_keys"] == []
    assert lint["bibliography_present"] is True
    assert "notified_at" not in lint
    assert await _all(db_session, Notification) == []


async def test_lint_respects_a_user_who_turned_citation_notices_off(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user, notify_research_note_citation_issues=False)
    s1 = await _document(db_session)
    note = await _note(db_session, test_user, UNDER_CITED, [s1])

    result = await monitoring_tasks._async_lint_recent_research_notes_citations()

    assert (result["updated"], result["notified"]) == (1, 0)
    lint = (await _reload(db_session, ResearchNote, note.id)).attribution["lint"]
    assert lint["notify_settings"]["enabled"] is False
    assert await _all(db_session, Notification) == []


async def test_lint_counts_a_note_with_no_sources_and_leaves_it_alone(
    task_sessions, db_session, test_user, redis_fake
):
    note = await _note(db_session, test_user, UNDER_CITED, [])
    old = await _note(db_session, test_user, UNDER_CITED, [], updated_at=_ago(days=3))

    result = await monitoring_tasks._async_lint_recent_research_notes_citations()

    assert result["processed"] == 1  # the three-day-old note is outside the window
    assert result["missing_sources"] == 1
    assert result["updated"] == 0
    assert (await _reload(db_session, ResearchNote, note.id)).attribution is None
    assert (await _reload(db_session, ResearchNote, old.id)).attribution is None


async def test_lint_on_no_recent_notes(task_sessions, db_session):
    result = await monitoring_tasks._async_lint_recent_research_notes_citations()
    assert (result["processed"], result["updated"], result["notified"]) == (0, 0, 0)


async def test_lint_does_not_relint_a_note_nobody_has_edited(
    task_sessions, db_session, test_user, redis_fake
):
    s1 = await _document(db_session)
    edited_at = _ago(hours=1)
    note = await _note(db_session, test_user, UNDER_CITED, [s1], updated_at=edited_at)

    await monitoring_tasks._async_lint_recent_research_notes_citations()
    second = await monitoring_tasks._async_lint_recent_research_notes_citations()

    assert (second["skipped"], second["updated"]) == (1, 0)
    row = await _reload(db_session, ResearchNote, note.id)
    assert row.updated_at.replace(tzinfo=None) == edited_at


# --------------------------------------------------------------------------
# sync_experiment_runs
# --------------------------------------------------------------------------


async def _experiment(db, user, *, job_status, results=None, run_status="running"):
    # Outside the citation lint's 24h window, so only this task sees it.
    note = await _note(db, user, "# Plan\n", [], updated_at=_ago(days=3))
    plan = ExperimentPlan(
        user_id=user.id,
        research_note_id=note.id,
        title="Prefetcher sweep",
        plan={"steps": []},
    )
    db.add(plan)
    await db.flush()
    job = AgentJob(
        name="sweep",
        goal="Measure prefetchers",
        user_id=user.id,
        status=job_status,
        progress=40,
        results=results,
        started_at=_ago(hours=2),
        completed_at=_ago(minutes=5) if job_status != "running" else None,
    )
    db.add(job)
    await db.flush()
    run = ExperimentRun(
        user_id=user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Stride vs ISB",
        status=run_status,
    )
    db.add(run)
    await db.commit()
    return plan, job, run


async def test_sync_completes_a_run_whose_job_completed_and_notifies_once(
    task_sessions, db_session, test_user, redis_fake
):
    exp = {"note": "Stride wins on streaming kernels.", "final_phase": "evaluate"}
    _, job, run = await _experiment(
        db_session,
        test_user,
        job_status="completed",
        results={"experiment_run": exp},
    )

    result = await monitoring_tasks._async_sync_experiment_runs()

    assert (result["processed"], result["updated"], result["missing_job"]) == (1, 1, 0)
    row = await _reload(db_session, ExperimentRun, run.id)
    assert row.status == "completed"
    assert row.progress == 100
    assert row.results == exp
    assert row.summary == "Stride wins on streaming kernels."
    assert row.completed_at is not None and row.started_at is not None

    (notification,) = await _all(db_session, Notification)
    assert notification.notification_type == NotificationType.EXPERIMENT_RUN_UPDATE
    assert notification.title == "Experiment run completed"
    assert notification.message == "Stride vs ISB · completed · phase evaluate"
    assert notification.action_url == f"/autonomous-agents?job={job.id}"
    assert notification.data["status"] == "completed"

    # A finished run is no longer selected; nothing is announced twice.
    again = await monitoring_tasks._async_sync_experiment_runs()
    assert again["processed"] == 0
    assert len(await _all(db_session, Notification)) == 1


async def test_sync_fails_a_run_whose_job_failed_with_a_high_priority_notice(
    task_sessions, db_session, test_user, redis_fake
):
    _, _, run = await _experiment(db_session, test_user, job_status="failed")

    await monitoring_tasks._async_sync_experiment_runs()

    row = await _reload(db_session, ExperimentRun, run.id)
    assert (row.status, row.progress) == ("failed", 40)
    (notification,) = await _all(db_session, Notification)
    assert (notification.title, notification.priority) == (
        "Experiment run failed",
        "high",
    )


async def test_sync_moves_a_planned_run_to_running_without_notifying(
    task_sessions, db_session, test_user, redis_fake
):
    _, _, run = await _experiment(
        db_session, test_user, job_status="running", run_status="planned"
    )

    result = await monitoring_tasks._async_sync_experiment_runs()

    assert result["updated"] == 1
    row = await _reload(db_session, ExperimentRun, run.id)
    assert (row.status, row.progress) == ("running", 40)
    assert await _all(db_session, Notification) == []


async def test_sync_refuses_a_job_that_belongs_to_someone_else(
    task_sessions, db_session, test_user, redis_fake
):
    stranger = await _user(db_session, "stranger")
    _, job, run = await _experiment(db_session, test_user, job_status="completed")
    job.user_id = stranger.id
    await db_session.commit()

    result = await monitoring_tasks._async_sync_experiment_runs()

    assert (result["processed"], result["missing_job"], result["updated"]) == (1, 1, 0)
    assert (await _reload(db_session, ExperimentRun, run.id)).status == "running"
    assert await _all(db_session, Notification) == []


async def test_sync_respects_a_user_who_turned_run_updates_off(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user, notify_experiment_run_updates=False)
    _, _, run = await _experiment(db_session, test_user, job_status="cancelled")

    await monitoring_tasks._async_sync_experiment_runs()

    assert (await _reload(db_session, ExperimentRun, run.id)).status == "cancelled"
    assert await _all(db_session, Notification) == []


async def test_sync_on_no_linked_runs(task_sessions, db_session):
    result = await monitoring_tasks._async_sync_experiment_runs()
    assert (result["processed"], result["updated"]) == (0, 0)


# --------------------------------------------------------------------------
# emit_queue_urgency_alerts
# --------------------------------------------------------------------------


async def _awaiting_approval(db, user, minutes):
    job = AgentJob(
        name="Merge the patch",
        goal="Fix the flaky test",
        job_type="coding",
        user_id=user.id,
        status="paused",
        current_phase="awaiting_approval",
        results={"approval_checkpoint": {"message": "Approve merging the patch"}},
        created_at=_ago(minutes=minutes + 10),
        last_activity_at=_ago(minutes=minutes),
    )
    db.add(job)
    await db.commit()
    return job


async def _alerts(db):
    return await _all(
        db,
        Notification,
        Notification.notification_type == NotificationType.QUEUE_URGENCY_ALERT,
    )


async def test_queue_alert_for_an_approval_waiting_past_its_sla_is_raised_once(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user)
    job = await _awaiting_approval(db_session, test_user, minutes=90)

    first = await monitoring_tasks._async_emit_queue_urgency_alerts()

    assert (first["processed_users"], first["emitted"]) == (1, 1)
    (alert,) = await _alerts(db_session)
    assert alert.title == "Queue alert: Merge the patch"
    assert alert.priority == "normal"
    assert alert.related_entity_id == job.id
    assert alert.data["queue_key"] == f"approval:{job.id}"
    assert alert.data["sla_bucket"] == "at_risk"
    assert alert.action_url == (
        f"/autonomous-agents?tab=queue&job={job.id}"
        "&queue_item_type=approval_checkpoint&queue_sla=at_risk"
    )

    second = await monitoring_tasks._async_emit_queue_urgency_alerts()

    assert (second["emitted"], second["skipped"]) == (0, 1)
    assert len(await _alerts(db_session)) == 1


async def test_queue_alert_is_raised_again_when_the_approval_becomes_overdue(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user)
    job = await _awaiting_approval(db_session, test_user, minutes=90)
    await monitoring_tasks._async_emit_queue_urgency_alerts()

    job.last_activity_at = _ago(minutes=300)
    await db_session.commit()
    result = await monitoring_tasks._async_emit_queue_urgency_alerts()

    assert result["emitted"] == 1
    latest = max(await _alerts(db_session), key=lambda n: n.created_at)
    assert latest.data["sla_bucket"] == "overdue"
    assert latest.priority == "high"


async def test_queue_alert_ignores_an_approval_still_inside_its_sla(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user)
    await _awaiting_approval(db_session, test_user, minutes=10)

    result = await monitoring_tasks._async_emit_queue_urgency_alerts()

    assert (result["processed_users"], result["emitted"]) == (1, 0)
    assert await _alerts(db_session) == []


async def test_queue_alert_respects_a_user_who_turned_queue_alerts_off(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user, notify_queue_urgency_alerts=False)
    await _awaiting_approval(db_session, test_user, minutes=300)

    result = await monitoring_tasks._async_emit_queue_urgency_alerts()

    assert result["emitted"] == 0
    assert await _alerts(db_session) == []


async def test_queue_alerts_with_no_users(task_sessions, db_session):
    result = await monitoring_tasks._async_emit_queue_urgency_alerts()
    assert (result["processed_users"], result["emitted"]) == (0, 0)


def test_queue_overdue_reminder_reads_a_postgres_timestamp():
    now = datetime.utcnow()
    item = {
        "queue_key": "approval:x",
        "item_type": "approval_checkpoint",
        "sla_bucket": "overdue",
        "escalation_level": "high",
    }
    existing = Notification(
        notification_type=NotificationType.QUEUE_URGENCY_ALERT,
        data={
            "queue_key": "approval:x",
            "sla_bucket": "overdue",
            "escalation_level": "high",
        },
        created_at=datetime.now(timezone.utc) - timedelta(hours=7),
    )

    assert (
        monitoring_tasks._queue_alert_should_emit(
            item=item,
            existing_notifications=[existing],
            reminder_cooldown_hours=6,
            now=now,
        )
        is True
    )


async def test_queue_alert_tells_someone_about_a_run_blocked_for_days(
    task_sessions, db_session, test_user, redis_fake
):
    await _prefs(db_session, test_user)
    job = AgentJob(
        name="survey: writeup",
        goal="Write up the survey",
        user_id=test_user.id,
        status="paused",
        current_phase="blocked_needs_input",
        results={"blocked": {"reason": "no new findings", "resumable": True}},
        created_at=_ago(days=9),
        last_activity_at=_ago(days=9),
    )
    db_session.add(job)
    await db_session.commit()

    result = await monitoring_tasks._async_emit_queue_urgency_alerts()

    assert result["emitted"] == 1
    (alert,) = await _alerts(db_session)
    assert alert.data["queue_key"] == f"blocked:{job.id}"


# --------------------------------------------------------------------------
# compile_latex_project_job
# --------------------------------------------------------------------------


MAIN_TEX = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    "\\input{intro}\n"
    "\\end{document}\n"
)


async def _latex(db, user, tex=MAIN_TEX, files=None, minio=None, **job_fields):
    project = LatexProject(user_id=user.id, title="Paper", tex_source=tex)
    db.add(project)
    await db.flush()
    for name, content in (files or {}).items():
        path = f"latex/{project.id}/{name}"
        if minio is not None:
            minio.objects[path] = content
        db.add(
            LatexProjectFile(
                project_id=project.id,
                filename=name,
                file_path=path,
                file_size=len(content),
            )
        )
    fields = {"status": "queued", "safe_mode": True, "preferred_engine": "pdflatex"}
    fields.update(job_fields)
    job = LatexCompileJob(user_id=user.id, project_id=project.id, **fields)
    db.add(job)
    await db.commit()
    return project, job


@pytest.fixture
def latex_task(monkeypatch):
    task = latex_tasks.compile_latex_project_job
    return task, _fake_update_state(monkeypatch, task)


async def test_latex_job_compiles_uploads_and_records_the_pdf(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, states = latex_task
    intro = b"Prefetchers predict misses.\n"
    project, job = await _latex(
        db_session, test_user, files={"intro.tex": intro}, minio=minio
    )

    result = await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    assert result == {
        "job_id": str(job.id),
        "status": "succeeded",
        "success": True,
        "engine": "pdflatex",
    }
    # The project's own files were beside main.tex when the compiler ran.
    assert processes.seen_files["intro.tex"] == intro
    row = await _reload(db_session, LatexCompileJob, job.id)
    assert row.status == "succeeded"
    assert row.pdf_file_path == f"{project.id}/paper.pdf"
    assert minio.objects[row.pdf_file_path] == FakeProcesses.PDF
    assert minio.content_types[row.pdf_file_path] == "application/pdf"
    assert "Output written on main.pdf" in row.log
    assert row.violations == []
    assert row.started_at is not None and row.finished_at is not None
    proj = await _reload(db_session, LatexProject, project.id)
    assert proj.pdf_file_path == row.pdf_file_path
    assert proj.last_compile_engine == "pdflatex"
    assert [s for _, s in states] == [
        "Fetching project files",
        "Compiling LaTeX",
        "Uploading PDF",
    ]


async def test_latex_job_records_a_compile_error_with_its_log(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, _ = latex_task
    tex = "\\documentclass{article}\n\\begin{document}\n\\undefinedmacro\n\\end{document}\n"
    project, job = await _latex(db_session, test_user, tex=tex)

    result = await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    assert result == {"job_id": str(job.id), "status": "failed", "success": False}
    row = await _reload(db_session, LatexCompileJob, job.id)
    assert row.status == "failed"
    assert row.pdf_file_path is None
    assert "Undefined control sequence" in row.log
    assert row.engine == "pdflatex"
    assert minio.objects == {}
    proj = await _reload(db_session, LatexProject, project.id)
    assert "Undefined control sequence" in proj.last_compile_log
    assert proj.pdf_file_path is None


async def test_latex_job_refuses_unsafe_source_without_running_the_compiler(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, _ = latex_task
    tex = "\\documentclass{article}\n\\immediate\\write18{rm -rf /}\n"
    _, job = await _latex(db_session, test_user, tex=tex)

    await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    row = await _reload(db_session, LatexCompileJob, job.id)
    assert row.status == "failed"
    assert row.violations == ["Disallowed: \\write18"]
    assert row.engine is None
    assert processes.calls == []


async def test_latex_job_names_a_project_file_that_could_not_be_fetched(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, _ = latex_task
    # The row exists, its object does not.
    _, job = await _latex(db_session, test_user, files={"intro.tex": b"x"})

    await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    row = await _reload(db_session, LatexCompileJob, job.id)
    assert row.status == "failed"
    assert row.violations == [
        "Missing project file for \\input{intro} (expected intro.tex)"
    ]
    assert processes.calls == []


async def test_latex_job_without_its_project_fails_cleanly(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, _ = latex_task
    job = LatexCompileJob(user_id=test_user.id, project_id=None, status="queued")
    db_session.add(job)
    await db_session.commit()

    result = await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    assert result["status"] == "failed"
    row = await _reload(db_session, LatexCompileJob, job.id)
    assert (row.log, row.finished_at is not None) == ("LaTeX project not found.", True)


async def test_latex_job_leaves_a_finished_job_alone(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, _ = latex_task
    _, job = await _latex(db_session, test_user, status="succeeded", log="done")

    result = await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    assert result == {"job_id": str(job.id), "status": "succeeded"}
    assert processes.calls == []
    assert (await _reload(db_session, LatexCompileJob, job.id)).log == "done"


async def test_latex_job_that_does_not_exist_raises(
    task_sessions, db_session, latex_task
):
    task, _ = latex_task
    missing = str(uuid4())
    with pytest.raises(ValueError, match=f"LatexCompileJob {missing} not found"):
        await latex_tasks._async_compile_latex_project_job(task, missing)


async def test_latex_job_whose_upload_fails_does_not_blame_the_compiler(
    task_sessions, db_session, test_user, monkeypatch, processes, latex_task
):
    FakeMinio(fail_uploads=ConnectionError("minio:9000 refused")).install(monkeypatch)
    task, _ = latex_task
    project, job = await _latex(db_session, test_user, tex="\\documentclass{article}\n")

    await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    row = await _reload(db_session, LatexCompileJob, job.id)
    assert row.status == "failed"
    assert row.pdf_file_path is None
    assert "Compilation failed" not in row.log
    assert (
        "Output written on main.pdf"
        in (await _reload(db_session, LatexProject, project.id)).last_compile_log
    )


async def test_latex_job_says_when_bibtex_was_needed_but_absent(
    task_sessions, db_session, test_user, minio, processes, latex_task
):
    task, _ = latex_task
    tex = (
        "\\documentclass{article}\n\\begin{document}\nSee \\cite{k}.\n"
        "\\bibliography{refs}\n\\end{document}\n"
    )
    _, job = await _latex(
        db_session, test_user, tex=tex, files={"refs.bib": b"@misc{k,}"}, minio=minio
    )

    await latex_tasks._async_compile_latex_project_job(task, str(job.id))

    row = await _reload(db_session, LatexCompileJob, job.id)
    assert "BibTeX requested but `bibtex` binary is not available" in row.log


# --------------------------------------------------------------------------
# transcode_to_mp4
# --------------------------------------------------------------------------


async def _video(db, minio, name="clip.mov", content=b"moov fake quicktime", **meta):
    doc = await _document(db, file_type="video/quicktime", extra_metadata=meta or None)
    path = f"{doc.id}/{name}"
    doc.file_path = path
    await db.commit()
    if content is not None:
        minio.objects[path] = content
    return doc


async def test_transcode_converts_uploads_and_hands_over_to_transcription(
    task_sessions, db_session, minio, processes, broker, redis_fake
):
    doc = await _video(db_session, minio, is_transcoding=True)

    result = await transcode_tasks._async_transcode_to_mp4(None, str(doc.id))

    mp4 = f"{doc.id}/clip.mp4"
    assert result == {
        "success": True,
        "document_id": str(doc.id),
        "stream_file_path": mp4,
    }
    assert processes.names() == ["ffmpeg", "ffprobe"]
    assert minio.objects[mp4] == FakeProcesses.MP4
    assert minio.content_types[mp4] == "video/mp4"
    row = await _reload(db_session, Document, doc.id)
    assert row.file_path == mp4
    assert row.file_type == "video/mp4"
    assert row.extra_metadata == {
        "original_file_path": f"{doc.id}/clip.mov",
        "stream_file_path": mp4,
        "is_transcoding": False,
        "is_transcoded": True,
        "is_transcribing": True,
    }
    # The original is kept; the temp files are not.
    assert f"{doc.id}/clip.mov" in minio.objects
    assert processes.temp_paths and not any(p.exists() for p in processes.temp_paths)
    # Transcription is queued with limits sized from the probed duration.
    from app.services.media_probe import transcription_time_limits

    (sent,) = broker.sent
    soft, hard = transcription_time_limits(12.5)
    assert sent["args"] == [str(doc.id)]
    assert sent["options"] == {"soft_time_limit": soft, "time_limit": hard}
    stages = [m for m in redis_fake.on(f"transcription_progress:{doc.id}")]
    assert stages[-1]["status"]["is_transcribing"] is True


async def test_transcode_records_an_ffmpeg_failure_on_the_document(
    task_sessions, db_session, minio, broker, redis_fake, monkeypatch
):
    processes = FakeProcesses(ffmpeg_fails=True).install(monkeypatch)
    doc = await _video(db_session, minio, is_transcoding=True)

    result = await transcode_tasks._async_transcode_to_mp4(None, str(doc.id))

    assert result == {
        "success": False,
        "document_id": str(doc.id),
        "error": "ffmpeg_failed",
    }
    row = await _reload(db_session, Document, doc.id)
    assert row.file_path == f"{doc.id}/clip.mov"
    assert row.extra_metadata["is_transcoding"] is False
    assert row.extra_metadata["is_transcoded"] is False
    assert "returned non-zero exit status 1" in row.extra_metadata["transcode_error"]
    assert broker.sent == []
    assert {"type": "error", "document_id": str(doc.id), "error": "ffmpeg_failed"} in (
        redis_fake.on(f"transcription_progress:{doc.id}")
    )
    assert not any(p.exists() for p in processes.temp_paths)


async def test_transcode_marks_an_mp4_as_streamable_without_converting(
    task_sessions, db_session, minio, processes, broker, redis_fake
):
    doc = await _video(db_session, minio, name="clip.mp4")

    result = await transcode_tasks._async_transcode_to_mp4(None, str(doc.id))

    assert result["skipped"] is True
    assert processes.calls == []
    row = await _reload(db_session, Document, doc.id)
    assert row.extra_metadata == {
        "is_transcoded": True,
        "stream_file_path": f"{doc.id}/clip.mp4",
    }


async def test_transcode_skips_what_is_not_a_video_or_is_already_done(
    task_sessions, db_session, minio, processes, broker, redis_fake
):
    pdf = await _video(db_session, minio, name="paper.pdf")
    done = await _video(
        db_session, minio, is_transcoded=True, stream_file_path="x/clip.mp4"
    )
    no_file = await _document(db_session)

    assert (await transcode_tasks._async_transcode_to_mp4(None, str(pdf.id)))["skipped"]
    assert (await transcode_tasks._async_transcode_to_mp4(None, str(done.id)))[
        "skipped"
    ]
    assert (await transcode_tasks._async_transcode_to_mp4(None, str(no_file.id))) == {
        "success": False,
        "error": "no_file_path",
        "document_id": str(no_file.id),
    }
    missing = str(uuid4())
    assert (await transcode_tasks._async_transcode_to_mp4(None, missing))[
        "error"
    ] == "document_not_found"
    assert processes.calls == []
    assert (await _reload(db_session, Document, pdf.id)).extra_metadata is None


async def test_transcode_says_when_the_original_could_not_be_downloaded(
    task_sessions, db_session, minio, processes, broker, redis_fake
):
    doc = await _video(db_session, minio, content=None)

    result = await transcode_tasks._async_transcode_to_mp4(None, str(doc.id))

    assert result["success"] is False
    assert "ffmpeg" not in processes.names()
    row = await _reload(db_session, Document, doc.id)
    assert "download" in row.extra_metadata["transcode_error"].lower()


async def test_transcode_records_a_failure_before_ffmpeg_is_reached(
    task_sessions, db_session, minio, processes, broker, redis_fake, monkeypatch
):
    import tempfile

    doc = await _video(db_session, minio)

    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", no_space)

    result = await transcode_tasks._async_transcode_to_mp4(None, str(doc.id))

    assert result["success"] is False
    row = await _reload(db_session, Document, doc.id)
    assert "No space left on device" in row.extra_metadata["transcode_error"]


async def test_transcode_does_not_claim_transcription_it_could_not_queue(
    task_sessions, db_session, minio, processes, redis_fake, monkeypatch
):
    FakeBroker(fail_with=ConnectionError("redis:6379 refused")).install(monkeypatch)
    doc = await _video(db_session, minio)

    await transcode_tasks._async_transcode_to_mp4(None, str(doc.id))

    row = await _reload(db_session, Document, doc.id)
    assert row.extra_metadata["is_transcoded"] is True
    assert row.extra_metadata.get("is_transcribing") is not True


# --------------------------------------------------------------------------
# The synchronous Celery functions, each in its own event loop
# --------------------------------------------------------------------------


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """A file database reachable from any event loop, as Postgres is."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"

    def factory():
        engine = create_async_engine(url, poolclass=NullPool)
        return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    for module in TASK_MODULES:
        monkeypatch.setattr(module, "create_celery_session", factory)

    async def create():
        engine = create_async_engine(url, poolclass=NullPool)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(create())

    def run(coro_fn):
        async def go():
            async with factory()() as db:
                return await coro_fn(db)

        return asyncio.run(go())

    return run


def _get(run, model, row_id):
    return run(lambda db: _reload(db, model, row_id))


def test_monitoring_tasks_run_through_celery(file_db, redis_fake, healthy_world):
    user = file_db(lambda db: _user(db, "syncuser"))
    file_db(lambda db: _document(db, is_processed=True))
    file_db(lambda db: _prefs(db, user))
    _, _, run = file_db(
        lambda db: _experiment(db, user, job_status="completed", results=None)
    )

    health = monitoring_tasks.health_check.run()
    stats = monitoring_tasks.generate_stats.run()
    lint = monitoring_tasks.lint_recent_research_notes_citations.run()
    synced = monitoring_tasks.sync_experiment_runs.run()
    alerts = monitoring_tasks.emit_queue_urgency_alerts.run()

    assert health["services"]["database"]["status"] == "healthy"
    assert stats["documents"]["processed"] == 1
    assert lint["processed"] == 0
    assert synced["updated"] == 1
    assert _get(file_db, ExperimentRun, run.id).status == "completed"
    assert alerts["processed_users"] == 1
    json.dumps([health, stats, lint, synced, alerts])  # a Celery result is JSON


def test_latex_job_runs_through_celery(
    file_db, minio, processes, latex_task, monkeypatch
):
    user = file_db(lambda db: _user(db, "syncuser"))
    _, job = file_db(
        lambda db: _latex(db, user, files={"intro.tex": b"hi"}, minio=minio)
    )

    result = latex_tasks.compile_latex_project_job.run(str(job.id))

    assert result["status"] == "succeeded"
    assert _get(file_db, LatexCompileJob, job.id).pdf_file_path in minio.objects


def test_transcode_runs_through_celery(file_db, minio, processes, broker, redis_fake):
    doc = file_db(lambda db: _video(db, minio))

    result = transcode_tasks.transcode_to_mp4.run(str(doc.id))

    assert result["success"] is True
    assert _get(file_db, Document, doc.id).file_path == f"{doc.id}/clip.mp4"
    assert len(broker.sent) == 1
