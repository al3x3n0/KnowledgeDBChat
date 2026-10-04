"""The export and synthesis Celery tasks, run for real.

Until 2735dbf neither module was in the worker's ``include`` list, so every
queued export and synthesis job was discarded: nothing here had ever run in a
worker. These tests run the task bodies -- the async implementations and the
synchronous Celery functions via ``.run`` -- against real rows, and judge on
the rows they leave behind.

Only edges that leave the process are replaced, and each replacement binds its
arguments against the real callee's signature, so a call the real thing would
refuse fails here too:

* Redis -- ``job_support.publish_progress`` / ``publish_message`` /
  ``publish_sync``;
* MinIO -- ``MinIOStorageService`` (``initialize``, ``upload_file``,
  ``upload_to_path``, ``delete_file``), the class both ``StorageService()`` and
  the ``storage_service`` singleton are;
* the model -- ``LLMService.generate_response``.

The DOCX, PDF and PPTX builders are real; their bytes are opened again with
python-docx, reportlab's header and python-pptx.

A test marked xfail(strict) is a defect in the application, not in the test.
"""

import asyncio
import inspect
import io
import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.database import Base
from app.models.agent_job import AgentJob
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, DocumentSource
from app.models.export_job import ExportJob
from app.models.research_paper import PaperClaim, ResearchPaper
from app.models.synthesis_job import SynthesisJob
from app.services.auth_service import AuthService
from app.services.export_service import ExportService
from app.services.llm_service import LLMService
from app.services.storage_service import MinIOStorageService
from app.tasks import export_tasks, job_support, synthesis_tasks

pytestmark = pytest.mark.unit

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


# --------------------------------------------------------------------------
# Edges that leave the process
# --------------------------------------------------------------------------


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
        bind_progress = _binding(job_support.publish_progress)
        bind_message = _binding(job_support.publish_message)
        bind_sync = _binding(job_support.publish_sync)

        async def publish_progress(*args, **kwargs):
            a = bind_progress(*args, **kwargs)
            message = {
                "type": "progress",
                "progress": a["progress"],
                "stage": a["stage"],
                "status": a["status"],
            }
            if a.get("error"):
                message["error"] = a["error"]
            json.dumps(message)
            self.messages.append((a["channel"], message))

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
    """An object store in a dict, behind MinIOStorageService's own signatures."""

    def __init__(self, delete_answers=None, fail_uploads=None):
        self.objects = {}
        self.content_types = {}
        self.deleted = []
        self.delete_answers = dict(delete_answers or {})
        self.fail_uploads = fail_uploads

    def install(self, monkeypatch):
        cls = MinIOStorageService
        bind_init = _binding(cls.initialize)
        bind_upload = _binding(cls.upload_file)
        bind_to_path = _binding(cls.upload_to_path)
        bind_delete = _binding(cls.delete_file)
        bind_get = _binding(cls.get_file_content)
        store = self

        async def initialize(*args, **kwargs):
            bind_init(*args, **kwargs)

        async def upload_file(*args, **kwargs):
            a = bind_upload(*args, **kwargs)
            assert isinstance(a["content"], (bytes, bytearray))
            path = f"{a['document_id']}/{a['filename']}"
            store._put(path, a["content"], a.get("content_type"))
            return path

        async def upload_to_path(*args, **kwargs):
            a = bind_to_path(*args, **kwargs)
            assert isinstance(a["content"], (bytes, bytearray))
            assert isinstance(a["object_path"], str)
            store._put(a["object_path"], a["content"], a.get("content_type"))
            return a["object_path"]

        async def delete_file(*args, **kwargs):
            a = bind_delete(*args, **kwargs)
            path = a["object_path"]
            store.deleted.append(path)
            answer = store.delete_answers.get(path, True)
            if answer:
                store.objects.pop(path, None)
            return answer

        async def get_file_content(*args, **kwargs):
            a = bind_get(*args, **kwargs)
            return store.objects[a["object_path"]]

        monkeypatch.setattr(cls, "initialize", initialize)
        monkeypatch.setattr(cls, "upload_file", upload_file)
        monkeypatch.setattr(cls, "upload_to_path", upload_to_path)
        monkeypatch.setattr(cls, "delete_file", delete_file)
        monkeypatch.setattr(cls, "get_file_content", get_file_content)
        return self

    def _put(self, path, content, content_type):
        if self.fail_uploads:
            raise self.fail_uploads
        self.objects[path] = bytes(content)
        self.content_types[path] = content_type


class FakeModel:
    """``LLMService.generate_response``, answering by what the prompt asks for.

    The JSON-array and JSON-object extraction prompts get JSON; anything else
    gets a markdown document that names the sources it was shown, so a test can
    tell a synthesis written from its sources from one written from nothing.
    """

    THEMES = ["Cache locality", "Branch prediction"]
    FINDINGS = ["Prefetching helps streaming loops"]

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def install(self, monkeypatch):
        bind = _binding(LLMService.generate_response)
        model = self

        async def generate_response(*args, **kwargs):
            a = bind(*args, **kwargs)
            prompt = a.get("query") or a.get("prompt") or a.get("user_message")
            assert isinstance(prompt, str) and prompt
            model.calls.append(a)
            if model.fail_with is not None:
                raise model.fail_with
            return model.answer(prompt)

        monkeypatch.setattr(LLMService, "generate_response", generate_response)
        return self

    def answer(self, prompt):
        if "Extract the main themes" in prompt:
            return json.dumps(self.THEMES)
        if "Extract the key findings" in prompt:
            return json.dumps(self.FINDINGS)
        if "Extract structured hypotheses" in prompt:
            return json.dumps(
                {
                    "summary": "Two gaps, one hypothesis.",
                    "hypotheses": [
                        {"id": "h1", "rank": 1, "title": "Stride beats ISB"}
                    ],
                    "gaps": ["No irregular workloads"],
                    "solution_sketches": ["Hybrid prefetcher"],
                }
            )
        return (
            "## Overview\n\n"
            "Synthesized from the sources below.\n\n"
            "## Key Findings\n\n"
            "- Prefetching helps streaming loops [Source: Prefetching Survey]\n"
            "- Tiling keeps the working set in L1\n\n"
            "## Conclusions\n\n"
            "Locality decides most of the outcome."
        )

    def prompts(self):
        return [c.get("query") or "" for c in self.calls]


@pytest.fixture
def redis_fake(monkeypatch):
    return FakeRedis().install(monkeypatch)


@pytest.fixture
def minio(monkeypatch):
    return FakeMinio().install(monkeypatch)


@pytest.fixture
def model(monkeypatch):
    return FakeModel().install(monkeypatch)


def _use_sessions(monkeypatch, factory):
    """Every task module opens its sessions from ``factory``."""
    for module in (export_tasks, synthesis_tasks, job_support):
        monkeypatch.setattr(module, "create_celery_session", lambda: factory)


@pytest.fixture
def task_sessions(db_session, monkeypatch):
    factory = async_sessionmaker(
        db_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    _use_sessions(monkeypatch, factory)
    return factory


async def _reload(db, model, row_id):
    """The row as the task left it (other objects in ``db`` stay loaded)."""
    return (
        await db.execute(
            select(model)
            .where(model.id == row_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


async def _source(db):
    source = DocumentSource(
        name=f"src-{uuid4().hex[:8]}", source_type="file", config={}
    )
    db.add(source)
    await db.flush()
    return source


async def _document(db, **overrides):
    source = await _source(db)
    fields = {
        "title": "Prefetching Survey",
        "content": "Hardware prefetchers predict future misses from past ones.",
        "content_hash": uuid4().hex,
        "source_id": source.id,
        "source_identifier": "survey.pdf",
        "file_type": "pdf",
    }
    fields.update(overrides)
    doc = Document(**fields)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _export_job(db, user, **overrides):
    fields = {
        "user_id": user.id,
        "export_type": "custom",
        "output_format": "docx",
        "source_type": "llm_content",
        "content": "# Results\n\nThe stride prefetcher issued 63127 prefetches.",
        "content_format": "markdown",
        "title": "Prefetcher Report",
        "style": "professional",
        "status": "pending",
        "progress": 0,
    }
    fields.update(overrides)
    job = ExportJob(**fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _synthesis_job(db, user, **overrides):
    fields = {
        "user_id": user.id,
        "job_type": "multi_doc_summary",
        "title": "Prefetching Synthesis",
        "document_ids": [],
        "paper_ids": [],
        "agent_job_ids": [],
        "options": {},
        "output_format": "markdown",
        "output_style": "professional",
        "status": "pending",
        "progress": 0,
    }
    fields.update(overrides)
    job = SynthesisJob(**fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _other_user(db):
    return await AuthService().create_user(
        username="someone_else",
        email="else@example.com",
        password="otherpassword123",
        full_name="Someone Else",
        db=db,
    )


async def _paper(db, owner, title="Irregular Stream Buffer"):
    doc = await _document(db, title=f"{title} (pdf)")
    paper = ResearchPaper(
        user_id=owner.id,
        document_id=doc.id,
        arxiv_id="2401.00001",
        title=title,
        abstract="ISB linearises irregular streams into structural addresses.",
        mechanisms=["structural address space"],
    )
    db.add(paper)
    await db.flush()
    db.add(
        PaperClaim(
            paper_id=paper.id,
            statement="ISB covers 40% more misses than stride on pointer chasing",
            rank=1,
        )
    )
    await db.commit()
    await db.refresh(paper)
    return paper


def _docx_text(blob):
    import docx

    document = docx.Document(io.BytesIO(blob))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def _pptx_text(blob):
    import pptx

    deck = pptx.Presentation(io.BytesIO(blob))
    out = []
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                out.append(shape.text_frame.text)
    return "\n".join(out)


# ==========================================================================
# Export
# ==========================================================================

UPLOAD_SIGNATURE = (
    "app/services/export_service.py:158 calls "
    "self.storage.upload_file(file_bytes, file_path, content_type=...), but "
    "MinIOStorageService.upload_file is (document_id, filename, content, "
    "content_type): the bytes land in document_id, the path in filename, and "
    "`content` is missing, so every export raises TypeError after building the "
    "document and ends failed. Fix: upload_to_path(file_path, file_bytes, "
    "content_type)."
)


@pytest.mark.parametrize(
    "output_format,content_format,content,expect",
    [
        (
            "docx",
            "markdown",
            "# Results\n\nThe stride prefetcher issued 63127 prefetches.",
            "63127 prefetches",
        ),
        (
            "docx",
            "html",
            "<h1>Results</h1><p>ISB identified zero candidates.</p>"
            "<ul><li>pfIssued = 0</li></ul>",
            "ISB identified zero candidates",
        ),
        ("pdf", "plain", "First paragraph.\n\nSecond paragraph.", None),
    ],
)
async def test_custom_content_export_completes_with_the_uploaded_file(
    db_session,
    test_user,
    task_sessions,
    redis_fake,
    minio,
    output_format,
    content_format,
    content,
    expect,
):
    job = await _export_job(
        db_session,
        test_user,
        output_format=output_format,
        content_format=content_format,
        content=content,
    )

    await export_tasks._process_export_async(str(job.id))

    row = await _reload(db_session, ExportJob, job.id)
    assert row.status == "completed", row.error
    assert row.file_path == f"exports/{test_user.id}/{job.id}.{output_format}"
    assert row.file_path in minio.objects
    blob = minio.objects[row.file_path]
    assert row.file_size == len(blob)
    assert row.completed_at is not None
    if output_format == "docx":
        assert minio.content_types[row.file_path] == DOCX_MIME
        text = _docx_text(blob)
        assert "Prefetcher Report" in text
        assert expect in text
    else:
        assert minio.content_types[row.file_path] == "application/pdf"
        assert blob.startswith(b"%PDF")
    final = redis_fake.on(f"export:{job.id}:progress")[-1]
    assert final["status"] == "completed" and final["progress"] == 100


async def test_document_summary_export_carries_the_summary(
    db_session, test_user, task_sessions, redis_fake, minio
):
    doc = await _document(
        db_session, summary="Stride prefetching reaches 2.1x on streaming kernels."
    )
    job = await _export_job(
        db_session,
        test_user,
        export_type="document_summary",
        source_type="document",
        source_id=doc.id,
        content=None,
    )

    await export_tasks._process_export_async(str(job.id))

    row = await _reload(db_session, ExportJob, job.id)
    assert row.status == "completed", row.error
    text = _docx_text(minio.objects[row.file_path])
    assert "Prefetching Survey" in text
    assert "2.1x on streaming kernels" in text
    assert "survey.pdf" in text  # the metadata table's Source row


async def test_chat_session_export_contains_the_conversation(
    db_session, test_user, task_sessions, redis_fake, minio
):
    session = ChatSession(user_id=test_user.id, title="Prefetcher chat")
    db_session.add(session)
    await db_session.flush()
    db_session.add_all(
        [
            ChatMessage(
                session_id=session.id,
                role="user",
                content="Why did ISB match the baseline to the cycle?",
            ),
            ChatMessage(
                session_id=session.id,
                role="assistant",
                content="Its pfIdentified counter reads **0**: it never engaged.",
                source_documents=[{"title": "gem5 stats.txt"}],
            ),
        ]
    )
    await db_session.commit()
    job = await _export_job(
        db_session,
        test_user,
        export_type="chat",
        source_type="chat_session",
        source_id=session.id,
        content=None,
    )

    await export_tasks._process_export_async(str(job.id))

    row = await _reload(db_session, ExportJob, job.id)
    assert row.status == "completed", row.error
    text = _docx_text(minio.objects[row.file_path])
    assert "Why did ISB match the baseline to the cycle?" in text
    assert "it never engaged" in text
    assert "gem5 stats.txt" in text


async def test_export_that_fails_to_upload_is_failed_with_the_cause(
    db_session, test_user, task_sessions, redis_fake, monkeypatch
):
    """Whatever the upload's arguments, a store that refuses fails the job."""
    store = FakeMinio(fail_uploads=ConnectionError("minio:9000 refused")).install(
        monkeypatch
    )
    job = await _export_job(db_session, test_user)

    await export_tasks._process_export_async(str(job.id))

    row = await _reload(db_session, ExportJob, job.id)
    assert row.status == "failed"
    assert row.error  # names the cause, whichever it is
    assert row.completed_at is not None
    assert row.file_path is None and row.file_size is None
    assert store.objects == {}
    final = redis_fake.on(f"export:{job.id}:progress")[-1]
    assert final["status"] == "failed"
    assert final["error"] == row.error


@pytest.mark.parametrize(
    "overrides,cause",
    [
        ({"source_type": "bogus"}, "Unknown source type: bogus"),
        (
            {"source_type": "document", "source_id": None, "content": None},
            "Document None not found",
        ),
    ],
)
async def test_export_with_an_unusable_source_fails_naming_it(
    db_session, test_user, task_sessions, redis_fake, minio, overrides, cause
):
    job = await _export_job(db_session, test_user, **overrides)

    await export_tasks._process_export_async(str(job.id))

    row = await _reload(db_session, ExportJob, job.id)
    assert row.status == "failed"
    assert row.error == cause
    assert row.started_at is not None and row.completed_at is not None
    assert minio.objects == {}
    messages = redis_fake.on(f"export:{job.id}:progress")
    assert messages[0]["stage"] == "starting"
    assert messages[-1] == {
        "type": "progress",
        "progress": row.progress,
        "stage": "error",
        "status": "failed",
        "error": cause,
    }


async def test_missing_export_job_is_a_no_op(
    db_session, test_user, task_sessions, redis_fake, minio
):
    await export_tasks._process_export_async(str(uuid4()))

    assert redis_fake.messages == []
    assert minio.objects == {}


async def test_cancelled_export_job_is_left_alone(
    db_session, test_user, task_sessions, redis_fake, minio
):
    job = await _export_job(db_session, test_user, status="cancelled")

    await export_tasks._process_export_async(str(job.id))

    row = await _reload(db_session, ExportJob, job.id)
    assert row.status == "cancelled"
    assert row.started_at is None and row.error is None
    assert redis_fake.messages == []
    assert minio.objects == {}


@pytest.mark.parametrize("output_format", ["docx", "pdf"])
def test_a_partial_custom_theme_from_the_api_builds(output_format):
    from app.schemas.export import ExportTheme

    theme = ExportTheme(title_color="#aa0000").model_dump()  # as the endpoint does
    items = [{"type": "paragraph", "text": "Body text under a custom theme."}]

    blob = ExportService()._build_document(
        title="Themed",
        content_items=items,
        output_format=output_format,
        style="professional",
        custom_theme=theme,
    )

    if output_format == "docx":
        assert "Body text under a custom theme." in _docx_text(blob)
    else:
        assert blob.startswith(b"%PDF")


# --------------------------------------------------------------------------
# The synchronous Celery functions, each in its own event loop
# --------------------------------------------------------------------------


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """A file database reachable from any event loop, as Postgres is.

    ``.run`` calls ``asyncio.run``, which makes a new loop; each call gets a
    fresh engine on it, which is what ``create_celery_session`` does.
    """
    url = f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"

    def factory():
        engine = create_async_engine(url, poolclass=NullPool)
        return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    for module in (export_tasks, synthesis_tasks, job_support):
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


def _seed_user(run):
    async def make(db):
        return await AuthService().create_user(
            username="syncuser",
            email="sync@example.com",
            password="testpassword123",
            full_name="Sync User",
            db=db,
        )

    return run(make)


def _get(run, model, row_id):
    async def fetch(db):
        return (
            await db.execute(select(model).where(model.id == row_id))
        ).scalar_one_or_none()

    return run(fetch)


def test_process_export_task_run_completes_a_job(file_db, redis_fake, minio):
    user = _seed_user(file_db)
    job = file_db(lambda db: _export_job(db, user))

    export_tasks.process_export_task.run(str(job.id))

    row = _get(file_db, ExportJob, job.id)
    assert row.status == "completed", row.error
    assert row.file_size == len(minio.objects[row.file_path])


def test_process_export_task_run_records_a_failure_without_raising(
    file_db, redis_fake, minio
):
    user = _seed_user(file_db)
    job = file_db(lambda db: _export_job(db, user, source_type="bogus"))

    export_tasks.process_export_task.run(str(job.id))  # must not raise or retry

    row = _get(file_db, ExportJob, job.id)
    assert row.status == "failed"
    assert row.error == "Unknown source type: bogus"


def test_process_export_task_run_on_a_missing_job(file_db, redis_fake, minio):
    export_tasks.process_export_task.run(str(uuid4()))
    assert redis_fake.messages == []


def _aged(job, days):
    job.created_at = datetime.utcnow() - timedelta(days=days)
    return job


def test_cleanup_old_exports_deletes_files_then_rows(file_db, monkeypatch):
    store = FakeMinio(delete_answers={"exports/stuck.docx": False}).install(monkeypatch)
    user = _seed_user(file_db)

    async def seed(db):
        jobs = {
            "old_done": _aged(
                ExportJob(
                    user_id=user.id,
                    export_type="custom",
                    output_format="docx",
                    source_type="llm_content",
                    title="old",
                    status="completed",
                    file_path="exports/old.docx",
                ),
                40,
            ),
            "old_stuck": _aged(
                ExportJob(
                    user_id=user.id,
                    export_type="custom",
                    output_format="docx",
                    source_type="llm_content",
                    title="stuck",
                    status="completed",
                    file_path="exports/stuck.docx",
                ),
                40,
            ),
            "old_running": _aged(
                ExportJob(
                    user_id=user.id,
                    export_type="custom",
                    output_format="docx",
                    source_type="llm_content",
                    title="running",
                    status="processing",
                ),
                40,
            ),
            "recent": _aged(
                ExportJob(
                    user_id=user.id,
                    export_type="custom",
                    output_format="docx",
                    source_type="llm_content",
                    title="recent",
                    status="completed",
                    file_path="exports/recent.docx",
                ),
                2,
            ),
        }
        db.add_all(jobs.values())
        await db.commit()
        return {k: j.id for k, j in jobs.items()}

    ids = file_db(seed)

    result = export_tasks.cleanup_old_exports.run(days=30)

    assert result == {"deleted": 1, "kept": 1}
    assert sorted(store.deleted) == ["exports/old.docx", "exports/stuck.docx"]
    assert _get(file_db, ExportJob, ids["old_done"]) is None
    assert _get(file_db, ExportJob, ids["old_stuck"]) is not None
    assert _get(file_db, ExportJob, ids["old_running"]) is not None
    assert _get(file_db, ExportJob, ids["recent"]) is not None


# ==========================================================================
# Synthesis
# ==========================================================================


async def _run_synthesis(job, user):
    return await synthesis_tasks._async_execute_synthesis(
        None, str(job.id), str(user.id)
    )


@pytest.mark.parametrize(
    "job_type,metadata_key,expected",
    [
        ("multi_doc_summary", "themes_found", FakeModel.THEMES),
        ("comparative_analysis", "documents_compared", 2),
        ("theme_extraction", "themes_extracted", FakeModel.THEMES),
        ("knowledge_synthesis", "key_findings", FakeModel.FINDINGS),
        ("research_report", "word_count", None),
        ("executive_brief", "report_type", "executive_brief"),
        ("decision_memo", "report_type", "decision_memo"),
    ],
)
async def test_document_synthesis_completes_from_its_documents(
    db_session,
    test_user,
    task_sessions,
    redis_fake,
    minio,
    model,
    job_type,
    metadata_key,
    expected,
):
    first = await _document(db_session)
    second = await _document(
        db_session,
        title="Tiling Notes",
        content="Loop tiling bounds the working set.",
    )
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type=job_type,
        document_ids=[str(first.id), str(second.id)],
        topic="prefetching",
    )

    outcome = await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    assert outcome["success"] is True
    assert outcome["word_count"] == row.result_metadata["word_count"] > 0
    assert row.progress == 100 and row.completed_at is not None
    assert "Locality decides most of the outcome." in row.result_content
    if expected is not None:
        assert row.result_metadata[metadata_key] == expected
    # Written from its sources: both reached the model.
    main_prompt = model.prompts()[0]
    assert "[Document: Prefetching Survey]" in main_prompt
    assert "[Document: Tiling Notes]" in main_prompt
    assert row.file_path is None and minio.objects == {}  # markdown: no file
    channel = redis_fake.on(f"synthesis_progress:{job.id}")
    assert channel[-1]["type"] == "complete"
    assert channel[-1]["result"]["word_count"] == row.result_metadata["word_count"]


async def test_theme_extraction_produces_its_theme_map(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type="theme_extraction",
        document_ids=[str(doc.id)],
        topic="prefetching",
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    assert row.result_metadata["themes_extracted"] == FakeModel.THEMES
    assert len(row.artifacts) == 1
    artifact = row.artifacts[0]
    assert artifact["format"] == "mermaid" and artifact["title"] == "Theme Map"
    assert "Cache locality" in artifact["code"]
    assert "Branch prediction" in artifact["code"]


@pytest.mark.parametrize("output_format", ["docx", "pdf", "pptx"])
async def test_synthesis_file_output_is_uploaded_and_recorded(
    db_session, test_user, task_sessions, redis_fake, minio, model, output_format
):
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session,
        test_user,
        document_ids=[str(doc.id)],
        output_format=output_format,
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    path = f"synthesis/{test_user.id}/synthesis_{job.id}.{output_format}"
    assert row.file_path == path
    blob = minio.objects[path]
    assert row.file_size == len(blob)
    if output_format == "docx":
        assert minio.content_types[path] == DOCX_MIME
        text = _docx_text(blob)
        assert "Prefetching Synthesis" in text
        assert "Locality decides most of the outcome." in text
    elif output_format == "pdf":
        assert minio.content_types[path] == "application/pdf"
        assert blob.startswith(b"%PDF")
    else:
        assert minio.content_types[path] == PPTX_MIME
        text = _pptx_text(blob)
        assert "Prefetching Synthesis" in text
        assert "Key Findings" in text
        assert "Tiling keeps the working set in L1" in text


async def test_synthesis_whose_file_cannot_be_stored_does_not_report_success(
    db_session, test_user, task_sessions, redis_fake, model, monkeypatch
):
    FakeMinio(fail_uploads=ConnectionError("minio:9000 refused")).install(monkeypatch)
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session, test_user, document_ids=[str(doc.id)], output_format="docx"
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "failed"
    assert "minio:9000 refused" in (row.error or "")


async def test_gap_analysis_from_papers_records_structured_hypotheses(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    paper = await _paper(db_session, test_user)
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type="gap_analysis_hypotheses",
        paper_ids=[str(paper.id)],
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    meta = row.result_metadata
    assert meta["source_kind"] == "papers"
    assert meta["structured_hypotheses"][0]["title"] == "Stride beats ISB"
    assert meta["structured_gaps"] == ["No irregular workloads"]
    main_prompt = model.prompts()[0]
    assert "Irregular Stream Buffer" in main_prompt
    assert "ISB covers 40% more misses" in main_prompt


async def test_gap_analysis_does_not_read_another_users_papers(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    stranger = await _other_user(db_session)
    paper = await _paper(db_session, stranger, title="Private Unpublished Result")
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type="gap_analysis_hypotheses",
        paper_ids=[str(paper.id)],
    )

    await _run_synthesis(job, test_user)

    assert not any("Private Unpublished Result" in p for p in model.prompts())
    assert not any("ISB covers 40% more misses" in p for p in model.prompts())


async def _agent_run(db, owner, name, findings):
    job = AgentJob(
        name=name,
        goal="Measure stride prefetching",
        job_type="research",
        user_id=owner.id,
        status="completed",
        config={},
        results={"findings": findings},
    )
    db.add(job)
    await db.commit()
    return job


MY_RUN_FINDINGS = [
    {"type": "document", "title": "A retrieved page"},
    {
        "type": "benchmark_result",
        "title": "Stride on streaming kernel",
        "fastest_ms": 15,
        "n": 5,
    },
]


async def test_agent_runs_are_cited_only_when_they_are_the_callers(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    """Gap analysis is the one job type that writes from ``sources``."""
    stranger = await _other_user(db_session)
    mine = await _agent_run(db_session, test_user, "My benchmark run", MY_RUN_FINDINGS)
    theirs = await _agent_run(
        db_session,
        stranger,
        "Their secret run",
        [{"type": "benchmark_result", "title": "Secret measurement"}],
    )
    empty = await _agent_run(
        db_session,
        test_user,
        "Run that only searched",
        [{"type": "paper", "title": "Only a retrieved paper"}],
    )
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type="gap_analysis_hypotheses",
        document_ids=[str(doc.id)],
        agent_job_ids=[str(mine.id), str(theirs.id), str(empty.id), "not-a-uuid"],
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    prompt = model.prompts()[0]
    assert "[Document: Prefetching Survey]" in prompt
    assert "[Document: Run: My benchmark run]" in prompt
    assert "Their secret run" not in prompt
    assert "Run that only searched" not in prompt  # nothing of its own
    assert row.result_metadata["documents_analyzed"] == 2


async def test_agent_run_findings_reach_the_prompt_with_their_numbers(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    stranger = await _other_user(db_session)
    mine = await _agent_run(db_session, test_user, "My benchmark run", MY_RUN_FINDINGS)
    theirs = await _agent_run(
        db_session,
        stranger,
        "Their secret run",
        [{"type": "benchmark_result", "title": "Secret measurement"}],
    )
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type="gap_analysis_hypotheses",
        document_ids=[str(doc.id)],
        agent_job_ids=[str(mine.id), str(theirs.id), "not-a-uuid"],
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    prompt = model.prompts()[0]
    assert "Stride on streaming kernel | fastest_ms=15, n=5" in prompt
    assert "A retrieved page" not in prompt  # retrieval is not a finding
    assert "Secret measurement" not in prompt


@pytest.mark.parametrize("with_document", [True, False])
async def test_document_synthesis_writes_from_the_agent_runs_it_cites(
    db_session, test_user, task_sessions, redis_fake, minio, model, with_document
):
    mine = await _agent_run(db_session, test_user, "My benchmark run", MY_RUN_FINDINGS)
    document_ids = [str((await _document(db_session)).id)] if with_document else []
    job = await _synthesis_job(
        db_session,
        test_user,
        job_type="multi_doc_summary",
        document_ids=document_ids,
        agent_job_ids=[str(mine.id)],
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    prompt = model.prompts()[0]
    assert "[Document: Run: My benchmark run]" in prompt
    assert row.result_metadata["documents_analyzed"] == 1 + len(document_ids)


async def test_synthesis_with_no_sources_fails_naming_it(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    job = await _synthesis_job(db_session, test_user, document_ids=[str(uuid4())])

    outcome = await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "failed"
    assert row.error == "No sources found for synthesis"
    assert row.completed_at is not None
    assert outcome == {
        "success": False,
        "job_id": str(job.id),
        "error": "No sources found for synthesis",
    }
    assert model.calls == []
    last = redis_fake.on(f"synthesis_progress:{job.id}")[-1]
    assert last == {
        "type": "error",
        "job_id": str(job.id),
        "error": "No sources found for synthesis",
    }


async def test_synthesis_whose_model_fails_is_failed_with_the_models_error(
    db_session, test_user, task_sessions, redis_fake, minio, monkeypatch
):
    FakeModel(fail_with=RuntimeError("deepseek: 401 invalid api key")).install(
        monkeypatch
    )
    doc = await _document(db_session)
    job = await _synthesis_job(db_session, test_user, document_ids=[str(doc.id)])

    outcome = await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "failed"
    assert row.error == "deepseek: 401 invalid api key"
    assert outcome["success"] is False
    assert row.result_content is None


async def test_synthesis_of_an_unknown_job_type_fails_naming_it(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session, test_user, job_type="poetry", document_ids=[str(doc.id)]
    )

    await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert row.status == "failed"
    assert row.error == "Unknown job type: poetry"


async def test_missing_synthesis_job(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    outcome = await synthesis_tasks._async_execute_synthesis(
        None, str(uuid4()), str(test_user.id)
    )

    assert outcome == {"success": False, "error": "Job not found"}
    assert redis_fake.messages == [] and model.calls == []


async def test_cancelled_synthesis_job_is_left_alone(
    db_session, test_user, task_sessions, redis_fake, minio, model
):
    doc = await _document(db_session)
    job = await _synthesis_job(
        db_session, test_user, status="cancelled", document_ids=[str(doc.id)]
    )

    outcome = await _run_synthesis(job, test_user)

    row = await _reload(db_session, SynthesisJob, job.id)
    assert outcome == {"success": False, "error": "Job cancelled"}
    assert row.status == "cancelled" and row.started_at is None
    assert model.calls == [] and redis_fake.messages == []


def test_execute_synthesis_task_run_completes_a_job(file_db, redis_fake, minio, model):
    user = _seed_user(file_db)

    async def seed(db):
        doc = await _document(db)
        return await _synthesis_job(
            db, user, document_ids=[str(doc.id)], output_format="docx"
        )

    job = file_db(seed)

    outcome = synthesis_tasks.execute_synthesis_task.run(str(job.id), str(user.id))

    row = _get(file_db, SynthesisJob, job.id)
    assert row.status == "completed", row.error
    assert outcome["success"] is True and outcome["job_id"] == str(job.id)
    assert row.file_size == len(minio.objects[row.file_path])


def _old_synthesis(user, status, file_path, days=40):
    return _aged(
        SynthesisJob(
            user_id=user.id,
            job_type="multi_doc_summary",
            title=f"{status} job",
            document_ids=[],
            status=status,
            file_path=file_path,
        ),
        days,
    )


def test_cleanup_old_synthesis_jobs_deletes_finished_jobs_and_files(
    file_db, monkeypatch
):
    store = FakeMinio().install(monkeypatch)
    user = _seed_user(file_db)

    async def seed(db):
        jobs = {
            "done": _old_synthesis(user, "completed", "synthesis/a.docx"),
            "failed": _old_synthesis(user, "failed", None),
            "running": _old_synthesis(user, "synthesizing", None),
            "recent": _old_synthesis(user, "completed", "synthesis/b.pdf", days=1),
        }
        db.add_all(jobs.values())
        await db.commit()
        return {k: j.id for k, j in jobs.items()}

    ids = file_db(seed)

    result = synthesis_tasks.cleanup_old_synthesis_jobs.run(days=30)

    # The shared prune helper reports rows kept, not files deleted.
    assert result == {"success": True, "jobs_deleted": 2, "kept": 0}
    assert store.deleted == ["synthesis/a.docx"]
    assert _get(file_db, SynthesisJob, ids["done"]) is None
    assert _get(file_db, SynthesisJob, ids["failed"]) is None
    assert _get(file_db, SynthesisJob, ids["running"]) is not None
    assert _get(file_db, SynthesisJob, ids["recent"]) is not None


def test_cleanup_old_synthesis_jobs_keeps_a_row_whose_file_survived(
    file_db, monkeypatch
):
    store = FakeMinio(delete_answers={"synthesis/stuck.docx": False}).install(
        monkeypatch
    )
    user = _seed_user(file_db)

    async def seed(db):
        job = _old_synthesis(user, "completed", "synthesis/stuck.docx")
        db.add(job)
        await db.commit()
        return job.id

    job_id = file_db(seed)

    result = synthesis_tasks.cleanup_old_synthesis_jobs.run(days=30)

    assert store.deleted == ["synthesis/stuck.docx"]
    assert _get(file_db, SynthesisJob, job_id) is not None
    assert result["kept"] == 1
