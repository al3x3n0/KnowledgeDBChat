"""Generation, template, source and workflow-drafting tools, through the real handlers.

Nineteen tools that no test called. Each is run the way a chat turn or an
autonomous job runs it -- `provider._handlers[name](params, ctx)` on the
provider `agent_tool_dispatch` builds -- against the in-memory database and
real rows, and judged on what it leaves behind: the rows written, the task
queued, the request sent.

Only the true edges are replaced, and each replacement refuses what the real
callee would refuse:

* the model -- `FakeLLM` binds every call against the real
  `LLMService.generate_response` signature;
* Celery -- `RecordingTask` binds every `.delay` against the real task's own
  signature, and the task is imported under the name the tool imports it by;
* arXiv -- `httpx.MockTransport` behind the real `ArxivSearchService`;
* the vector store and the document search, which bind against the real
  `search` signatures and answer in the shape the real ones answer in.

A refusal is an `{"error": ...}` result or a `ValueError`: `_execute_tool`
turns a raised exception into a failed call carrying its message, and several
of these handlers refuse by raising on purpose. Anything else escaping
(`TypeError`, `AttributeError`, a lazy load outside a greenlet) is a crash.
"""

import hashlib
import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJob
from app.models.agent_retraction import AgentRetraction, RetractionKind
from app.models.document import Document, DocumentChunk, DocumentSource
from app.models.presentation import PresentationJob
from app.models.reading_list import ReadingList, ReadingListItem
from app.models.template import TemplateJob
from app.models.workflow import UserTool, Workflow, WorkflowEdge, WorkflowNode
from app.services import agent_prior_findings
from app.services import agent_service as agent_service_module
from app.services import arxiv_search_service as arxiv_module
from app.services import content_generation_service as content_module
from app.services import workflow_synthesis_service as synthesis_module
from app.services.agent_service import AgentService
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_agent_service_analytics_content_provider,
    build_agent_service_research_provider,
    build_agent_service_workflow_provider,
    build_autonomous_research_provider,
    build_autonomous_workspace_mutation_provider,
)
from app.services.agent_tools import get_tool_by_name
from app.services.arxiv_search_service import ArxivSearchService
from app.services.auth_service import AuthService
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.search_service import SearchService
from app.services.vector_store import VectorStore

pytestmark = pytest.mark.unit

TOOLS = (
    "draft_email",
    "generate_meeting_notes",
    "generate_documentation",
    "generate_executive_summary",
    "generate_diagram",
    "generate_chart_data",
    "export_data",
    "get_reading_lists",
    "retract_finding",
    "list_template_jobs",
    "get_template_job_status",
    "start_template_fill",
    "summarize_documents_in_source",
    "generate_slides_for_source",
    "enrich_arxiv_metadata_for_source",
    "find_related_papers",
    "monitor_arxiv_topic",
    "create_workflow_from_description",
    "propose_workflow_from_description",
)

_GENERATE_RESPONSE = inspect.signature(LLMService.generate_response)
_VECTOR_SEARCH = inspect.signature(VectorStore.search)
_DOCUMENT_SEARCH = inspect.signature(SearchService.search)


# --- the edges -------------------------------------------------------------


class FakeLLM:
    """The model. Answers with `replies` in turn (the last one repeats)."""

    def __init__(self, *replies, error=None):
        self.replies = list(replies) or ["A reply."]
        self.error = error
        self.attempts = 0
        self.calls = []

    async def generate_response(self, *args, **kwargs):
        self.attempts += 1
        bound = _GENERATE_RESPONSE.bind(self, *args, **kwargs)
        self.calls.append(dict(bound.arguments))
        if self.error is not None:
            raise self.error
        index = min(len(self.calls), len(self.replies)) - 1
        return self.replies[index]

    @property
    def last(self):
        return self.calls[-1]

    @property
    def text(self):
        """Everything the model was shown on the last call."""
        return "\n".join(
            str(self.last.get(key) or "")
            for key in ("system_prompt", "query", "prompt", "user_message")
        )


class RecordingTask:
    """Stands in for `.delay` on a real Celery task, with that task's signature."""

    def __init__(self, task):
        self._signature = inspect.signature(task.run)
        self.calls = []

    def delay(self, *args, **kwargs):
        bound = self._signature.bind(*args, **kwargs)
        self.calls.append(dict(bound.arguments))
        return SimpleNamespace(id=f"task-{len(self.calls)}")


def _record(monkeypatch, task):
    recorder = RecordingTask(task)
    monkeypatch.setattr(task, "delay", recorder.delay)
    return recorder


class FakeVectorStore:
    """`VectorStore`, answering in the shape the real `search` answers in."""

    hits: list = []
    queries: list = []

    async def search(self, *args, **kwargs):
        bound = _VECTOR_SEARCH.bind(self, *args, **kwargs)
        type(self).queries.append(bound.arguments["query"])
        return [
            {
                "id": f"doc_{doc_id}_chunk_0",
                "content": "chunk",
                "page_content": "chunk",
                "metadata": {"document_id": str(doc_id), "chunk_index": 0},
                "score": 0.9,
            }
            for doc_id in type(self).hits
        ]


class FakeDocumentSearch:
    """`search_service`, for the content generators' `search_query`."""

    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    async def search(self, *args, **kwargs):
        bound = _DOCUMENT_SEARCH.bind(self, *args, **kwargs)
        self.calls.append(dict(bound.arguments))
        return self.results, len(self.results), 1


def _feed(entries):
    body = "".join(
        "<entry>"
        f"<id>http://arxiv.org/abs/{e['id']}</id>"
        f"<title>{e['title']}</title>"
        "<summary>An abstract.</summary>"
        f"<published>{e.get('published', '2026-09-01T00:00:00Z')}</published>"
        "<author><name>A. Author</name></author>"
        "</entry>"
        for e in entries
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{len(entries)}</opensearch:totalResults>"
        f"{body}</feed>"
    )


class FakeArxiv:
    """arXiv's API behind the real `ArxivSearchService`."""

    def __init__(self, monkeypatch, answer=None):
        self.requests = []
        self._answer = answer or (lambda query: [])
        transport = httpx.MockTransport(self._handle)

        def _client(**kwargs):
            return httpx.AsyncClient(transport=transport, **kwargs)

        monkeypatch.setattr(arxiv_module, "httpx", SimpleNamespace(AsyncClient=_client))

    def _handle(self, request):
        params = dict(request.url.params)
        self.requests.append(params)
        return httpx.Response(200, text=_feed(self._answer(params["search_query"])))

    @property
    def queries(self):
        return [r["search_query"] for r in self.requests]


# --- rows ------------------------------------------------------------------


async def _stranger(db, username="stranger"):
    return await AuthService().create_user(
        username=username,
        email=f"{username}@example.com",
        password="strangerpassword123",
        full_name="Somebody Else",
        db=db,
    )


async def _source(db, name="arXiv import", source_type="arxiv", config=None):
    source = DocumentSource(name=name, source_type=source_type, config=config or {})
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _doc(db, source, title, **fields):
    fields.setdefault("content", f"Body of {title}.")
    fields.setdefault("is_processed", True)
    fields.setdefault("source_identifier", f"id-{uuid4().hex}")
    doc = Document(
        title=title,
        source_id=source.id,
        content_hash=hashlib.sha256(uuid4().bytes).hexdigest(),
        **fields,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _template_job(db, user, **fields):
    fields.setdefault("template_file_path", "templates/report.docx")
    fields.setdefault("template_filename", "report.docx")
    fields.setdefault("source_document_ids", [str(uuid4())])
    job = TemplateJob(user_id=user.id, **fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _agent_job(db, user, **fields):
    fields.setdefault("name", "A run")
    fields.setdefault("goal", "Measure the prefetcher")
    fields.setdefault("job_type", "research")
    fields.setdefault("status", "completed")
    job = AgentJob(user_id=user.id, **fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _all(db, model):
    return list((await db.execute(select(model))).scalars().all())


# --- calling ---------------------------------------------------------------


def _service():
    service = AgentService.__new__(AgentService)
    service.document_service = DocumentService.__new__(DocumentService)
    return service


_CHAT_BUILDERS = (
    build_agent_service_analytics_content_provider,
    build_agent_service_research_provider,
    build_agent_service_workflow_provider,
)


async def _chat(tool, params, db, user_id=None):
    """Run a chat tool through the provider that answers it."""
    service = _service()
    for build in _CHAT_BUILDERS:
        provider = build(service)
        if tool in provider._handlers:
            ctx = AgentToolExecutionContext(
                mode="chat", db=db, service=service, user_id=user_id
            )
            return await provider._handlers[tool](params, ctx)
    raise AssertionError(f"no chat provider answers {tool}")


async def _research(tool, params, db, job, state=None):
    executor = SimpleNamespace(arxiv_service=ArxivSearchService())
    provider = build_autonomous_research_provider(executor)
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=job.user_id,
        job=job,
        state=state if state is not None else {},
    )
    return await provider._handlers[tool](params, ctx)


async def _retract(params, db, job, state):
    provider = build_autonomous_workspace_mutation_provider(SimpleNamespace())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=job.user_id,
        job=job,
        state=state,
    )
    return await provider._handlers["retract_finding"](params, ctx)


async def _refusal(awaitable):
    """The message of a refusal: an error result, or a raised ValueError."""
    try:
        result = await awaitable
    except ValueError as exc:
        return str(exc) or "ValueError"
    assert isinstance(result, dict) and result.get("error"), result
    assert not result.get("success"), result
    return str(result["error"])


def _plain(result):
    """The result survives JSON, which is what a tool result is sent as."""
    return json.loads(json.dumps(result))


# --- every tool is answered ------------------------------------------------


class TestEveryToolIsDeclaredAndAnswered:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_the_tool_is_declared(self, tool):
        assert get_tool_by_name(tool) is not None

    @pytest.mark.parametrize("tool", TOOLS)
    def test_some_provider_answers_it(self, tool):
        service = _service()
        answered = set()
        for build in _CHAT_BUILDERS:
            answered |= set(build(service)._handlers)
        answered |= set(build_autonomous_research_provider(SimpleNamespace())._handlers)
        answered |= set(
            build_autonomous_workspace_mutation_provider(SimpleNamespace())._handlers
        )
        assert tool in answered


# --- draft_email -----------------------------------------------------------


@pytest.fixture
def content_llm(monkeypatch):
    llm = FakeLLM("Subject: Q3 numbers\nHi Dana,\n\nHere they are.\n\nBest,\nSam")
    monkeypatch.setattr(content_module.content_generation_service, "llm", llm)
    return llm


@pytest.fixture
def document_search(monkeypatch):
    search = FakeDocumentSearch(
        [{"title": "Prefetcher survey", "snippet": "Stride beats ISB on SPEC."}]
    )
    monkeypatch.setattr(content_module, "search_service", search)
    return search


class TestDraftEmail:
    async def test_a_draft_carries_the_subject_and_body_the_model_wrote(
        self, db_session, content_llm
    ):
        result = await _chat(
            "draft_email",
            {"subject": "Q3 numbers", "recipient": "Dana", "context": "Be brief."},
            db_session,
        )

        assert result["subject"] == "Q3 numbers"
        assert result["body"].startswith("Hi Dana,")
        assert result["recipient"] == "Dana"
        assert "Dana" in content_llm.text
        assert "Be brief." in content_llm.text
        _plain(result)

    async def test_tone_and_length_change_what_the_model_is_asked(
        self, db_session, content_llm
    ):
        await _chat("draft_email", {"subject": "s"}, db_session)
        default = content_llm.text
        await _chat(
            "draft_email",
            {"subject": "s", "tone": "casual", "length": "short"},
            db_session,
        )

        assert content_llm.text != default
        assert "conversational" in content_llm.text
        assert "2-3 paragraphs" in content_llm.text

    async def test_named_documents_reach_the_model(self, db_session, content_llm):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Budget", content="Spend was 4.2M.")

        result = await _chat(
            "draft_email",
            {"subject": "Budget", "document_ids": [str(doc.id)]},
            db_session,
        )

        assert "Spend was 4.2M." in content_llm.text
        assert result["documents_referenced"] == 1

    async def test_a_search_query_brings_search_results_to_the_model(
        self, db_session, content_llm, document_search
    ):
        await _chat(
            "draft_email",
            {"subject": "Prefetchers", "search_query": "stride prefetcher"},
            db_session,
        )

        assert document_search.calls[0]["query"] == "stride prefetcher"
        assert "Stride beats ISB on SPEC." in content_llm.text

    async def test_a_subject_is_required(self, db_session, content_llm):
        await _refusal(_chat("draft_email", {"recipient": "Dana"}, db_session))
        assert content_llm.attempts == 0

    async def test_a_malformed_document_id_is_refused(self, db_session, content_llm):
        await _refusal(
            _chat(
                "draft_email",
                {"subject": "s", "document_ids": ["not-a-uuid"]},
                db_session,
            )
        )
        assert content_llm.attempts == 0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "ContentGenerationService.draft_email reports "
            "documents_referenced as len(document_ids), counting ids that "
            "matched no document."
        ),
    )
    async def test_a_document_that_does_not_exist_is_not_counted_as_referenced(
        self, db_session, content_llm
    ):
        result = await _chat(
            "draft_email",
            {"subject": "s", "document_ids": [str(uuid4())]},
            db_session,
        )

        assert result.get("error") or result["documents_referenced"] == 0

    async def test_a_document_with_no_content_does_not_crash_the_draft(
        self, db_session, content_llm
    ):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Empty upload", content=None)

        result = await _chat(
            "draft_email",
            {"subject": "s", "document_ids": [str(doc.id)]},
            db_session,
        )

        assert isinstance(result, dict)

    async def test_an_empty_reply_is_not_reported_as_a_draft(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            content_module.content_generation_service, "llm", FakeLLM("")
        )

        await _refusal(_chat("draft_email", {"subject": "Q3 numbers"}, db_session))


# --- generate_meeting_notes ------------------------------------------------


NOTES = (
    "## Summary\nWe shipped.\n\n## Action Items\n"
    "- [ ] Write the postmortem (@lee)\n- [x] Close the incident\n"
)


class TestGenerateMeetingNotes:
    async def test_notes_come_from_the_transcript(self, db_session, monkeypatch):
        llm = FakeLLM(NOTES)
        monkeypatch.setattr(content_module.content_generation_service, "llm", llm)

        result = await _chat(
            "generate_meeting_notes",
            {
                "transcript": "Lee: the rollout finished at noon.",
                "meeting_title": "Rollout review",
                "participants": ["Lee", "Sam"],
            },
            db_session,
        )

        assert result["title"] == "Rollout review"
        assert result["participants"] == ["Lee", "Sam"]
        assert result["notes"] == NOTES
        assert result["action_items"] == [
            "Write the postmortem (@lee)",
            "Close the incident",
        ]
        assert "the rollout finished at noon" in llm.text
        assert "Lee, Sam" in llm.text
        _plain(result)

    async def test_notes_come_from_a_named_document(self, db_session, monkeypatch):
        llm = FakeLLM(NOTES)
        monkeypatch.setattr(content_module.content_generation_service, "llm", llm)
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Standup", content="Sam: blocked on CI.")

        await _chat(
            "generate_meeting_notes", {"document_ids": [str(doc.id)]}, db_session
        )

        assert "Sam: blocked on CI." in llm.text

    async def test_each_section_switch_removes_its_section(
        self, db_session, monkeypatch
    ):
        llm = FakeLLM(NOTES)
        monkeypatch.setattr(content_module.content_generation_service, "llm", llm)

        await _chat("generate_meeting_notes", {"transcript": "t"}, db_session)
        assert "- Action Items" in llm.text and "- Decisions Made" in llm.text

        await _chat(
            "generate_meeting_notes",
            {"transcript": "t", "include_action_items": False},
            db_session,
        )
        assert "- Action Items" not in llm.text and "- Decisions Made" in llm.text

        await _chat(
            "generate_meeting_notes",
            {"transcript": "t", "include_decisions": False},
            db_session,
        )
        assert "- Action Items" in llm.text and "- Decisions Made" not in llm.text

    async def test_nothing_to_summarise_is_refused(self, db_session, content_llm):
        await _refusal(_chat("generate_meeting_notes", {}, db_session))
        assert content_llm.attempts == 0

    async def test_an_unknown_document_is_refused(self, db_session, content_llm):
        await _refusal(
            _chat(
                "generate_meeting_notes",
                {"document_ids": [str(uuid4())]},
                db_session,
            )
        )
        assert content_llm.attempts == 0

    async def test_a_malformed_document_id_is_refused(self, db_session, content_llm):
        await _refusal(
            _chat("generate_meeting_notes", {"document_ids": ["nope"]}, db_session)
        )

    async def test_an_empty_reply_is_not_reported_as_notes(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            content_module.content_generation_service, "llm", FakeLLM("")
        )

        await _refusal(
            _chat("generate_meeting_notes", {"transcript": "Lee: hi."}, db_session)
        )


# --- generate_documentation ------------------------------------------------


class TestGenerateDocumentation:
    async def test_documentation_is_titled_from_its_first_heading(
        self, db_session, monkeypatch
    ):
        llm = FakeLLM("# Cache API\n\nCall `get` to read.")
        monkeypatch.setattr(content_module.content_generation_service, "llm", llm)
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Cache", content="get(key) returns it.")

        result = await _chat(
            "generate_documentation",
            {"topic": "the cache", "document_ids": [str(doc.id)]},
            db_session,
        )

        assert result["title"] == "Cache API"
        assert result["topic"] == "the cache"
        assert result["content"] == "# Cache API\n\nCall `get` to read."
        assert result["word_count"] == 7
        assert "get(key) returns it." in llm.text
        _plain(result)

    async def test_doc_type_audience_and_examples_change_the_request(
        self, db_session, content_llm
    ):
        await _chat("generate_documentation", {"topic": "t"}, db_session)
        default = content_llm.text
        assert "Code examples and usage patterns" in default

        result = await _chat(
            "generate_documentation",
            {
                "topic": "t",
                "doc_type": "how_to",
                "target_audience": "admins",
                "include_examples": False,
            },
            db_session,
        )

        assert "how-to guide" in content_llm.text
        assert "system administrators" in content_llm.text
        assert "Code examples and usage patterns" not in content_llm.text
        assert result["doc_type"] == "how_to"
        assert result["target_audience"] == "admins"

    async def test_a_search_query_brings_source_material(
        self, db_session, content_llm, document_search
    ):
        await _chat(
            "generate_documentation",
            {"topic": "prefetchers", "search_query": "stride"},
            db_session,
        )

        assert document_search.calls[0]["query"] == "stride"
        assert "Stride beats ISB on SPEC." in content_llm.text

    async def test_a_topic_is_required(self, db_session, content_llm):
        await _refusal(_chat("generate_documentation", {}, db_session))
        assert content_llm.attempts == 0

    async def test_a_malformed_document_id_is_refused(self, db_session, content_llm):
        await _refusal(
            _chat(
                "generate_documentation",
                {"topic": "t", "document_ids": ["nope"]},
                db_session,
            )
        )

    async def test_an_empty_reply_is_not_reported_as_documentation(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            content_module.content_generation_service, "llm", FakeLLM("")
        )

        await _refusal(_chat("generate_documentation", {"topic": "t"}, db_session))


# --- generate_executive_summary --------------------------------------------


SUMMARY = "## Overview\nRevenue grew 12% to $4.2M.\nHeadcount is flat.\n"


class TestGenerateExecutiveSummary:
    async def test_a_summary_reports_the_documents_it_read(
        self, db_session, monkeypatch
    ):
        llm = FakeLLM(SUMMARY)
        monkeypatch.setattr(content_module.content_generation_service, "llm", llm)
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Q3", content="Revenue was 4.2M.")

        result = await _chat(
            "generate_executive_summary",
            {"topic": "Q3", "document_ids": [str(doc.id)], "max_length": 120},
            db_session,
        )

        assert result["summary"] == SUMMARY
        assert result["topic"] == "Q3"
        assert result["documents_analyzed"] == 1
        assert result["key_metrics_found"] == ["Revenue grew 12% to $4.2M."]
        assert "Revenue was 4.2M." in llm.text
        assert "120 words" in llm.text
        _plain(result)

    async def test_each_section_switch_removes_its_section(
        self, db_session, content_llm
    ):
        result = await _chat("generate_executive_summary", {"topic": "t"}, db_session)
        assert "Recommendations" in result["sections_included"]
        assert "Key Metrics & Numbers" in result["sections_included"]

        result = await _chat(
            "generate_executive_summary",
            {"topic": "t", "include_recommendations": False, "include_metrics": False},
            db_session,
        )

        assert "Recommendations" not in result["sections_included"]
        assert "Key Metrics & Numbers" not in result["sections_included"]
        assert "- Recommendations" not in content_llm.text
        assert "- Key Metrics" not in content_llm.text

    async def test_a_search_query_brings_source_material(
        self, db_session, content_llm, document_search
    ):
        await _chat(
            "generate_executive_summary", {"search_query": "stride"}, db_session
        )

        assert document_search.calls[0]["query"] == "stride"
        assert "Stride beats ISB on SPEC." in content_llm.text

    async def test_nothing_to_summarise_is_refused(self, db_session, content_llm):
        await _refusal(_chat("generate_executive_summary", {}, db_session))
        assert content_llm.attempts == 0

    async def test_a_malformed_document_id_is_refused(self, db_session, content_llm):
        await _refusal(
            _chat(
                "generate_executive_summary",
                {"topic": "t", "document_ids": ["nope"]},
                db_session,
            )
        )

    async def test_a_line_with_no_number_is_not_a_metric(self, db_session, monkeypatch):
        monkeypatch.setattr(
            content_module.content_generation_service,
            "llm",
            FakeLLM("Costs in $ are unknown.\nMargin rose 3%."),
        )

        result = await _chat("generate_executive_summary", {"topic": "t"}, db_session)

        assert result["key_metrics_found"] == ["Margin rose 3%."]

    async def test_an_empty_reply_is_not_reported_as_a_summary(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(
            content_module.content_generation_service, "llm", FakeLLM("")
        )

        await _refusal(_chat("generate_executive_summary", {"topic": "t"}, db_session))


# --- generate_diagram ------------------------------------------------------


@pytest.fixture
def diagram_llm(monkeypatch):
    llm = FakeLLM('```mermaid\nflowchart TD\n  A["Client"] --> B["API"]\n```')
    monkeypatch.setattr(agent_service_module, "LLMService", lambda: llm)
    return llm


@pytest.fixture
def vector_store(monkeypatch):
    FakeVectorStore.hits = []
    FakeVectorStore.queries = []
    monkeypatch.setattr(agent_service_module, "VectorStore", FakeVectorStore)
    return FakeVectorStore


class TestGenerateDiagram:
    async def test_a_description_becomes_mermaid(
        self, db_session, test_user, diagram_llm
    ):
        result = await _chat(
            "generate_diagram",
            {"source": "description", "description": "A client calls an API."},
            db_session,
            test_user.id,
        )

        assert "error" not in result, result
        assert result["mermaid_code"] == 'flowchart TD\n  A["Client"] --> B["API"]'
        assert result["can_render"] is True
        assert "A client calls an API." in diagram_llm.text
        _plain(result)

    async def test_type_focus_and_detail_change_the_request(
        self, db_session, test_user, diagram_llm
    ):
        result = await _chat(
            "generate_diagram",
            {
                "source": "description",
                "description": "Login.",
                "diagram_type": "sequence",
                "focus": "token refresh",
                "detail_level": "low",
            },
            db_session,
            test_user.id,
        )

        assert "error" not in result, result
        assert "sequence diagram" in diagram_llm.text
        assert "token refresh" in diagram_llm.text
        assert "Keep it simple" in diagram_llm.text
        assert result["diagram_type"] == "sequence"
        assert result["focus"] == "token refresh"
        assert result["detail_level"] == "low"

    async def test_named_documents_are_what_gets_diagrammed(
        self, db_session, test_user, diagram_llm
    ):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Design", content="Worker pulls a queue.")

        result = await _chat(
            "generate_diagram",
            {"source": "documents", "document_ids": [str(doc.id)]},
            db_session,
            test_user.id,
        )

        assert "error" not in result, result
        assert "Worker pulls a queue." in diagram_llm.text
        assert result["source_documents"] == [{"id": str(doc.id), "title": "Design"}]

    async def test_a_search_finds_the_documents_to_diagram(
        self, db_session, test_user, diagram_llm, vector_store
    ):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Design", content="Worker pulls a queue.")
        vector_store.hits = [doc.id]

        await _chat(
            "generate_diagram",
            {"source": "search", "search_query": "queue worker"},
            db_session,
            test_user.id,
        )

        assert vector_store.queries == ["queue worker"]
        assert diagram_llm.attempts == 1

    async def test_an_empty_reply_is_not_reported_as_a_diagram(
        self, db_session, test_user, monkeypatch
    ):
        monkeypatch.setattr(agent_service_module, "LLMService", lambda: FakeLLM(""))

        result = await _chat(
            "generate_diagram",
            {"source": "description", "description": "A client calls an API."},
            db_session,
            test_user.id,
        )

        assert result.get("error")
        assert "Failed to generate diagram" not in result["error"]

    @pytest.mark.parametrize(
        "params",
        [
            {"source": "description"},
            {"source": "description", "description": ""},
            {"source": "documents"},
            {"source": "documents", "document_ids": []},
            {"source": "search"},
            {"source": "gitlab_repo"},
            {"source": "somewhere_else", "description": "x"},
        ],
    )
    async def test_a_source_without_its_input_is_refused(
        self, db_session, test_user, diagram_llm, params
    ):
        await _refusal(_chat("generate_diagram", params, db_session, test_user.id))
        assert diagram_llm.attempts == 0

    async def test_documents_that_cannot_be_found_are_refused(
        self, db_session, test_user, diagram_llm
    ):
        await _refusal(
            _chat(
                "generate_diagram",
                {"source": "documents", "document_ids": ["nope", str(uuid4())]},
                db_session,
                test_user.id,
            )
        )
        assert diagram_llm.attempts == 0

    async def test_a_repository_without_a_gitlab_source_is_refused(
        self, db_session, test_user, diagram_llm
    ):
        message = await _refusal(
            _chat(
                "generate_diagram",
                {"source": "gitlab_repo", "gitlab_project": "group/project"},
                db_session,
                test_user.id,
            )
        )

        assert "GitLab" in message


# --- generate_chart_data ---------------------------------------------------


@pytest.fixture
async def corpus(db_session):
    """Three processed documents across two source types, and one unprocessed."""
    arxiv = await _source(db_session, "arXiv import", "arxiv")
    web = await _source(db_session, "Docs site", "web")
    docs = {
        "a": await _doc(
            db_session,
            arxiv,
            "Stride",
            content="12345",
            file_type="pdf",
            file_size=100,
            author="Ada",
            tags=["prefetch"],
            created_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
        ),
        "b": await _doc(
            db_session,
            arxiv,
            "ISB",
            content="123",
            file_type="pdf",
            file_size=300,
            author="Ada",
            created_at=datetime(2026, 2, 5, tzinfo=timezone.utc),
        ),
        "c": await _doc(
            db_session,
            web,
            "Handbook",
            content="1",
            file_size=50,
            created_at=datetime(2026, 3, 5, tzinfo=timezone.utc),
        ),
        "raw": await _doc(
            db_session,
            web,
            "Not processed yet",
            is_processed=False,
            file_size=999,
            created_at=datetime(2026, 3, 6, tzinfo=timezone.utc),
        ),
    }
    return SimpleNamespace(arxiv=arxiv, web=web, **docs)


class TestGenerateChartData:
    async def test_documents_are_counted_by_source_type(self, db_session, corpus):
        result = await _chat(
            "generate_chart_data",
            {"metric": "document_count", "group_by": "source_type"},
            db_session,
        )

        assert result["labels"] == ["arxiv", "web"]
        assert result["values"] == [2, 1]
        assert result["datasets"][0]["data"] == [2, 1]
        assert result["chart_type"] == "bar"
        _plain(result)

    async def test_the_metric_decides_what_is_summed(self, db_session, corpus):
        size = await _chat(
            "generate_chart_data",
            {"metric": "file_size", "group_by": "source_type"},
            db_session,
        )
        content = await _chat(
            "generate_chart_data",
            {"metric": "content_size", "group_by": "source_type"},
            db_session,
        )

        assert dict(zip(size["labels"], size["values"])) == {"arxiv": 400, "web": 50}
        assert dict(zip(content["labels"], content["values"])) == {
            "arxiv": 8,
            "web": 1,
        }

    async def test_the_grouping_decides_the_labels(self, db_session, corpus):
        by_type = await _chat(
            "generate_chart_data",
            {"metric": "document_count", "group_by": "file_type"},
            db_session,
        )
        by_author = await _chat(
            "generate_chart_data",
            {"metric": "document_count", "group_by": "author"},
            db_session,
        )
        by_date = await _chat(
            "generate_chart_data",
            {"metric": "document_count", "group_by": "date"},
            db_session,
        )

        assert dict(zip(by_type["labels"], by_type["values"])) == {
            "pdf": 2,
            "unknown": 1,
        }
        assert dict(zip(by_author["labels"], by_author["values"])) == {
            "Ada": 2,
            "Unknown": 1,
        }
        assert by_date["labels"] == ["2026-01-05", "2026-02-05", "2026-03-05"]
        assert by_date["values"] == [1, 1, 1]

    async def test_the_date_range_bounds_what_is_counted(self, db_session, corpus):
        result = await _chat(
            "generate_chart_data",
            {
                "metric": "document_count",
                "group_by": "date",
                "date_from": "2026-01-20",
                "date_to": "2026-02-20",
            },
            db_session,
        )

        assert result["labels"] == ["2026-02-05"]

    async def test_the_limit_bounds_the_data_points(self, db_session, corpus):
        result = await _chat(
            "generate_chart_data",
            {"metric": "document_count", "group_by": "source_type", "limit": 1},
            db_session,
        )

        assert result["labels"] == ["arxiv"]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "AnalyticsService.generate_chart_data applies `limit` to "
            "every grouping except group_by='date', whose query has no "
            ".limit()."
        ),
    )
    async def test_the_limit_bounds_a_time_series_too(self, db_session, corpus):
        result = await _chat(
            "generate_chart_data",
            {"metric": "document_count", "group_by": "date", "limit": 2},
            db_session,
        )

        assert len(result["labels"]) == 2

    async def test_the_chart_type_is_carried_through(self, db_session, corpus):
        result = await _chat(
            "generate_chart_data",
            {"chart_type": "pie", "metric": "document_count", "group_by": "author"},
            db_session,
        )

        assert result["chart_type"] == "pie"

    async def test_an_unknown_grouping_is_refused(self, db_session, corpus):
        message = await _refusal(
            _chat(
                "generate_chart_data",
                {"metric": "document_count", "group_by": "colour"},
                db_session,
            )
        )

        assert "colour" in message

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "AnalyticsService.generate_chart_data falls back to "
            "count(Document.id) for a metric it does not know and labels "
            "the result with the name it was given: 'word_count' (which "
            "its own docstring advertises) returns document counts titled "
            "'Word Count'."
        ),
    )
    async def test_an_unknown_metric_is_not_charted_as_a_document_count(
        self, db_session, corpus
    ):
        await _refusal(
            _chat(
                "generate_chart_data",
                {"metric": "word_count", "group_by": "source_type"},
                db_session,
            )
        )

    async def test_an_unreadable_date_is_not_silently_dropped(self, db_session, corpus):
        await _refusal(
            _chat(
                "generate_chart_data",
                {
                    "metric": "document_count",
                    "group_by": "date",
                    "date_from": "last tuesday",
                },
                db_session,
            )
        )


# --- export_data -----------------------------------------------------------


async def _export(db, params, fresh=False):
    """Export; `fresh` empties the session first, as a request's session is.

    The session the fixtures wrote through still holds every source, so a lazy
    load of `doc.source` is answered from memory. A real request starts with
    an empty session, which is what `fresh` reproduces.
    """
    if fresh:
        db.expunge_all()
    return await _chat("export_data", params, db)


class TestExportData:
    async def test_processed_documents_are_exported_with_their_source(
        self, db_session, corpus
    ):
        result = await _export(db_session, {"format": "jsonl"})

        rows = [json.loads(line) for line in result["preview"].splitlines()]
        assert {r["title"] for r in rows} == {"Stride", "ISB", "Handbook"}
        by_title = {r["title"]: r for r in rows}
        assert by_title["Stride"]["source_type"] == "arxiv"
        assert by_title["Handbook"]["source_name"] == "Docs site"
        assert "content" not in by_title["Stride"]
        assert result["content_type"] == "application/x-ndjson"
        assert result["filename"].endswith(".jsonl")
        _plain(result)

    async def test_a_session_that_has_not_loaded_the_sources_can_export(
        self, db_session, corpus
    ):
        result = await _export(db_session, {"format": "jsonl"}, fresh=True)

        rows = [json.loads(line) for line in result["preview"].splitlines()]
        assert {r["source_type"] for r in rows} == {"arxiv", "web"}

    async def test_the_default_format_is_json(self, db_session, corpus):
        result = await _export(db_session, {"limit": 1})

        assert result["format"] == "json"
        assert result["filename"].endswith(".json")
        assert len(json.loads(result["preview"])) == 1

    async def test_csv_has_a_header_and_a_row_per_document(self, db_session, corpus):
        result = await _export(db_session, {"format": "csv"})

        lines = result["preview"].strip().splitlines()
        assert lines[0].startswith("id,title,author,source_type")
        assert len(lines) == 4
        assert result["content_type"] == "text/csv"

    async def test_the_source_filter_narrows_the_export(self, db_session, corpus):
        web_id = str(corpus.web.id)

        result = await _export(db_session, {"format": "jsonl", "source_id": web_id})

        rows = [json.loads(line) for line in result["preview"].splitlines()]
        assert [r["title"] for r in rows] == ["Handbook"]

    async def test_the_tag_filter_narrows_the_export(self, db_session, corpus):
        result = await _export(db_session, {"format": "jsonl", "tag": "prefetch"})

        rows = [json.loads(line) for line in result["preview"].splitlines()]
        assert [r["title"] for r in rows] == ["Stride"]

    async def test_content_is_included_only_when_asked(self, db_session, corpus):
        web_id = str(corpus.web.id)

        result = await _export(
            db_session,
            {"format": "jsonl", "source_id": web_id, "include_content": True},
        )

        row = json.loads(result["preview"])
        assert row["content"] == "1"

    async def test_chunks_are_exported_with_their_own_metadata(
        self, db_session, corpus
    ):
        web_id = str(corpus.web.id)
        db_session.add(
            DocumentChunk(
                document_id=corpus.c.id,
                content="1",
                content_hash="0" * 64,
                chunk_index=0,
                extra_metadata={"page": 3},
            )
        )
        await db_session.commit()

        result = await _export(
            db_session,
            {"format": "jsonl", "source_id": web_id, "include_chunks": True},
        )

        row = json.loads(result["preview"])
        assert row["chunks"] == [
            {"index": 0, "content": "1", "metadata": {"page": 3}},
        ]

    async def test_a_long_export_is_previewed_and_sized(self, db_session, corpus):
        await _doc(db_session, corpus.web, "Long", content="x" * 5000)

        result = await _export(db_session, {"format": "jsonl", "include_content": True})

        assert len(result["preview"]) == 1000
        assert result["size_bytes"] > 5000

    async def test_an_unsupported_format_is_refused(self, db_session, corpus):
        message = await _refusal(_export(db_session, {"format": "xlsx"}))

        assert "xlsx" in message

    async def test_a_malformed_source_id_is_refused(self, db_session, corpus):
        await _refusal(_export(db_session, {"source_id": "not-a-uuid"}))

    async def test_an_empty_knowledge_base_exports_nothing(self, db_session):
        result = await _export(db_session, {"format": "json"})

        assert json.loads(result["preview"]) == []


# --- get_reading_lists -----------------------------------------------------


@pytest.fixture
async def reading(db_session, test_user):
    source = await _source(db_session)
    first = await _doc(db_session, source, "Read me first")
    second = await _doc(db_session, source, "Read me second")
    mine = ReadingList(user_id=test_user.id, name="Prefetchers", description="To read")
    other = ReadingList(user_id=test_user.id, name="Compilers")
    db_session.add_all([mine, other])
    await db_session.flush()
    db_session.add_all(
        [
            ReadingListItem(
                reading_list_id=mine.id, document_id=second.id, position=2, notes="n"
            ),
            ReadingListItem(
                reading_list_id=mine.id,
                document_id=first.id,
                position=1,
                status="read",
                priority=5,
            ),
        ]
    )
    await db_session.commit()
    job = await _agent_job(db_session, test_user, status="running")
    return SimpleNamespace(
        source=source, first=first, second=second, mine=mine, other=other, job=job
    )


class TestGetReadingLists:
    async def test_lists_come_back_with_their_items_in_order(self, db_session, reading):
        result = await _research("get_reading_lists", {}, db_session, reading.job)

        assert result["success"] is True
        data = result["data"]
        assert data["total"] == 2
        lists = {entry["name"]: entry for entry in data["reading_lists"]}
        assert lists["Prefetchers"]["description"] == "To read"
        assert lists["Compilers"]["items"] == []
        items = lists["Prefetchers"]["items"]
        assert [i["document_title"] for i in items] == [
            "Read me first",
            "Read me second",
        ]
        assert items[0]["document_id"] == str(reading.first.id)
        assert items[0]["status"] == "read"
        assert items[0]["priority"] == 5
        assert items[1]["notes"] == "n"
        _plain(result)

    async def test_a_name_selects_one_list(self, db_session, reading):
        result = await _research(
            "get_reading_lists", {"list_name": "Compilers"}, db_session, reading.job
        )

        assert [e["name"] for e in result["data"]["reading_lists"]] == ["Compilers"]

    async def test_a_name_nobody_has_finds_nothing(self, db_session, reading):
        result = await _research(
            "get_reading_lists", {"list_name": "Nope"}, db_session, reading.job
        )

        assert result["data"] == {"reading_lists": [], "total": 0}

    async def test_items_can_be_left_out(self, db_session, reading):
        result = await _research(
            "get_reading_lists", {"include_items": False}, db_session, reading.job
        )

        assert all("items" not in e for e in result["data"]["reading_lists"])
        assert result["data"]["total"] == 2

    async def test_another_users_lists_are_not_shown(self, db_session, reading):
        stranger = await _stranger(db_session)
        theirs = ReadingList(user_id=stranger.id, name="Prefetchers")
        db_session.add(theirs)
        await db_session.flush()
        db_session.add(
            ReadingListItem(reading_list_id=theirs.id, document_id=reading.first.id)
        )
        await db_session.commit()
        their_job = await _agent_job(db_session, stranger, status="running")

        mine = await _research("get_reading_lists", {}, db_session, reading.job)
        seen = await _research(
            "get_reading_lists", {"list_name": "Prefetchers"}, db_session, their_job
        )

        assert str(theirs.id) not in {e["id"] for e in mine["data"]["reading_lists"]}
        assert [e["id"] for e in seen["data"]["reading_lists"]] == [str(theirs.id)]
        assert len(seen["data"]["reading_lists"][0]["items"]) == 1


# --- retract_finding -------------------------------------------------------


REASON = "The prefetcher reported pfIssued = 0, so the arm never engaged."
OWN = {"type": "mechanism_evaluation", "title": "Stride issues 63k prefetches"}


@pytest.fixture
async def earlier(db_session, test_user):
    """An earlier run with two findings, and the run that is doing the retracting."""
    old = await _agent_job(
        db_session,
        test_user,
        results={
            "findings": [
                {"type": "mechanism_evaluation", "title": "ISB is 0.78x"},
                {"type": "headroom", "title": "L1D headroom is 73%"},
            ]
        },
    )
    current = await _agent_job(db_session, test_user, name="Now", status="running")
    return SimpleNamespace(old=old, current=current, ref=f"{old.id}#0")


async def _recalled(db, user_id, exclude):
    result = await agent_prior_findings.recall(
        db=db, user_id=user_id, exclude_job_id=exclude
    )
    return [f["title"] for f in result["findings"]]


class TestRetractFinding:
    async def test_a_retracted_finding_is_no_longer_recalled(
        self, db_session, test_user, earlier
    ):
        before = await _recalled(db_session, test_user.id, earlier.current.id)
        assert "ISB is 0.78x" in before

        result = await _retract(
            {
                "ref": earlier.ref,
                "reason": REASON,
                "contradicted_by": ["mechanism_evaluation"],
            },
            db_session,
            earlier.current,
            {"findings": [OWN]},
        )

        assert result["success"] is True
        assert result["data"]["retracted"] == earlier.ref
        after = await _recalled(db_session, test_user.id, earlier.current.id)
        assert after == ["L1D headroom is 73%"]
        _plain(result)

    async def test_the_retraction_is_a_row_owned_by_the_user_with_its_reason(
        self, db_session, test_user, earlier
    ):
        result = await _retract(
            {
                "ref": earlier.ref,
                "reason": REASON,
                "contradicted_by": ["Stride issues 63k prefetches"],
            },
            db_session,
            earlier.current,
            {"findings": [OWN]},
        )

        rows = await _all(db_session, AgentRetraction)
        assert len(rows) == 1
        row = rows[0]
        assert str(row.id) == result["data"]["retraction_id"]
        assert row.user_id == test_user.id
        assert row.subject_kind == RetractionKind.FINDING
        assert row.subject_ref == earlier.ref
        assert row.reason == REASON
        assert "Stride issues 63k prefetches" in row.source
        # The finding itself is kept: withdrawn, not deleted.
        await db_session.refresh(earlier.old)
        assert len(earlier.old.results["findings"]) == 2

    async def test_the_retraction_records_the_run_that_made_it(
        self, db_session, earlier
    ):
        await _retract(
            {
                "ref": earlier.ref,
                "reason": REASON,
                "contradicted_by": ["mechanism_evaluation"],
            },
            db_session,
            earlier.current,
            {"findings": [OWN]},
        )

        row = (await _all(db_session, AgentRetraction))[0]
        assert row.source_job_id == earlier.current.id

    @pytest.mark.parametrize(
        "cited",
        [
            '["mechanism_evaluation"]',
            "mechanism_evaluation, Stride issues 63k prefetches",
        ],
    )
    async def test_evidence_sent_as_text_is_still_read(
        self, db_session, earlier, cited
    ):
        result = await _retract(
            {"ref": earlier.ref, "reason": REASON, "contradicted_by": cited},
            db_session,
            earlier.current,
            {"findings": [OWN]},
        )

        assert result.get("success") is True, result
        assert "mechanism_evaluation" in result["data"]["contradicted_by"]

    @pytest.mark.parametrize(
        "change",
        [
            {"ref": ""},
            {"ref": "just-a-job-id"},
            {"reason": ""},
            {"reason": "wrong"},
            {"contradicted_by": []},
            {"contradicted_by": None},
            {"contradicted_by": ["a finding this run never produced"]},
        ],
    )
    async def test_an_incomplete_retraction_is_refused_and_writes_nothing(
        self, db_session, test_user, earlier, change
    ):
        params = {
            "ref": earlier.ref,
            "reason": REASON,
            "contradicted_by": ["mechanism_evaluation"],
        }
        params.update(change)

        await _refusal(
            _retract(params, db_session, earlier.current, {"findings": [OWN]})
        )

        assert await _all(db_session, AgentRetraction) == []
        assert "ISB is 0.78x" in await _recalled(
            db_session, test_user.id, earlier.current.id
        )

    @pytest.mark.parametrize(
        "state",
        [
            {},
            {"findings": []},
            {"findings": [dict(OWN, recalled=True)]},
        ],
    )
    async def test_a_run_with_no_evidence_of_its_own_cannot_retract(
        self, db_session, earlier, state
    ):
        await _refusal(
            _retract(
                {
                    "ref": earlier.ref,
                    "reason": REASON,
                    "contradicted_by": ["mechanism_evaluation"],
                },
                db_session,
                earlier.current,
                state,
            )
        )

        assert await _all(db_session, AgentRetraction) == []

    @pytest.mark.parametrize("ref", ["{missing}#0", "{old}#7", "{old}#first"])
    async def test_a_finding_that_does_not_exist_is_refused(
        self, db_session, earlier, ref
    ):
        ref = ref.format(missing=uuid4(), old=earlier.old.id)

        await _refusal(
            _retract(
                {
                    "ref": ref,
                    "reason": REASON,
                    "contradicted_by": ["mechanism_evaluation"],
                },
                db_session,
                earlier.current,
                {"findings": [OWN]},
            )
        )

        assert await _all(db_session, AgentRetraction) == []

    async def test_another_users_finding_cannot_be_retracted(self, db_session, earlier):
        stranger = await _stranger(db_session)
        theirs = await _agent_job(
            db_session,
            stranger,
            results={"findings": [{"type": "headroom", "title": "Theirs"}]},
        )

        await _refusal(
            _retract(
                {
                    "ref": f"{theirs.id}#0",
                    "reason": REASON,
                    "contradicted_by": ["mechanism_evaluation"],
                },
                db_session,
                earlier.current,
                {"findings": [OWN]},
            )
        )

    async def test_a_retraction_does_not_reach_into_another_users_recall(
        self, db_session, earlier
    ):
        stranger = await _stranger(db_session)
        theirs = await _agent_job(
            db_session,
            stranger,
            results={"findings": [{"type": "headroom", "title": "Theirs"}]},
        )

        try:
            await _retract(
                {
                    "ref": f"{theirs.id}#0",
                    "reason": REASON,
                    "contradicted_by": ["mechanism_evaluation"],
                },
                db_session,
                earlier.current,
                {"findings": [OWN]},
            )
        except ValueError:
            pass

        assert await _recalled(db_session, stranger.id, None) == ["Theirs"]


# --- template tools --------------------------------------------------------


class TestStartTemplateFill:
    async def test_known_documents_lead_to_the_upload_step(self, db_session, test_user):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Findings")

        result = await _chat(
            "start_template_fill",
            {"source_document_ids": [str(doc.id), "not-a-uuid", str(uuid4())]},
            db_session,
            test_user.id,
        )

        assert result["action"] == "template_upload_required"
        assert result["source_documents"] == [{"id": str(doc.id), "title": "Findings"}]
        assert result["source_document_ids"] == [str(doc.id)]
        _plain(result)

    async def test_it_starts_no_job_before_a_template_exists(
        self, db_session, test_user
    ):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Findings")

        await _chat(
            "start_template_fill",
            {"source_document_ids": [str(doc.id)]},
            db_session,
            test_user.id,
        )

        assert await _all(db_session, TemplateJob) == []

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"source_document_ids": []},
            {"source_document_ids": ["not-a-uuid"]},
            {"source_document_ids": [None]},
        ],
    )
    async def test_no_usable_document_is_refused(self, db_session, test_user, params):
        await _refusal(_chat("start_template_fill", params, db_session, test_user.id))

    async def test_an_id_that_is_not_text_is_refused(self, db_session, test_user):
        await _refusal(
            _chat(
                "start_template_fill",
                {"source_document_ids": [7]},
                db_session,
                test_user.id,
            )
        )

    async def test_documents_that_do_not_exist_are_refused(self, db_session, test_user):
        await _refusal(
            _chat(
                "start_template_fill",
                {"source_document_ids": [str(uuid4())]},
                db_session,
                test_user.id,
            )
        )


@pytest.fixture
async def template_jobs(db_session, test_user):
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    pending = await _template_job(
        db_session, test_user, template_filename="a.docx", created_at=now
    )
    filling = await _template_job(
        db_session,
        test_user,
        template_filename="b.docx",
        status="filling",
        progress=40,
        current_section="Results",
        sections=[{"title": "Intro", "level": 1}, {"title": "Results", "level": 1}],
        source_document_ids=["d1", "d2", "d3"],
        created_at=now + timedelta(hours=1),
    )
    completed = await _template_job(
        db_session,
        test_user,
        template_filename="c.docx",
        status="completed",
        progress=100,
        filled_file_path="filled/c.docx",
        filled_filename="c-filled.docx",
        completed_at=now + timedelta(hours=3),
        created_at=now + timedelta(hours=2),
    )
    failed = await _template_job(
        db_session,
        test_user,
        template_filename="d.docx",
        status="failed",
        error_message="The template has no headings.",
        created_at=now + timedelta(hours=3),
    )
    return SimpleNamespace(
        pending=pending, filling=filling, completed=completed, failed=failed
    )


class TestListTemplateJobs:
    async def test_jobs_are_listed_newest_first(
        self, db_session, test_user, template_jobs
    ):
        result = await _chat("list_template_jobs", {}, db_session, test_user.id)

        assert result["count"] == 4
        assert [j["template_filename"] for j in result["jobs"]] == [
            "d.docx",
            "c.docx",
            "b.docx",
            "a.docx",
        ]
        filling = result["jobs"][2]
        assert filling["id"] == str(template_jobs.filling.id)
        assert filling["progress"] == 40
        assert filling["current_section"] == "Results"
        assert filling["section_count"] == 2
        assert filling["source_doc_count"] == 3
        assert filling["has_download"] is False
        assert result["jobs"][1]["has_download"] is True
        assert result["jobs"][0]["error"] == "The template has no headings."
        _plain(result)

    @pytest.mark.parametrize(
        "status,expected",
        [
            ("all", {"a.docx", "b.docx", "c.docx", "d.docx"}),
            ("completed", {"c.docx"}),
            ("failed", {"d.docx"}),
            ("pending", {"a.docx"}),
            ("processing", {"a.docx", "b.docx"}),
        ],
    )
    async def test_the_status_filter_selects_jobs(
        self, db_session, test_user, template_jobs, status, expected
    ):
        result = await _chat(
            "list_template_jobs", {"status_filter": status}, db_session, test_user.id
        )

        assert {j["template_filename"] for j in result["jobs"]} == expected

    async def test_the_limit_bounds_the_list(
        self, db_session, test_user, template_jobs
    ):
        result = await _chat(
            "list_template_jobs", {"limit": 2}, db_session, test_user.id
        )

        assert [j["template_filename"] for j in result["jobs"]] == ["d.docx", "c.docx"]

    async def test_a_limit_sent_as_null_means_the_default(
        self, db_session, test_user, template_jobs
    ):
        result = await _chat(
            "list_template_jobs", {"limit": None}, db_session, test_user.id
        )

        assert result["count"] == 4

    async def test_another_users_jobs_are_not_listed(
        self, db_session, test_user, template_jobs
    ):
        stranger = await _stranger(db_session)

        result = await _chat("list_template_jobs", {}, db_session, stranger.id)

        assert result == {"count": 0, "jobs": []}


class TestGetTemplateJobStatus:
    async def test_a_running_job_reports_where_it_is(
        self, db_session, test_user, template_jobs
    ):
        job = template_jobs.filling

        result = await _chat(
            "get_template_job_status", {"job_id": str(job.id)}, db_session, test_user.id
        )

        assert result["id"] == str(job.id)
        assert result["status"] == "filling"
        assert result["progress"] == 40
        assert result["current_section"] == "Results"
        assert result["sections"] == [
            {"title": "Intro", "level": 1},
            {"title": "Results", "level": 1},
        ]
        assert result["total_sections"] == 2
        assert result["source_document_count"] == 3
        assert "download_url" not in result
        _plain(result)

    async def test_a_completed_job_offers_its_download(
        self, db_session, test_user, template_jobs
    ):
        job = template_jobs.completed

        result = await _chat(
            "get_template_job_status", {"job_id": str(job.id)}, db_session, test_user.id
        )

        assert result["download_available"] is True
        assert result["filled_filename"] == "c-filled.docx"
        assert result["download_url"] == f"/api/v1/templates/{job.id}/download"
        assert result["completed_at"]

    async def test_a_failed_job_says_why(self, db_session, test_user, template_jobs):
        result = await _chat(
            "get_template_job_status",
            {"job_id": str(template_jobs.failed.id)},
            db_session,
            test_user.id,
        )

        assert result["error_message"] == "The template has no headings."
        assert "download_url" not in result

    async def test_an_id_that_is_not_text_is_refused(self, db_session, test_user):
        message = await _refusal(
            _chat("get_template_job_status", {"job_id": 7}, db_session, test_user.id)
        )

        assert "Invalid job ID" in message

    @pytest.mark.parametrize("job_id", [None, "", "not-a-uuid"])
    async def test_a_malformed_id_is_refused(self, db_session, test_user, job_id):
        params = {} if job_id is None else {"job_id": job_id}

        message = await _refusal(
            _chat("get_template_job_status", params, db_session, test_user.id)
        )

        assert "Invalid job ID" in message

    async def test_an_unknown_job_is_not_found(self, db_session, test_user):
        message = await _refusal(
            _chat(
                "get_template_job_status",
                {"job_id": str(uuid4())},
                db_session,
                test_user.id,
            )
        )

        assert "not found" in message

    async def test_another_users_job_is_not_found(
        self, db_session, test_user, template_jobs
    ):
        stranger = await _stranger(db_session)

        message = await _refusal(
            _chat(
                "get_template_job_status",
                {"job_id": str(template_jobs.completed.id)},
                db_session,
                stranger.id,
            )
        )

        assert "not found" in message
        assert "c-filled.docx" not in message


# --- summarize_documents_in_source -----------------------------------------


@pytest.fixture
def summarize_task(monkeypatch):
    from app.tasks.summarization_tasks import summarize_document

    return _record(monkeypatch, summarize_document)


@pytest.fixture
async def arxiv_source(db_session, test_user):
    """An arXiv import the test user asked for, holding three papers."""
    source = await _source(
        db_session,
        config={
            "topic": "Prefetching",
            "requested_by": str(test_user.id),
            "requested_by_user_id": str(test_user.id),
        },
    )
    papers = [
        await _doc(
            db_session,
            source,
            f"Paper {n}",
            summary="Already summarised." if n == 1 else None,
            created_at=datetime(2026, 1, n, tzinfo=timezone.utc),
        )
        for n in (1, 2, 3)
    ]
    return SimpleNamespace(source=source, papers=papers)


class TestSummarizeDocumentsInSource:
    async def test_documents_without_a_summary_are_queued_for_the_caller(
        self, db_session, test_user, arxiv_source, summarize_task
    ):
        result = await _chat(
            "summarize_documents_in_source",
            {"source_id": str(arxiv_source.source.id)},
            db_session,
            test_user.id,
        )

        assert result["queued"] == 2
        assert result["considered"] == 3
        assert result["source_id"] == str(arxiv_source.source.id)
        assert {c["document_id"] for c in summarize_task.calls} == {
            str(arxiv_source.papers[1].id),
            str(arxiv_source.papers[2].id),
        }
        assert all(c["user_id"] == str(test_user.id) for c in summarize_task.calls)
        assert all(c["force"] is False for c in summarize_task.calls)
        _plain(result)

    async def test_force_queues_every_document_and_tells_the_task(
        self, db_session, test_user, arxiv_source, summarize_task
    ):
        result = await _chat(
            "summarize_documents_in_source",
            {"source_id": str(arxiv_source.source.id), "force": True},
            db_session,
            test_user.id,
        )

        assert result["queued"] == 3
        assert all(c["force"] is True for c in summarize_task.calls)

    async def test_only_missing_can_be_turned_off(
        self, db_session, test_user, arxiv_source, summarize_task
    ):
        result = await _chat(
            "summarize_documents_in_source",
            {"source_id": str(arxiv_source.source.id), "only_missing": False},
            db_session,
            test_user.id,
        )

        assert result["queued"] == 3

    async def test_the_limit_takes_the_newest_documents(
        self, db_session, test_user, arxiv_source, summarize_task
    ):
        result = await _chat(
            "summarize_documents_in_source",
            {"source_id": str(arxiv_source.source.id), "limit": 1},
            db_session,
            test_user.id,
        )

        assert result["considered"] == 1
        assert [c["document_id"] for c in summarize_task.calls] == [
            str(arxiv_source.papers[2].id)
        ]

    async def test_documents_of_another_source_are_left_alone(
        self, db_session, test_user, arxiv_source, summarize_task
    ):
        other = await _source(db_session, "Other", "web")
        await _doc(db_session, other, "Elsewhere")

        await _chat(
            "summarize_documents_in_source",
            {"source_id": str(other.id)},
            db_session,
            test_user.id,
        )

        assert len(summarize_task.calls) == 1

    @pytest.mark.parametrize("source_id", [None, "", "not-a-uuid", str(uuid4())])
    async def test_a_missing_or_unknown_source_is_refused(
        self, db_session, test_user, summarize_task, source_id
    ):
        params = {} if source_id is None else {"source_id": source_id}

        await _refusal(
            _chat("summarize_documents_in_source", params, db_session, test_user.id)
        )

        assert summarize_task.calls == []


# --- enrich_arxiv_metadata_for_source --------------------------------------


@pytest.fixture
def enrich_task(monkeypatch):
    from app.tasks.paper_enrichment_tasks import enrich_arxiv_source

    return _record(monkeypatch, enrich_arxiv_source)


class TestEnrichArxivMetadataForSource:
    async def test_enrichment_is_queued_for_the_source(
        self, db_session, test_user, arxiv_source, enrich_task
    ):
        result = await _chat(
            "enrich_arxiv_metadata_for_source",
            {"source_id": str(arxiv_source.source.id)},
            db_session,
            test_user.id,
        )

        assert result["queued"] is True
        assert result["task_id"] == "task-1"
        assert enrich_task.calls == [
            {"source_id": str(arxiv_source.source.id), "force": False, "limit": 500}
        ]
        _plain(result)

    async def test_force_and_limit_reach_the_task(
        self, db_session, test_user, arxiv_source, enrich_task
    ):
        result = await _chat(
            "enrich_arxiv_metadata_for_source",
            {"source_id": str(arxiv_source.source.id), "force": True, "limit": 25},
            db_session,
            test_user.id,
        )

        assert enrich_task.calls[0]["force"] is True
        assert enrich_task.calls[0]["limit"] == 25
        assert result["limit"] == 25

    async def test_the_limit_is_capped(
        self, db_session, test_user, arxiv_source, enrich_task
    ):
        await _chat(
            "enrich_arxiv_metadata_for_source",
            {"source_id": str(arxiv_source.source.id), "limit": 999999},
            db_session,
            test_user.id,
        )

        assert enrich_task.calls[0]["limit"] == 5000

    async def test_another_users_source_is_refused(
        self, db_session, arxiv_source, enrich_task
    ):
        stranger = await _stranger(db_session)

        message = await _refusal(
            _chat(
                "enrich_arxiv_metadata_for_source",
                {"source_id": str(arxiv_source.source.id)},
                db_session,
                stranger.id,
            )
        )

        assert "authorized" in message
        assert enrich_task.calls == []

    async def test_an_admin_may_enrich_any_source(
        self, db_session, admin_user, arxiv_source, enrich_task
    ):
        result = await _chat(
            "enrich_arxiv_metadata_for_source",
            {"source_id": str(arxiv_source.source.id)},
            db_session,
            admin_user.id,
        )

        assert result["queued"] is True

    async def test_a_source_that_is_not_arxiv_is_refused(
        self, db_session, test_user, enrich_task
    ):
        web = await _source(db_session, "Docs", "web")

        await _refusal(
            _chat(
                "enrich_arxiv_metadata_for_source",
                {"source_id": str(web.id)},
                db_session,
                test_user.id,
            )
        )

        assert enrich_task.calls == []

    @pytest.mark.parametrize("source_id", [None, "", "not-a-uuid", str(uuid4())])
    async def test_a_missing_or_unknown_source_is_refused(
        self, db_session, test_user, enrich_task, source_id
    ):
        params = {} if source_id is None else {"source_id": source_id}

        await _refusal(
            _chat("enrich_arxiv_metadata_for_source", params, db_session, test_user.id)
        )

        assert enrich_task.calls == []


# --- generate_slides_for_source --------------------------------------------


@pytest.fixture
def slides_task(monkeypatch):
    from app.tasks.presentation_tasks import generate_presentation_task

    return _record(monkeypatch, generate_presentation_task)


class TestGenerateSlidesForSource:
    async def test_a_presentation_job_is_written_and_queued(
        self, db_session, test_user, arxiv_source, slides_task
    ):
        result = await _chat(
            "generate_slides_for_source",
            {"source_id": str(arxiv_source.source.id)},
            db_session,
            test_user.id,
        )

        jobs = await _all(db_session, PresentationJob)
        assert len(jobs) == 1
        job = jobs[0]
        assert result["presentation_job_id"] == str(job.id)
        assert job.user_id == test_user.id
        assert job.title == "Slides: arXiv import"
        assert job.topic == "Prefetching"
        assert job.slide_count == 10
        assert job.style == "professional"
        assert job.include_diagrams == 1
        assert job.status == "pending"
        assert set(job.source_document_ids) == {str(p.id) for p in arxiv_source.papers}
        assert slides_task.calls == [
            {"job_id": str(job.id), "user_id": str(test_user.id)}
        ]
        _plain(result)

    async def test_every_option_reaches_the_job(
        self, db_session, test_user, arxiv_source, slides_task
    ):
        await _chat(
            "generate_slides_for_source",
            {
                "source_id": str(arxiv_source.source.id),
                "title": "Group meeting",
                "topic": "Stride vs ISB",
                "slide_count": 6,
                "style": "minimal",
                "include_diagrams": False,
            },
            db_session,
            test_user.id,
        )

        job = (await _all(db_session, PresentationJob))[0]
        assert job.title == "Group meeting"
        assert job.topic == "Stride vs ISB"
        assert job.slide_count == 6
        assert job.style == "minimal"
        assert job.include_diagrams == 0

    @pytest.mark.parametrize("asked,stored", [(1, 3), (400, 40)])
    async def test_the_slide_count_is_kept_in_range(
        self, db_session, test_user, arxiv_source, slides_task, asked, stored
    ):
        await _chat(
            "generate_slides_for_source",
            {"source_id": str(arxiv_source.source.id), "slide_count": asked},
            db_session,
            test_user.id,
        )

        assert (await _all(db_session, PresentationJob))[0].slide_count == stored

    async def test_the_literature_review_is_preferred_when_there_is_one(
        self, db_session, test_user, arxiv_source, slides_task
    ):
        review = await _doc(
            db_session,
            arxiv_source.source,
            "Review",
            source_identifier=f"literature_review:{arxiv_source.source.id}",
        )

        preferred = await _chat(
            "generate_slides_for_source",
            {"source_id": str(arxiv_source.source.id)},
            db_session,
            test_user.id,
        )
        everything = await _chat(
            "generate_slides_for_source",
            {
                "source_id": str(arxiv_source.source.id),
                "prefer_review_document": False,
            },
            db_session,
            test_user.id,
        )

        assert preferred["source_document_ids"] == [str(review.id)]
        assert len(everything["source_document_ids"]) == 4

    async def test_a_source_with_no_documents_is_refused_and_queues_nothing(
        self, db_session, test_user, slides_task
    ):
        empty = await _source(db_session, "Empty import")

        await _refusal(
            _chat(
                "generate_slides_for_source",
                {"source_id": str(empty.id)},
                db_session,
                test_user.id,
            )
        )

        assert await _all(db_session, PresentationJob) == []
        assert slides_task.calls == []

    async def test_a_source_that_is_not_arxiv_is_refused(
        self, db_session, test_user, slides_task
    ):
        web = await _source(db_session, "Docs", "web")
        await _doc(db_session, web, "Page")

        await _refusal(
            _chat(
                "generate_slides_for_source",
                {"source_id": str(web.id)},
                db_session,
                test_user.id,
            )
        )

        assert slides_task.calls == []

    @pytest.mark.parametrize("source_id", [None, "", "not-a-uuid", str(uuid4())])
    async def test_a_missing_or_unknown_source_is_refused(
        self, db_session, test_user, slides_task, source_id
    ):
        params = {} if source_id is None else {"source_id": source_id}

        await _refusal(
            _chat("generate_slides_for_source", params, db_session, test_user.id)
        )

        assert slides_task.calls == []

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Two implementations disagree: "
            "_tool_enrich_arxiv_metadata_for_source refuses a source "
            "another user requested, while "
            "_tool_generate_slides_for_source (and "
            "_tool_summarize_documents_in_source) perform no ownership "
            "check on the same source."
        ),
    )
    async def test_another_users_source_is_refused(
        self, db_session, arxiv_source, slides_task
    ):
        """The same rule `enrich_arxiv_metadata_for_source` applies to this source."""
        stranger = await _stranger(db_session)

        await _refusal(
            _chat(
                "generate_slides_for_source",
                {"source_id": str(arxiv_source.source.id)},
                db_session,
                stranger.id,
            )
        )

        assert slides_task.calls == []


# --- find_related_papers ---------------------------------------------------


REFERENCE = {"id": "2401.00001v1", "title": "Stride Prefetching Revisited"}
NEIGHBOURS = [
    {"id": "2402.00002v1", "title": "Irregular Stream Buffers"},
    {"id": "2403.00003v1", "title": "Temporal Prefetching"},
]


def _related_answers(query):
    if query.startswith("id:"):
        return [REFERENCE]
    return [REFERENCE, *NEIGHBOURS]


@pytest.fixture
async def research_job(db_session, test_user):
    return await _agent_job(db_session, test_user, status="running")


class TestFindRelatedPapers:
    async def test_a_document_is_searched_for_by_its_title(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch, lambda query: NEIGHBOURS)
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Stride Prefetching Revisited")

        result = await _research(
            "find_related_papers",
            {"document_id": str(doc.id)},
            db_session,
            research_job,
        )

        assert result["success"] is True
        assert arxiv.queries == ["Stride Prefetching Revisited"]
        assert [p["title"] for p in result["data"]] == [
            "Irregular Stream Buffers",
            "Temporal Prefetching",
        ]
        assert result["findings"] == [
            {
                "type": "related_paper_set",
                "title": "Irregular Stream Buffers",
                "arxiv_id": "2402.00002v1",
            },
            {
                "type": "related_paper_set",
                "title": "Temporal Prefetching",
                "arxiv_id": "2403.00003v1",
            },
        ]
        _plain(result)

    async def test_an_arxiv_id_is_resolved_to_its_title_first(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch, _related_answers)

        result = await _research(
            "find_related_papers",
            {"arxiv_id": "2401.00001"},
            db_session,
            research_job,
        )

        assert arxiv.queries == ["id:2401.00001", "Stride Prefetching Revisited"]
        assert result["success"] is True

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_find_related_papers searches arXiv for the reference "
            "paper's own title and returns every hit, so the paper is "
            "recorded as a related_paper_set finding for itself."
        ),
    )
    async def test_the_reference_paper_is_not_related_to_itself(
        self, db_session, research_job, monkeypatch
    ):
        FakeArxiv(monkeypatch, _related_answers)

        result = await _research(
            "find_related_papers",
            {"arxiv_id": "2401.00001"},
            db_session,
            research_job,
        )

        assert REFERENCE["id"] not in [f["arxiv_id"] for f in result["findings"]]

    async def test_the_limit_is_what_arxiv_is_asked_for(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch, lambda query: NEIGHBOURS)
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Stride")

        await _research(
            "find_related_papers",
            {"document_id": str(doc.id), "limit": 3},
            db_session,
            research_job,
        )

        assert arxiv.requests[0]["max_results"] == "3"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "search_external=false returns {'error': 'No query could be "
            "built'} although a query was built. The tool never searches "
            "the knowledge base, so the declared option can only produce "
            "an error."
        ),
    )
    async def test_without_external_search_arxiv_is_not_called_and_nothing_fails(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch, lambda query: NEIGHBOURS)
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Stride")

        result = await _research(
            "find_related_papers",
            {"document_id": str(doc.id), "search_external": False},
            db_session,
            research_job,
        )

        assert arxiv.requests == []
        assert "error" not in result, result

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The declared `relation_type` parameter is never read by "
            "_find_related_papers: every value runs the same arXiv title "
            "search."
        ),
    )
    async def test_the_relation_type_changes_what_is_looked_for(
        self, db_session, research_job, monkeypatch
    ):
        source = await _source(db_session)
        doc = await _doc(db_session, source, "Stride", author="Ada Lovelace")
        seen = {}
        for relation in ("semantic", "shared_authors"):
            arxiv = FakeArxiv(monkeypatch, lambda query: NEIGHBOURS)
            await _research(
                "find_related_papers",
                {"document_id": str(doc.id), "relation_type": relation},
                db_session,
                research_job,
            )
            seen[relation] = arxiv.queries

        assert seen["semantic"] != seen["shared_authors"]

    async def test_no_paper_named_is_refused(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch)

        await _refusal(_research("find_related_papers", {}, db_session, research_job))

        assert arxiv.requests == []

    @pytest.mark.parametrize("document_id", ["not-a-uuid", str(uuid4())])
    async def test_a_malformed_or_unknown_document_is_refused(
        self, db_session, research_job, monkeypatch, document_id
    ):
        arxiv = FakeArxiv(monkeypatch)

        await _refusal(
            _research(
                "find_related_papers",
                {"document_id": document_id},
                db_session,
                research_job,
            )
        )

        assert arxiv.requests == []

    async def test_an_arxiv_id_nothing_answers_to_is_refused(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch)

        await _refusal(
            _research(
                "find_related_papers",
                {"arxiv_id": "0000.00000"},
                db_session,
                research_job,
            )
        )

        assert len(arxiv.requests) == 1


# --- monitor_arxiv_topic ---------------------------------------------------


def _days_ago(days):
    moment = datetime.now(timezone.utc) - timedelta(days=days)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class TestMonitorArxivTopic:
    async def test_recent_papers_on_the_topic_come_back_as_findings(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(
            monkeypatch,
            lambda query: [
                {
                    "id": "2609.00001v1",
                    "title": "A new prefetcher",
                    "published": "2026-09-28T00:00:00Z",
                }
            ],
        )

        result = await _research(
            "monitor_arxiv_topic",
            {"topic": "hardware prefetching"},
            db_session,
            research_job,
        )

        assert result["success"] is True
        request = arxiv.requests[0]
        assert request["search_query"] == "all:hardware prefetching"
        assert request["sortBy"] == "submittedDate"
        assert request["sortOrder"] == "descending"
        assert request["max_results"] == "20"
        assert result["findings"] == [
            {
                "type": "new_paper",
                "title": "A new prefetcher",
                "arxiv_id": "2609.00001v1",
                "published": "2026-09-28T00:00:00Z",
            }
        ]
        assert result["data"][0]["title"] == "A new prefetcher"
        _plain(result)

    async def test_a_query_expression_replaces_the_topic_search(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch)

        await _research(
            "monitor_arxiv_topic",
            {"topic": "prefetching", "query": "ti:prefetch AND cat:cs.AR"},
            db_session,
            research_job,
        )

        assert arxiv.queries == ["ti:prefetch AND cat:cs.AR"]

    async def test_max_results_is_what_arxiv_is_asked_for(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch)

        await _research(
            "monitor_arxiv_topic",
            {"topic": "prefetching", "max_results": 5},
            db_session,
            research_job,
        )

        assert arxiv.requests[0]["max_results"] == "5"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The spec requires `topic`; with none, _monitor_arxiv_topic "
            "searches arXiv for the literal query 'all:None' and reports "
            "success."
        ),
    )
    async def test_a_topic_is_required(self, db_session, research_job, monkeypatch):
        arxiv = FakeArxiv(monkeypatch)

        await _refusal(_research("monitor_arxiv_topic", {}, db_session, research_job))

        assert arxiv.requests == []

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The declared `categories` parameter is never read by "
            "_monitor_arxiv_topic."
        ),
    )
    async def test_categories_narrow_the_search(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(monkeypatch)

        await _research(
            "monitor_arxiv_topic",
            {"topic": "prefetching", "categories": ["cs.AR"]},
            db_session,
            research_job,
        )

        assert "cs.AR" in arxiv.queries[0]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The declared `since_days` parameter is never read by "
            "_monitor_arxiv_topic: papers of any age are returned as "
            "new_paper findings."
        ),
    )
    async def test_since_days_leaves_out_older_papers(
        self, db_session, research_job, monkeypatch
    ):
        arxiv = FakeArxiv(
            monkeypatch,
            lambda query: [
                {"id": "new", "title": "This week", "published": _days_ago(2)},
                {"id": "old", "title": "Last quarter", "published": _days_ago(90)},
            ],
        )

        result = await _research(
            "monitor_arxiv_topic",
            {"topic": "prefetching", "since_days": 7},
            db_session,
            research_job,
        )

        bounded_by_query = "submittedDate" in arxiv.queries[0]
        titles = [f["title"] for f in result["findings"]]
        assert bounded_by_query or titles == ["This week"]


# --- propose / create workflow from description ----------------------------


def _draft(**overrides):
    """A model reply: a three-node workflow that searches the knowledge base."""
    draft = {
        "name": "Weekly digest",
        "description": "Search and summarise.",
        "is_active": True,
        "trigger_config": {"type": "manual"},
        "nodes": [
            {"node_id": "start", "node_type": "start"},
            {
                "node_id": "search",
                "node_type": "tool",
                "builtin_tool": "search_documents",
                "config": {"input_mapping": {"query": "prefetching"}},
            },
            {"node_id": "end", "node_type": "end"},
        ],
        "edges": [
            {"source_node_id": "start", "target_node_id": "search"},
            {"source_node_id": "search", "target_node_id": "end"},
        ],
        "custom_tools": [],
        "workflow_tool": None,
    }
    draft.update(overrides)
    return json.dumps(draft)


def _workflow_llm(monkeypatch, *replies, error=None):
    llm = FakeLLM(*(replies or [_draft()]), error=error)
    monkeypatch.setattr(synthesis_module, "LLMService", lambda: llm)
    return llm


WORKFLOW_TOOLS = (
    "propose_workflow_from_description",
    "create_workflow_from_description",
)

TRANSFORM_TOOL = {
    "name": "fetch_prices",
    "description": "Reshape a price list.",
    "tool_type": "transform",
    "parameters_schema": {"type": "object", "properties": {}},
    "config": {"template": "{{ input }}"},
}


class TestBothWorkflowTools:
    @pytest.mark.parametrize("tool", WORKFLOW_TOOLS)
    @pytest.mark.parametrize("params", [{}, {"description": ""}, {"description": " "}])
    async def test_a_description_is_required(
        self, db_session, test_user, monkeypatch, tool, params
    ):
        llm = _workflow_llm(monkeypatch)

        await _refusal(_chat(tool, params, db_session, test_user.id))

        assert llm.attempts == 0
        assert await _all(db_session, Workflow) == []

    @pytest.mark.parametrize("tool", WORKFLOW_TOOLS)
    async def test_the_description_name_and_trigger_reach_the_model(
        self, db_session, test_user, monkeypatch, tool
    ):
        llm = _workflow_llm(monkeypatch)

        await _chat(
            tool,
            {
                "description": "Every Monday, search for prefetching papers.",
                "name": "Monday digest",
                "trigger_config": {"type": "schedule", "schedule": "0 9 * * 1"},
            },
            db_session,
            test_user.id,
        )

        assert "Every Monday, search for prefetching papers." in llm.text
        assert "Monday digest" in llm.text
        assert "0 9 * * 1" in llm.text

    @pytest.mark.parametrize("tool", WORKFLOW_TOOLS)
    async def test_a_reply_that_is_never_json_is_an_error_and_saves_nothing(
        self, db_session, test_user, monkeypatch, tool
    ):
        llm = _workflow_llm(monkeypatch, "I would rather describe it in prose.")

        await _refusal(
            _chat(tool, {"description": "Do something."}, db_session, test_user.id)
        )

        assert llm.attempts == 2
        assert await _all(db_session, Workflow) == []

    @pytest.mark.parametrize("tool", WORKFLOW_TOOLS)
    async def test_an_unreachable_model_is_an_error_and_saves_nothing(
        self, db_session, test_user, monkeypatch, tool
    ):
        _workflow_llm(monkeypatch, error=RuntimeError("provider is down"))

        message = await _refusal(
            _chat(tool, {"description": "Do something."}, db_session, test_user.id)
        )

        assert "provider is down" in message
        assert await _all(db_session, Workflow) == []

    @pytest.mark.parametrize("tool", WORKFLOW_TOOLS)
    async def test_a_draft_with_no_steps_is_asked_for_again(
        self, db_session, test_user, monkeypatch, tool
    ):
        empty = _draft(nodes=[], edges=[])
        llm = _workflow_llm(monkeypatch, empty, _draft())

        result = await _chat(tool, {"description": "Search."}, db_session, test_user.id)

        assert "error" not in result, result
        assert llm.attempts == 2
        assert any("attempt 2" in w for w in result["warnings"])


class TestProposeWorkflowFromDescription:
    async def test_a_draft_is_returned_and_nothing_is_saved(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch)

        result = await _chat(
            "propose_workflow_from_description",
            {"description": "Search for prefetching papers."},
            db_session,
            test_user.id,
        )

        workflow = result["workflow"]
        assert workflow["name"] == "Weekly digest"
        assert [n["node_id"] for n in workflow["nodes"]] == ["start", "search", "end"]
        assert workflow["nodes"][1]["builtin_tool"] == "search_documents"
        assert len(workflow["edges"]) == 2
        assert result["custom_tools"] == []
        assert result["workflow_tool"] is None
        assert await _all(db_session, Workflow) == []
        assert await _all(db_session, UserTool) == []
        _plain(result)

    async def test_a_name_is_used_when_the_model_offers_none(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch, _draft(name=None))

        result = await _chat(
            "propose_workflow_from_description",
            {"description": "Search.", "name": "Monday digest"},
            db_session,
            test_user.id,
        )

        assert result["workflow"]["name"] == "Monday digest"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "WorkflowSynthesisService._normalize_workflow takes is_active "
            "from the model's reply and uses the caller's value only when "
            "the reply has none. The prompt asks for the field and never "
            "states the caller's choice, so is_active=false is ignored."
        ),
    )
    async def test_asking_for_an_inactive_workflow_gives_an_inactive_draft(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch)

        result = await _chat(
            "propose_workflow_from_description",
            {"description": "Search.", "is_active": False},
            db_session,
            test_user.id,
        )

        assert result["workflow"]["is_active"] is False

    async def test_custom_tools_are_drafted_only_when_asked(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch, _draft(custom_tools=[TRANSFORM_TOOL]))

        without = await _chat(
            "propose_workflow_from_description",
            {"description": "Reshape prices."},
            db_session,
            test_user.id,
        )
        with_tools = await _chat(
            "propose_workflow_from_description",
            {"description": "Reshape prices.", "synthesize_custom_tools": True},
            db_session,
            test_user.id,
        )

        assert without["custom_tools"] == []
        assert [t["name"] for t in with_tools["custom_tools"]] == ["fetch_prices"]
        assert await _all(db_session, UserTool) == []

    async def test_the_preferred_tool_type_fills_in_a_missing_one(
        self, db_session, test_user, monkeypatch
    ):
        untyped = {k: v for k, v in TRANSFORM_TOOL.items() if k != "tool_type"}
        llm = _workflow_llm(monkeypatch, _draft(custom_tools=[untyped]))

        result = await _chat(
            "propose_workflow_from_description",
            {
                "description": "Reshape prices.",
                "synthesize_custom_tools": True,
                "preferred_tool_type": "python",
            },
            db_session,
            test_user.id,
        )

        assert result["custom_tools"][0]["tool_type"] == "python"
        assert "preferred custom tool type: python" in llm.text

    async def test_a_runner_tool_is_drafted_only_when_asked(
        self, db_session, test_user, monkeypatch
    ):
        runner = {"name": "run_digest", "tool_type": "workflow_runner"}
        _workflow_llm(monkeypatch, _draft(workflow_tool=runner))

        without = await _chat(
            "propose_workflow_from_description",
            {"description": "Search."},
            db_session,
            test_user.id,
        )
        with_runner = await _chat(
            "propose_workflow_from_description",
            {"description": "Search.", "expose_workflow_as_tool": True},
            db_session,
            test_user.id,
        )

        assert without["workflow_tool"] is None
        assert with_runner["workflow_tool"]["name"] == "run_digest"
        assert with_runner["workflow_tool"]["tool_type"] == "workflow_runner"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The two tools disagree: _normalize_workflow_tool prefers the "
            "model's name over workflow_tool_name, while "
            "_tool_create_workflow_from_description prefers "
            "workflow_tool_name. The proposal shows a different tool name "
            "than the one that would be saved."
        ),
    )
    async def test_the_runner_draft_carries_the_name_that_was_asked_for(
        self, db_session, test_user, monkeypatch
    ):
        """`create_workflow_from_description` saves under the asked-for name."""
        runner = {"name": "run_digest", "tool_type": "workflow_runner"}
        _workflow_llm(monkeypatch, _draft(workflow_tool=runner))

        result = await _chat(
            "propose_workflow_from_description",
            {
                "description": "Search.",
                "expose_workflow_as_tool": True,
                "workflow_tool_name": "digest_runner",
            },
            db_session,
            test_user.id,
        )

        assert result["workflow_tool"]["name"] == "digest_runner"


class TestCreateWorkflowFromDescription:
    async def test_the_workflow_is_saved_with_its_nodes_and_edges(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch)

        result = await _chat(
            "create_workflow_from_description",
            {"description": "Search for prefetching papers."},
            db_session,
            test_user.id,
        )

        assert "error" not in result, result
        workflows = await _all(db_session, Workflow)
        assert len(workflows) == 1
        workflow = workflows[0]
        assert result["workflow_id"] == str(workflow.id)
        assert result["workflow_name"] == "Weekly digest"
        assert workflow.user_id == test_user.id
        assert workflow.name == "Weekly digest"
        assert workflow.is_active is True
        assert workflow.trigger_config == {"type": "manual"}
        nodes = await _all(db_session, WorkflowNode)
        assert {n.workflow_id for n in nodes} == {workflow.id}
        by_id = {n.node_id: n for n in nodes}
        assert set(by_id) == {"start", "search", "end"}
        assert by_id["search"].node_type == "tool"
        assert by_id["search"].builtin_tool == "search_documents"
        assert by_id["search"].config["input_mapping"] == {"query": "prefetching"}
        edges = await _all(db_session, WorkflowEdge)
        assert {(e.source_node_id, e.target_node_id) for e in edges} == {
            ("start", "search"),
            ("search", "end"),
        }
        assert result["node_count"] == 3 and result["edge_count"] == 2
        assert result["created_custom_tools"] == []
        assert result["workflow_tool_id"] is None
        _plain(result)

    async def test_the_trigger_asked_for_is_stored_when_the_model_names_none(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch, _draft(trigger_config=None))
        trigger = {"type": "schedule", "schedule": "0 9 * * 1"}

        await _chat(
            "create_workflow_from_description",
            {"description": "Search.", "trigger_config": trigger},
            db_session,
            test_user.id,
        )

        assert (await _all(db_session, Workflow))[0].trigger_config == trigger

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "WorkflowSynthesisService._normalize_workflow lets the "
            "model's is_active override the caller's: a workflow asked "
            "for as inactive is saved active."
        ),
    )
    async def test_asking_for_an_inactive_workflow_saves_an_inactive_one(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch)

        await _chat(
            "create_workflow_from_description",
            {"description": "Search.", "is_active": False},
            db_session,
            test_user.id,
        )

        assert (await _all(db_session, Workflow))[0].is_active is False

    async def test_a_runner_tool_points_at_the_saved_workflow(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch)

        result = await _chat(
            "create_workflow_from_description",
            {
                "description": "Search.",
                "expose_workflow_as_tool": True,
                "workflow_tool_name": "digest_runner",
            },
            db_session,
            test_user.id,
        )

        workflow = (await _all(db_session, Workflow))[0]
        tools = await _all(db_session, UserTool)
        assert len(tools) == 1
        tool = tools[0]
        assert result["workflow_tool_id"] == str(tool.id)
        assert tool.user_id == test_user.id
        assert tool.name == "digest_runner"
        assert tool.tool_type == "workflow_runner"
        assert tool.config == {"workflow_id": str(workflow.id)}

    async def test_no_runner_tool_is_made_unless_asked(
        self, db_session, test_user, monkeypatch
    ):
        runner = {"name": "run_digest", "tool_type": "workflow_runner"}
        _workflow_llm(monkeypatch, _draft(workflow_tool=runner))

        await _chat(
            "create_workflow_from_description",
            {"description": "Search."},
            db_session,
            test_user.id,
        )

        assert await _all(db_session, UserTool) == []

    async def test_synthesized_tools_are_saved_and_wired_to_their_nodes(
        self, db_session, test_user, monkeypatch
    ):
        nodes = [
            {"node_id": "start", "node_type": "start"},
            {"node_id": "fetch", "node_type": "tool", "tool_name": "fetch_prices"},
            {"node_id": "end", "node_type": "end"},
        ]
        edges = [
            {"source_node_id": "start", "target_node_id": "fetch"},
            {"source_node_id": "fetch", "target_node_id": "end"},
        ]
        _workflow_llm(
            monkeypatch,
            _draft(nodes=nodes, edges=edges, custom_tools=[TRANSFORM_TOOL]),
        )

        result = await _chat(
            "create_workflow_from_description",
            {"description": "Reshape prices.", "synthesize_custom_tools": True},
            db_session,
            test_user.id,
        )

        assert "error" not in result, result
        tools = await _all(db_session, UserTool)
        assert len(tools) == 1
        tool = tools[0]
        assert tool.user_id == test_user.id
        assert tool.name == "fetch_prices"
        assert tool.tool_type == "transform"
        assert tool.config == {"template": "{{ input }}"}
        assert result["created_custom_tools"] == [
            {"id": str(tool.id), "name": "fetch_prices", "tool_type": "transform"}
        ]
        fetch = [
            n for n in await _all(db_session, WorkflowNode) if n.node_id == "fetch"
        ]
        assert fetch[0].tool_id == tool.id

    async def test_synthesized_tools_are_not_saved_unless_asked(
        self, db_session, test_user, monkeypatch
    ):
        _workflow_llm(monkeypatch, _draft(custom_tools=[TRANSFORM_TOOL]))

        result = await _chat(
            "create_workflow_from_description",
            {"description": "Reshape prices."},
            db_session,
            test_user.id,
        )

        assert await _all(db_session, UserTool) == []
        assert result["created_custom_tools"] == []

    async def test_a_docker_tool_is_skipped_while_docker_tools_are_disabled(
        self, db_session, test_user, monkeypatch
    ):
        monkeypatch.setattr(
            agent_service_module.settings, "CUSTOM_TOOL_DOCKER_ENABLED", False
        )
        docker = dict(TRANSFORM_TOOL, tool_type="docker_container")
        _workflow_llm(monkeypatch, _draft(custom_tools=[docker]))

        result = await _chat(
            "create_workflow_from_description",
            {"description": "Run a container.", "synthesize_custom_tools": True},
            db_session,
            test_user.id,
        )

        assert await _all(db_session, UserTool) == []
        assert any("CUSTOM_TOOL_DOCKER_ENABLED" in w for w in result["warnings"])

    async def test_another_users_tool_of_the_same_name_is_not_reused(
        self, db_session, test_user, monkeypatch
    ):
        stranger = await _stranger(db_session)
        theirs = UserTool(
            user_id=stranger.id,
            name="fetch_prices",
            tool_type="webhook",
            config={"url": "https://example.com/hook"},
        )
        db_session.add(theirs)
        await db_session.commit()
        _workflow_llm(monkeypatch, _draft(custom_tools=[TRANSFORM_TOOL]))

        result = await _chat(
            "create_workflow_from_description",
            {"description": "Reshape prices.", "synthesize_custom_tools": True},
            db_session,
            test_user.id,
        )

        mine = [
            t for t in await _all(db_session, UserTool) if t.user_id == test_user.id
        ]
        assert [t.tool_type for t in mine] == ["transform"]
        assert result["created_custom_tools"][0]["id"] == str(mine[0].id)

    async def test_a_node_cannot_be_pointed_at_another_users_tool(
        self, db_session, test_user, monkeypatch
    ):
        stranger = await _stranger(db_session)
        theirs = UserTool(
            user_id=stranger.id,
            name="their_hook",
            tool_type="webhook",
            config={"url": "https://example.com/hook"},
        )
        db_session.add(theirs)
        await db_session.commit()
        nodes = [
            {"node_id": "start", "node_type": "start"},
            {"node_id": "call", "node_type": "tool", "tool_id": str(theirs.id)},
            {"node_id": "end", "node_type": "end"},
        ]
        _workflow_llm(monkeypatch, _draft(nodes=nodes, edges=[]))

        result = await _chat(
            "create_workflow_from_description",
            {"description": "Call the hook."},
            db_session,
            test_user.id,
        )

        saved = await _all(db_session, WorkflowNode)
        assert all(n.tool_id != theirs.id for n in saved)
        assert any("Unknown custom tool id" in w for w in result.get("warnings", []))

    async def test_the_workflow_belongs_to_the_caller_only(
        self, db_session, test_user, monkeypatch
    ):
        stranger = await _stranger(db_session)
        _workflow_llm(monkeypatch)

        await _chat(
            "create_workflow_from_description",
            {"description": "Search."},
            db_session,
            test_user.id,
        )

        owners = {w.user_id for w in await _all(db_session, Workflow)}
        assert owners == {test_user.id}
        assert stranger.id not in owners
