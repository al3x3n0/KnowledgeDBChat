"""The batch tools: batch_search and batch_summarize.

These call the real handlers. The file used to restate each handler inline
and assert on the restatement -- `queries = raw[:10]; assert len(queries) ==
10` -- so thirty tests passed whatever the tools did.

`batch_search` is run against a fake search service (the real one reaches the
vector store); `batch_summarize` reads real `Document` rows from the in-memory
database, with only the LLM summarisation replaced.
"""

import hashlib
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import pytest

from app.models.document import Document, DocumentSource
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_observability_provider,
)
from app.services.document_service import DocumentService

pytestmark = pytest.mark.unit


class FakeSearch:
    """Stands in for SearchService.search, with the same keyword names."""

    def __init__(self, by_query=None, failing=()):
        self.by_query = by_query or {}
        self.failing = dict(failing)
        self.calls = []

    async def search(
        self, query, mode="smart", page=1, page_size=10, source_id=None, db=None
    ):
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "page": page,
                "page_size": page_size,
                "source_id": source_id,
                "db": db,
            }
        )
        if query in self.failing:
            raise RuntimeError(self.failing[query])
        rows = self.by_query.get(query, [])
        return list(rows), len(rows), 1


class FakeSummarizer:
    """Stands in for the LLM-backed DocumentService.summarize_document."""

    def __init__(self, summaries=None, failing=()):
        self.summaries = summaries or {}
        self.failing = dict(failing)
        self.calls = []

    async def summarize_document(self, document_id, db, **kwargs):
        self.calls.append(document_id)
        if document_id in self.failing:
            raise RuntimeError(self.failing[document_id])
        return self.summaries.get(document_id, f"summary of {document_id}")


def _row(doc_id, title=None, **extra):
    return {"id": doc_id, "title": title or f"title {doc_id}", **extra}


async def _run(tool, params, *, search=None, documents=None, db=None, user_id=None):
    user_id = user_id or uuid4()
    provider = build_autonomous_observability_provider(
        SimpleNamespace(search_service=search, document_service=documents)
    )
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(user_id),
        job=SimpleNamespace(id=uuid4(), user_id=user_id, goal="g", config={}),
        state={},
    )
    return await provider._handlers[tool](params, ctx)


def _ids(query_result):
    return [row["id"] for row in query_result["results"]]


# --------------------------------------------------------------------------
# batch_search
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params", [{}, {"queries": []}, {"queries": "one query"}, {"queries": None}]
)
async def test_search_refuses_without_a_query_list(params):
    search = FakeSearch()

    result = await _run("batch_search", params, search=search)

    assert "queries" in result["error"]
    assert "success" not in result
    assert search.calls == []


async def test_search_refuses_when_every_query_is_blank():
    search = FakeSearch()

    result = await _run("batch_search", {"queries": ["", "   "]}, search=search)

    assert "error" in result
    assert search.calls == []


async def test_each_query_is_searched_and_reported_in_order():
    search = FakeSearch({"cache": [_row("a"), _row("b")], "branch": [_row("c")]})
    db = object()

    result = await _run(
        "batch_search", {"queries": ["cache", "  ", " branch "]}, search=search, db=db
    )

    assert result["success"] is True
    assert [c["query"] for c in search.calls] == ["cache", "branch"]
    data = result["data"]
    assert data["queries_executed"] == 2
    assert [qr["query"] for qr in data["results"]] == ["cache", "branch"]
    assert _ids(data["results"][0]) == ["a", "b"]
    assert data["results"][0]["total"] == 2
    assert _ids(data["results"][1]) == ["c"]
    assert data["total_unique_documents"] == 3
    # Searched the way search_documents does, on the run's own session.
    assert {c["mode"] for c in search.calls} == {"smart"}
    assert all(c["db"] is db for c in search.calls)


async def test_at_most_ten_queries_run():
    search = FakeSearch()

    result = await _run(
        "batch_search", {"queries": [f"q{i}" for i in range(15)]}, search=search
    )

    assert [c["query"] for c in search.calls] == [f"q{i}" for i in range(10)]
    assert result["data"]["queries_executed"] == 10
    assert len(result["data"]["results"]) == 10


@pytest.mark.parametrize(
    "given, expected",
    [({}, 5), ({"limit_per_query": 3}, 3), ({"limit_per_query": 500}, 20)],
)
async def test_the_per_query_limit_defaults_to_five_and_caps_at_twenty(given, expected):
    search = FakeSearch()

    await _run("batch_search", {"queries": ["a", "b"], **given}, search=search)

    assert [c["page_size"] for c in search.calls] == [expected, expected]


async def test_a_negative_per_query_limit_never_reaches_the_search():
    search = FakeSearch()

    result = await _run(
        "batch_search", {"queries": ["a"], "limit_per_query": -3}, search=search
    )

    assert "error" in result or all(c["page_size"] >= 1 for c in search.calls)


async def test_a_non_numeric_per_query_limit_is_an_error_result():
    search = FakeSearch()

    result = await _run(
        "batch_search", {"queries": ["a"], "limit_per_query": "lots"}, search=search
    )

    assert "error" in result
    assert search.calls == []


async def test_the_source_filter_is_passed_through_or_absent():
    search = FakeSearch()

    await _run(
        "batch_search", {"queries": ["a"], "source_id": " src-1 "}, search=search
    )
    await _run("batch_search", {"queries": ["a"]}, search=search)
    await _run("batch_search", {"queries": ["a"], "source_id": ""}, search=search)

    assert [c["source_id"] for c in search.calls] == ["src-1", None, None]


async def test_a_document_found_twice_is_reported_once_by_default():
    search = FakeSearch(
        {"first": [_row("a"), _row("b")], "second": [_row("b"), _row("c")]}
    )

    result = await _run("batch_search", {"queries": ["first", "second"]}, search=search)

    first, second = result["data"]["results"]
    assert _ids(first) == ["a", "b"]
    assert _ids(second) == ["c"]
    assert result["data"]["total_unique_documents"] == 3


async def test_an_explicit_null_deduplicate_means_the_default():
    search = FakeSearch({"first": [_row("a")], "second": [_row("a")]})

    result = await _run(
        "batch_search",
        {"queries": ["first", "second"], "deduplicate": None},
        search=search,
    )

    assert _ids(result["data"]["results"][1]) == []


async def test_deduplication_can_be_turned_off():
    search = FakeSearch(
        {"first": [_row("a"), _row("b")], "second": [_row("b"), _row("c")]}
    )

    result = await _run(
        "batch_search",
        {"queries": ["first", "second"], "deduplicate": False},
        search=search,
    )

    first, second = result["data"]["results"]
    assert _ids(first) == ["a", "b"]
    assert _ids(second) == ["b", "c"]
    # Still a count of documents, not of rows.
    assert result["data"]["total_unique_documents"] == 3


async def test_one_failing_query_does_not_lose_the_others():
    search = FakeSearch(
        {"before": [_row("a")], "after": [_row("b")]},
        failing={"broken": "E" * 500},
    )

    result = await _run(
        "batch_search", {"queries": ["before", "broken", "after"]}, search=search
    )

    assert result["success"] is True
    before, broken, after = result["data"]["results"]
    assert _ids(before) == ["a"]
    assert _ids(after) == ["b"]
    assert broken["query"] == "broken"
    assert broken["results"] == []
    assert broken["total"] == 0
    assert broken["error"] == "E" * 200
    assert "error" not in before and "error" not in after
    assert result["data"]["queries_executed"] == 3
    assert result["data"]["total_unique_documents"] == 2
    assert [f["id"] for f in result["findings"]] == ["a", "b"]


async def test_findings_name_the_top_five_documents_of_each_query():
    search = FakeSearch(
        {
            "many": [_row(f"m{i}", relevance_score=1 - i / 10) for i in range(8)],
            "one": [_row("z", "Zed", score=0.4)],
        }
    )

    result = await _run(
        "batch_search",
        {"queries": ["many", "one"], "limit_per_query": 8},
        search=search,
    )

    findings = result["findings"]
    assert [f["id"] for f in findings] == ["m0", "m1", "m2", "m3", "m4", "z"]
    assert findings[0] == {
        "type": "document",
        "title": "title m0",
        "id": "m0",
        "score": 1.0,
        "query": "many",
    }
    # A row scored under `score` rather than `relevance_score` keeps its score.
    assert findings[-1] == {
        "type": "document",
        "title": "Zed",
        "id": "z",
        "score": 0.4,
        "query": "one",
    }
    # The result itself is not cut to five.
    assert len(result["data"]["results"][0]["results"]) == 8


async def test_no_results_is_a_success_with_nothing_in_it():
    search = FakeSearch()

    result = await _run("batch_search", {"queries": ["nothing"]}, search=search)

    assert result["success"] is True
    assert result["data"]["results"] == [
        {"query": "nothing", "results": [], "total": 0}
    ]
    assert result["data"]["total_unique_documents"] == 0
    assert result["findings"] == []


# --------------------------------------------------------------------------
# batch_summarize
# --------------------------------------------------------------------------


async def _source(db):
    source = DocumentSource(name="Uploads", source_type="file", config={})
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _doc(db, source, title, summary=None, content="body"):
    doc = Document(
        title=title,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        summary=summary,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


@pytest.mark.parametrize(
    "params",
    [{}, {"document_ids": []}, {"document_ids": "abc"}, {"document_ids": None}],
)
async def test_summarize_refuses_without_an_id_list(db_session, params):
    summarizer = FakeSummarizer()

    result = await _run("batch_summarize", params, documents=summarizer, db=db_session)

    assert "document_ids" in result["error"]
    assert "success" not in result
    assert summarizer.calls == []


async def test_existing_summaries_are_returned_without_generating(db_session):
    source = await _source(db_session)
    with_summary = await _doc(db_session, source, "Has one", summary="Already written.")
    without = await _doc(db_session, source, "Has none")
    summarizer = FakeSummarizer()

    result = await _run(
        "batch_summarize",
        {"document_ids": [str(with_summary.id), str(without.id)]},
        documents=summarizer,
        db=db_session,
    )

    assert result["success"] is True
    assert result["data"]["summaries"] == [
        {
            "document_id": str(with_summary.id),
            "title": "Has one",
            "summary": "Already written.",
            "status": "available",
        },
        {"document_id": str(without.id), "title": "Has none", "status": "no_summary"},
    ]
    assert result["data"]["total_requested"] == 2
    assert result["data"]["available"] == 1
    assert result["data"]["missing"] == 1
    # generate_missing defaults to false: no model call was spent.
    assert summarizer.calls == []


async def test_unknown_and_malformed_ids_do_not_lose_the_others(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Real", summary="S")
    unknown = str(uuid4())

    result = await _run(
        "batch_summarize",
        {"document_ids": ["not-a-uuid", unknown, str(doc.id)]},
        documents=FakeSummarizer(),
        db=db_session,
    )

    assert result["success"] is True
    malformed, missing, real = result["data"]["summaries"]
    assert malformed == {"document_id": "not-a-uuid", "status": "error"}
    assert missing == {"document_id": unknown, "status": "not_found"}
    assert real["status"] == "available"
    assert real["summary"] == "S"
    assert result["data"]["total_requested"] == 3
    assert result["data"]["available"] == 1
    assert result["data"]["missing"] == 2


async def test_at_most_twenty_documents_are_summarized(db_session):
    ids = [str(uuid4()) for _ in range(25)]

    result = await _run(
        "batch_summarize",
        {"document_ids": ids},
        documents=FakeSummarizer(),
        db=db_session,
    )

    assert [s["document_id"] for s in result["data"]["summaries"]] == ids[:20]
    assert result["data"]["total_requested"] == 20


async def test_blank_ids_are_dropped(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Real", summary="S")

    result = await _run(
        "batch_summarize",
        {"document_ids": ["", "  ", f" {doc.id} "]},
        documents=FakeSummarizer(),
        db=db_session,
    )

    assert [s["document_id"] for s in result["data"]["summaries"]] == [str(doc.id)]
    assert result["data"]["total_requested"] == 1


async def test_generate_missing_summarizes_only_what_lacks_a_summary(db_session):
    source = await _source(db_session)
    has = await _doc(db_session, source, "Has one", summary="Already written.")
    lacks = await _doc(db_session, source, "Has none")
    summarizer = FakeSummarizer({lacks.id: "Fresh summary."})

    result = await _run(
        "batch_summarize",
        {"document_ids": [str(has.id), str(lacks.id)], "generate_missing": True},
        documents=summarizer,
        db=db_session,
    )

    assert summarizer.calls == [lacks.id]
    assert result["data"]["summaries"][0]["status"] == "available"
    assert result["data"]["summaries"][1] == {
        "document_id": str(lacks.id),
        "title": "Has none",
        "summary": "Fresh summary.",
        "status": "generated",
    }
    assert result["data"]["available"] == 2
    assert result["data"]["missing"] == 0


async def test_generate_missing_calls_the_document_service_as_it_is_declared(
    db_session,
):
    source = await _source(db_session)
    lacks = await _doc(db_session, source, "Has none")
    # Signature-checked against the real class: only the LLM work is replaced.
    documents = mock.create_autospec(DocumentService, instance=True)
    documents.summarize_document.return_value = "Fresh summary."

    result = await _run(
        "batch_summarize",
        {"document_ids": [str(lacks.id)], "generate_missing": True},
        documents=documents,
        db=db_session,
    )

    (entry,) = result["data"]["summaries"]
    assert entry["status"] == "generated", entry
    assert entry["summary"] == "Fresh summary."


async def test_one_failed_generation_does_not_lose_the_others(db_session):
    source = await _source(db_session)
    first = await _doc(db_session, source, "First")
    broken = await _doc(db_session, source, "Broken")
    last = await _doc(db_session, source, "Last")
    summarizer = FakeSummarizer(
        {first.id: "one", last.id: "three"}, failing={broken.id: "E" * 500}
    )

    result = await _run(
        "batch_summarize",
        {
            "document_ids": [str(first.id), str(broken.id), str(last.id)],
            "generate_missing": True,
        },
        documents=summarizer,
        db=db_session,
    )

    assert result["success"] is True
    one, failed, three = result["data"]["summaries"]
    assert (one["status"], one["summary"]) == ("generated", "one")
    assert (three["status"], three["summary"]) == ("generated", "three")
    assert failed == {
        "document_id": str(broken.id),
        "title": "Broken",
        "status": "generation_failed",
        "error": "E" * 200,
    }
    assert result["data"]["available"] == 2
    assert result["data"]["missing"] == 1


async def test_a_generation_that_produced_nothing_is_not_counted_available(db_session):
    source = await _source(db_session)
    lacks = await _doc(db_session, source, "Has none")
    summarizer = FakeSummarizer({lacks.id: None})

    result = await _run(
        "batch_summarize",
        {"document_ids": [str(lacks.id)], "generate_missing": True},
        documents=summarizer,
        db=db_session,
    )

    (entry,) = result["data"]["summaries"]
    assert entry["status"] != "generated"
    assert result["data"]["available"] == 0
    assert result["data"]["missing"] == 1


# --------------------------------------------------------------------------
# Declarations: schema and registry
# --------------------------------------------------------------------------


class TestBatchToolSchemas:
    """Tests for batch tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "batch_search" in names
        assert "batch_summarize" in names

    def test_batch_search_requires_queries(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("batch_search")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "queries" in required

    def test_batch_summarize_requires_document_ids(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("batch_summarize")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_ids" in required

    def test_batch_search_has_deduplicate_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("batch_search")
        assert "deduplicate" in tool["parameters"]["properties"]

    def test_batch_summarize_has_generate_missing_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("batch_summarize")
        assert "generate_missing" in tool["parameters"]["properties"]

    def test_batch_search_queries_is_array(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("batch_search")
        queries_prop = tool["parameters"]["properties"]["queries"]
        assert queries_prop["type"] == "array"

    def test_batch_summarize_ids_is_array(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("batch_summarize")
        ids_prop = tool["parameters"]["properties"]["document_ids"]
        assert ids_prop["type"] == "array"


class TestBatchToolRegistry:
    """Tests for batch tool registry classification."""

    def test_batch_search_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("batch_search")
        assert meta is not None
        assert meta.effects == "read"

    def test_batch_summarize_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("batch_summarize")
        assert meta is not None
        assert meta.effects == "read"

    def test_both_are_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["batch_search", "batch_summarize"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "medium"

    def test_neither_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["batch_search", "batch_summarize"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
