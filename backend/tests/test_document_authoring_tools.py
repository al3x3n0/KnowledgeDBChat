"""The document authoring tools: list_documents_by_tag and merge_documents.

These call the real handlers against the in-memory database. The file used to
restate each handler inline and assert on the restatement -- `tags = None;
assert not tags` -- so twenty-five tests passed whatever the tools did.

Only the indexing step (`reprocess_document`, which reaches the vector store)
is replaced; the notes source and the merged row are created by the real
`DocumentService` and read back from the database.
"""

import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.document import Document, DocumentSource
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_document_provider,
)
from app.services.document_service import DocumentService

pytestmark = pytest.mark.unit


class _Indexing(DocumentService):
    """The real document service with the vector-store step recorded."""

    def __init__(self, fail=False):
        self.reprocessed = []
        self._fail = fail

    async def reprocess_document(self, document_id, db, user_id=None):
        self.reprocessed.append((document_id, user_id))
        if self._fail:
            raise RuntimeError("vector store unreachable")
        return True


def _job(user):
    return SimpleNamespace(
        id=uuid4(), user_id=user.id, goal="Merge the notes", config={}
    )


async def _run(tool, params, db, user, service=None, job=None):
    service = service or _Indexing()
    provider = build_autonomous_document_provider(
        SimpleNamespace(document_service=service, search_service=None)
    )
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(user.id),
        job=job or _job(user),
        state={},
    )
    return await provider._handlers[tool](params, ctx)


async def _source(db, name="Uploads"):
    source = DocumentSource(name=name, source_type="file", config={})
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _doc(db, source, title, content="body", tags=None, **extra):
    doc = Document(
        title=title,
        content=content,
        content_hash=hashlib.sha256((content or "").encode()).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        tags=tags,
        **extra,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _all_documents(db):
    return (await db.execute(select(Document))).scalars().all()


def _titles(result):
    return sorted(d["title"] for d in result["data"]["documents"])


# --------------------------------------------------------------------------
# list_documents_by_tag
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"tags": []}, {"tags": "ml"}, {"tags": None}])
async def test_listing_refuses_without_a_tag_list(db_session, test_user, params):
    result = await _run("list_documents_by_tag", params, db_session, test_user)

    assert "error" in result
    assert "tags" in result["error"]
    assert "success" not in result


async def test_any_tag_matches_by_default(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "both", tags=["ml", "cache"])
    await _doc(db_session, source, "ml only", tags=["ml"])
    await _doc(db_session, source, "cache only", tags=["cache"])
    await _doc(db_session, source, "other", tags=["compilers"])
    await _doc(db_session, source, "untagged", tags=None)
    await _doc(db_session, source, "empty tags", tags=[])

    result = await _run(
        "list_documents_by_tag", {"tags": ["ml", "cache"]}, db_session, test_user
    )

    assert result["success"] is True
    assert result["data"]["match_all"] is False
    assert _titles(result) == ["both", "cache only", "ml only"]
    assert result["data"]["count"] == 3
    assert sorted(result["data"]["tags"]) == ["cache", "ml"]


async def test_match_all_requires_every_tag(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "both", tags=["ml", "cache", "extra"])
    await _doc(db_session, source, "ml only", tags=["ml"])
    await _doc(db_session, source, "cache only", tags=["cache"])

    result = await _run(
        "list_documents_by_tag",
        {"tags": ["ml", "cache"], "match_all": True},
        db_session,
        test_user,
    )

    assert result["data"]["match_all"] is True
    assert _titles(result) == ["both"]
    assert result["data"]["count"] == 1


async def test_a_tag_nobody_has_lists_nothing(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])

    result = await _run(
        "list_documents_by_tag", {"tags": ["absent"]}, db_session, test_user
    )

    assert result["success"] is True
    assert result["data"]["documents"] == []
    assert result["data"]["count"] == 0


async def test_tags_are_matched_exactly_not_as_substrings(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "long", tags=["machine-learning"])
    await _doc(db_session, source, "short", tags=["ml"])

    result = await _run(
        "list_documents_by_tag", {"tags": ["machine"]}, db_session, test_user
    )

    assert result["data"]["documents"] == []


async def test_surrounding_whitespace_in_a_requested_tag_is_ignored(
    db_session, test_user
):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])

    result = await _run(
        "list_documents_by_tag", {"tags": ["  ml  "]}, db_session, test_user
    )

    assert _titles(result) == ["a"]
    assert result["data"]["tags"] == ["ml"]


async def test_a_listed_document_carries_its_fields(db_session, test_user):
    source = await _source(db_session)
    doc = await _doc(
        db_session,
        source,
        "Cache study",
        tags=["ml", "cache"],
        file_type="text/markdown",
        summary="S" * 500,
    )

    result = await _run(
        "list_documents_by_tag", {"tags": ["ml"]}, db_session, test_user
    )

    (listed,) = result["data"]["documents"]
    assert listed["id"] == str(doc.id)
    assert listed["title"] == "Cache study"
    assert listed["tags"] == ["ml", "cache"]
    assert listed["file_type"] == "text/markdown"
    assert listed["summary"] == "S" * 200
    assert listed["created_at"] == doc.created_at.isoformat()
    # A listing is not the place the full text travels.
    assert "content" not in listed


async def test_a_document_without_a_summary_lists_an_empty_one(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])

    result = await _run(
        "list_documents_by_tag", {"tags": ["ml"]}, db_session, test_user
    )

    assert result["data"]["documents"][0]["summary"] == ""


async def test_listing_defaults_to_twenty_and_honours_a_smaller_limit(
    db_session, test_user
):
    source = await _source(db_session)
    for i in range(25):
        await _doc(db_session, source, f"d{i}", tags=["ml"])

    default = await _run(
        "list_documents_by_tag", {"tags": ["ml"]}, db_session, test_user
    )
    small = await _run(
        "list_documents_by_tag", {"tags": ["ml"], "limit": 3}, db_session, test_user
    )

    assert default["data"]["count"] == 20
    assert len(default["data"]["documents"]) == 20
    assert small["data"]["count"] == 3


async def test_listing_is_capped_at_one_hundred(db_session, test_user):
    source = await _source(db_session)
    db_session.add_all(
        [
            Document(
                title=f"d{i}",
                content="x",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{i}",
                tags=["ml"],
            )
            for i in range(105)
        ]
    )
    await db_session.commit()

    result = await _run(
        "list_documents_by_tag", {"tags": ["ml"], "limit": 1000}, db_session, test_user
    )

    assert result["data"]["count"] == 100
    assert len(result["data"]["documents"]) == 100


async def test_a_match_is_found_however_many_tagged_documents_exist(
    db_session, test_user
):
    source = await _source(db_session)
    db_session.add_all(
        [
            Document(
                title=f"noise{i}",
                content="x",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{i}",
                tags=["noise"],
            )
            for i in range(500)
        ]
    )
    await db_session.commit()
    await _doc(db_session, source, "the one", tags=["rare"])

    result = await _run(
        "list_documents_by_tag", {"tags": ["rare"]}, db_session, test_user
    )

    assert _titles(result) == ["the one"]


async def test_blank_tags_with_match_all_do_not_list_everything(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])
    await _doc(db_session, source, "b", tags=["cache"])

    result = await _run(
        "list_documents_by_tag",
        {"tags": ["  ", ""], "match_all": True},
        db_session,
        test_user,
    )

    assert "error" in result or result["data"]["documents"] == []


async def test_blank_tags_without_match_all_list_nothing(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])

    result = await _run("list_documents_by_tag", {"tags": [" "]}, db_session, test_user)

    assert "error" in result or result["data"]["documents"] == []


async def test_a_negative_limit_does_not_drop_matches(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])
    await _doc(db_session, source, "b", tags=["ml"])

    result = await _run(
        "list_documents_by_tag", {"tags": ["ml"], "limit": -1}, db_session, test_user
    )

    assert "error" in result or result["data"]["count"] == 2


async def test_a_non_numeric_limit_is_refused_not_raised(db_session, test_user):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml"])

    result = await _run(
        "list_documents_by_tag",
        {"tags": ["ml"], "limit": "many"},
        db_session,
        test_user,
    )

    assert "error" in result or result["data"]["count"] == 1


# --------------------------------------------------------------------------
# merge_documents
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, named",
    [
        ({"title": "T"}, "document_ids"),
        ({"title": "T", "document_ids": []}, "document_ids"),
        ({"title": "T", "document_ids": "abc"}, "document_ids"),
        ({"document_ids": ["x"]}, "title"),
        ({"document_ids": ["x"], "title": "   "}, "title"),
    ],
)
async def test_merge_refuses_without_ids_or_title(db_session, test_user, params, named):
    service = _Indexing()

    result = await _run("merge_documents", params, db_session, test_user, service)

    assert named in result["error"]
    assert await _all_documents(db_session) == []
    assert service.reprocessed == []


async def test_merge_stores_one_document_built_from_its_sources(db_session, test_user):
    source = await _source(db_session)
    first = await _doc(db_session, source, "Alpha", content="alpha body")
    second = await _doc(db_session, source, "Beta", content="beta body")
    service = _Indexing()
    job = _job(test_user)

    result = await _run(
        "merge_documents",
        {
            "document_ids": [str(second.id), str(first.id)],
            "title": "  Combined  ",
            "tags": ["merged", "report"],
        },
        db_session,
        test_user,
        service,
        job,
    )

    assert result.get("success") is True, result
    expected = "# Beta\n\nbeta body\n\n---\n\n# Alpha\n\nalpha body"
    merged_id = result["data"]["document_id"]

    stored = [d for d in await _all_documents(db_session) if str(d.id) == merged_id]
    assert len(stored) == 1
    merged = stored[0]
    assert merged.title == "Combined"
    assert merged.content == expected
    assert merged.content_hash == hashlib.sha256(expected.encode()).hexdigest()
    assert merged.file_size == len(expected.encode())
    assert merged.file_type == "text/plain"
    assert merged.tags == ["merged", "report"]
    assert merged.source_identifier.startswith("agent_merge:")
    assert merged.extra_metadata == {
        "origin": "agent_merge",
        "source_document_ids": [str(second.id), str(first.id)],
        "job_id": str(job.id),
    }

    notes = await db_session.get(DocumentSource, merged.source_id)
    assert notes.name == "Agent Notes"
    assert merged.source_id != source.id

    assert result["data"] == {
        "document_id": merged_id,
        "title": "Combined",
        "source_count": 2,
        "content_length": len(expected),
    }
    assert result["artifacts"] == [
        {"type": "document", "id": merged_id, "title": "Combined"}
    ]
    # Three documents now: the two sources, untouched, and the merge.
    assert len(await _all_documents(db_session)) == 3
    await db_session.refresh(first)
    assert first.content == "alpha body"


async def test_the_merged_document_is_sent_for_indexing(db_session, test_user):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Alpha")
    service = _Indexing()

    result = await _run(
        "merge_documents",
        {"document_ids": [str(doc.id)], "title": "M"},
        db_session,
        test_user,
        service,
    )

    assert [(str(i), u) for i, u in service.reprocessed] == [
        (result["data"]["document_id"], test_user.id)
    ]


async def test_a_failed_indexing_step_does_not_lose_the_merge(db_session, test_user):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Alpha")

    result = await _run(
        "merge_documents",
        {"document_ids": [str(doc.id)], "title": "M"},
        db_session,
        test_user,
        _Indexing(fail=True),
    )

    assert result.get("success") is True, result
    titles = sorted(d.title for d in await _all_documents(db_session))
    assert titles == ["Alpha", "M"]
    assert "could not be indexed" in result["data"]["warning"]


async def test_an_id_given_twice_is_merged_once(db_session, test_user):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Alpha", content="once")

    result = await _run(
        "merge_documents",
        {"document_ids": [str(doc.id), str(doc.id)], "title": "M"},
        db_session,
        test_user,
    )

    assert result["data"]["source_count"] == 1
    assert result["data"]["skipped"] == [
        {"id": str(doc.id), "reason": "listed more than once"}
    ]
    merged = await db_session.get(
        Document, doc.id.__class__(result["data"]["document_id"])
    )
    assert merged.content == "# Alpha\n\nonce"


async def test_merge_uses_the_separator_it_is_given(db_session, test_user):
    source = await _source(db_session)
    a = await _doc(db_session, source, "A", content="one")
    b = await _doc(db_session, source, "B", content="two")

    result = await _run(
        "merge_documents",
        {"document_ids": [str(a.id), str(b.id)], "title": "M", "separator": "\n==\n"},
        db_session,
        test_user,
    )

    merged = await db_session.get(
        Document, a.id.__class__(result["data"]["document_id"])
    )
    assert merged.content == "# A\n\none\n==\n# B\n\ntwo"


async def test_a_null_separator_means_the_default(db_session, test_user):
    source = await _source(db_session)
    a = await _doc(db_session, source, "A", content="one")
    b = await _doc(db_session, source, "B", content="two")

    result = await _run(
        "merge_documents",
        {"document_ids": [str(a.id), str(b.id)], "title": "M", "separator": None},
        db_session,
        test_user,
    )

    merged = await db_session.get(
        Document, a.id.__class__(result["data"]["document_id"])
    )
    assert merged.content == "# A\n\none\n\n---\n\n# B\n\ntwo"


async def test_merge_without_tags_stores_none(db_session, test_user):
    source = await _source(db_session)
    a = await _doc(db_session, source, "A", tags=["kept-on-source"])

    for tags in ({}, {"tags": "not-a-list"}):
        result = await _run(
            "merge_documents",
            {"document_ids": [str(a.id)], "title": "M", **tags},
            db_session,
            test_user,
        )
        merged = await db_session.get(
            Document, a.id.__class__(result["data"]["document_id"])
        )
        assert merged.tags == []


async def test_merge_takes_at_most_twenty_documents(db_session, test_user):
    source = await _source(db_session)
    docs = [await _doc(db_session, source, f"d{i}", content=f"c{i}") for i in range(22)]
    ids = [str(d.id) for d in docs]

    result = await _run(
        "merge_documents", {"document_ids": ids, "title": "M"}, db_session, test_user
    )

    assert result["data"]["source_count"] == 20
    # The two left out are named, not dropped in silence.
    assert result["data"]["skipped"] == [
        {"id": i, "reason": "over the 20 limit"} for i in ids[20:]
    ]
    merged = await db_session.get(
        Document, docs[0].id.__class__(result["data"]["document_id"])
    )
    assert merged.extra_metadata["source_document_ids"] == ids[:20]
    assert "# d19\n" in merged.content
    assert "# d20\n" not in merged.content


async def test_merge_skips_ids_it_cannot_use_and_keeps_the_rest(db_session, test_user):
    source = await _source(db_session)
    good = await _doc(db_session, source, "Good", content="kept")
    empty = await _doc(db_session, source, "Empty", content=None)

    result = await _run(
        "merge_documents",
        {
            "document_ids": ["not-a-uuid", str(uuid4()), str(empty.id), str(good.id)],
            "title": "M",
        },
        db_session,
        test_user,
    )

    assert result.get("success") is True, result
    assert result["data"]["source_count"] == 1
    missing = result["data"]["skipped"][1]["id"]
    assert result["data"]["skipped"] == [
        {"id": "not-a-uuid", "reason": "not a document id"},
        {"id": missing, "reason": "no such document"},
        {"id": str(empty.id), "reason": "document has no content"},
    ]
    merged = await db_session.get(
        Document, good.id.__class__(result["data"]["document_id"])
    )
    assert merged.content == "# Good\n\nkept"
    assert merged.extra_metadata["source_document_ids"] == [str(good.id)]


async def test_merge_of_nothing_usable_stores_nothing(db_session, test_user):
    source = await _source(db_session)
    empty = await _doc(db_session, source, "Empty", content=None)
    service = _Indexing()

    result = await _run(
        "merge_documents",
        {"document_ids": ["not-a-uuid", str(uuid4()), str(empty.id)], "title": "M"},
        db_session,
        test_user,
        service,
    )

    assert "error" in result
    assert "success" not in result
    assert [d.title for d in await _all_documents(db_session)] == ["Empty"]
    assert service.reprocessed == []


async def test_merge_over_two_megabytes_is_refused_and_stores_nothing(
    db_session, test_user
):
    source = await _source(db_session)
    a = await _doc(db_session, source, "A", content="x" * 1_100_000)
    b = await _doc(db_session, source, "B", content="y" * 1_100_000)
    service = _Indexing()

    result = await _run(
        "merge_documents",
        {"document_ids": [str(a.id), str(b.id)], "title": "M"},
        db_session,
        test_user,
        service,
    )

    assert "2MB" in result["error"]
    assert sorted(d.title for d in await _all_documents(db_session)) == ["A", "B"]
    assert service.reprocessed == []


async def test_two_merges_are_two_documents_in_one_notes_source(db_session, test_user):
    source = await _source(db_session)
    a = await _doc(db_session, source, "A")
    params = {"document_ids": [str(a.id)], "title": "M"}

    first = await _run("merge_documents", params, db_session, test_user)
    second = await _run("merge_documents", params, db_session, test_user)

    assert first["data"]["document_id"] != second["data"]["document_id"]
    sources = (await db_session.execute(select(DocumentSource))).scalars().all()
    assert sorted(s.name for s in sources) == ["Agent Notes", "Uploads"]


# --------------------------------------------------------------------------
# Declarations: schema and registry
# --------------------------------------------------------------------------


class TestDocumentAuthoringToolSchemas:
    """Tests for document authoring tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "list_documents_by_tag" in names
        assert "merge_documents" in names

    def test_list_by_tag_requires_tags(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("list_documents_by_tag")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "tags" in required

    def test_merge_documents_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("merge_documents")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_ids" in required
        assert "title" in required

    def test_list_by_tag_has_match_all(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("list_documents_by_tag")
        assert "match_all" in tool["parameters"]["properties"]

    def test_merge_documents_has_separator(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("merge_documents")
        assert "separator" in tool["parameters"]["properties"]


class TestDocumentAuthoringToolRegistry:
    """Tests for document authoring tool registry classification."""

    def test_list_by_tag_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("list_documents_by_tag")
        assert meta is not None
        assert meta.effects == "read"

    def test_merge_documents_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("merge_documents")
        assert meta is not None
        assert meta.effects == "write"

    def test_both_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["list_documents_by_tag", "merge_documents"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
