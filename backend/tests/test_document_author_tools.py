"""The document author tools, driven through their real handlers.

plan_document, write_section, revise_section, insert_figure, assemble_document
and export_document share one piece of state (`state["document_workspace"]`),
so most tests here run several of them in sequence against one context and
then read what came out the far end: the assembled markdown, the bytes the
real DOCX/PDF builders produced, the row in the database.

This file used to build the workspace dict by hand and assert on its own dict
-- `target["content"] = "..."; assert target["content"] is not None` -- so
twenty-three tests passed without a handler being imported.

Nothing is faked except the LaTeX compiler (a subprocess) and, where the
bytes are inspected, a spy that calls the real builder and keeps what it
returned. Tests marked xfail(strict) record a defect in the handler.
"""

import io
import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.models.document import Document
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_document_authoring_provider,
)

pytestmark = pytest.mark.unit

SECTIONS = [
    {"id": "intro", "title": "Introduction", "description": "Opening section"},
    {"id": "methods", "title": "Methods", "description": "How it was measured"},
    {"id": "results", "title": "Results", "description": "Key findings"},
]

BODY = {
    "intro": "Prefetchers hide latency that caches cannot.",
    "methods": "Every kernel ran on gem5 with a stride prefetcher on L2.",
    "results": "The geomean speedup was 2.1x over four kernels.",
}


class _Recorder:
    """An executor that remembers which of its services a handler asked for."""

    def __init__(self):
        self.touched = []

    def __getattr__(self, name):
        self.touched.append(name)
        return SimpleNamespace()


class _Session:
    """One job's context: the state every call in a run shares."""

    def __init__(self, db=None, user=None, executor=None):
        self.executor = executor or SimpleNamespace()
        self.provider = build_autonomous_document_authoring_provider(self.executor)
        self.state = {}
        self.job = SimpleNamespace(
            id=uuid4(),
            user_id=getattr(user, "id", None),
            goal="Write the prefetcher report",
            config={},
            iteration=1,
        )
        self.ctx = AgentToolExecutionContext(
            mode="autonomous",
            db=db,
            service=None,
            user_id=str(getattr(user, "id", "") or ""),
            job=self.job,
            state=self.state,
        )

    async def call(self, tool, params=None):
        return await self.provider._handlers[tool](dict(params or {}), self.ctx)

    @property
    def workspace(self):
        return self.state.get("document_workspace")

    @property
    def markdown(self):
        return self.workspace["assembled_markdown"]

    async def plan(self, **overrides):
        params = {
            "title": "Prefetcher Study",
            "abstract": "What a prefetcher buys on four kernels.",
            "sections": SECTIONS,
        }
        params.update(overrides)
        result = await self.call("plan_document", params)
        assert result.get("success") is True, result
        return result

    async def write_all(self):
        for section_id, content in BODY.items():
            result = await self.call(
                "write_section", {"section_id": section_id, "content": content}
            )
            assert result.get("success") is True, result

    async def assembled(self, **params):
        result = await self.call("assemble_document", params)
        assert result.get("success") is True, result
        return self.markdown


def _refused(result):
    assert isinstance(result, dict)
    assert result.get("error")
    assert "success" not in result
    return result["error"]


def _in_order(text, *needles):
    positions = [text.index(needle) for needle in needles]
    return positions == sorted(positions)


def _spy_on_build(monkeypatch, cls):
    """Run the real builder and keep what it was given and what it returned."""
    calls = []
    real_build = cls.build

    def build(self, *args, **kwargs):
        data = real_build(self, *args, **kwargs)
        calls.append({"args": args, "kwargs": kwargs, "bytes": data})
        return data

    monkeypatch.setattr(cls, "build", build)
    return calls


def _record_pptx(monkeypatch, cls):
    """Record the outline handed to the PPTX builder.

    tests/conftest.py replaces python-pptx with a stub whose Presentation is
    `object`, so the real builder cannot run here; what these tests can check
    is the outline the handler derives from the document.
    """
    calls = []

    def build(self, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs, "bytes": b"PK-recorded"})
        return b"PK-recorded"

    monkeypatch.setattr(cls, "build", build)
    return calls


# --------------------------------------------------------------------------
# plan_document
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"sections": SECTIONS},
        {"title": "   ", "sections": SECTIONS},
        {"title": "Report"},
        {"title": "Report", "sections": []},
        {"title": "Report", "sections": "intro, methods"},
        {"title": "Report", "sections": None},
    ],
)
async def test_plan_refuses_without_a_title_and_sections(params):
    session = _Session()

    _refused(await session.call("plan_document", params))

    assert session.workspace is None


async def test_plan_stores_the_outline_the_model_gave():
    session = _Session()

    result = await session.call(
        "plan_document",
        {
            "title": "  Prefetcher Study  ",
            "abstract": "What a prefetcher buys.",
            "doc_type": "design_doc",
            "style": "academic",
            "sections": SECTIONS,
        },
    )

    assert result["success"] is True
    assert result["data"] == {
        "title": "Prefetcher Study",
        "sections_count": 3,
        "section_ids": ["intro", "methods", "results"],
    }
    plan = session.workspace["plan"]
    assert plan["title"] == "Prefetcher Study"
    assert plan["abstract"] == "What a prefetcher buys."
    assert plan["doc_type"] == "design_doc"
    assert plan["style"] == "academic"
    assert [s["title"] for s in plan["sections"]] == [
        "Introduction",
        "Methods",
        "Results",
    ]
    assert plan["sections"][1]["description"] == "How it was measured"
    for section in plan["sections"]:
        assert section["content"] is None
        assert section["revision_count"] == 0
        assert section["citations"] == []
        assert section["figures"] == []


async def test_plan_defaults_match_the_declared_defaults():
    session = _Session()

    await session.call("plan_document", {"title": "Report", "sections": SECTIONS})

    plan = session.workspace["plan"]
    assert plan["doc_type"] == "research_report"
    assert plan["style"] == "professional"
    assert plan["abstract"] == ""


async def test_a_section_without_an_id_is_given_one_that_can_be_written_to():
    session = _Session()

    result = await session.call(
        "plan_document",
        {"title": "Report", "sections": [{"title": "First"}, {"title": "Second"}]},
    )

    ids = result["data"]["section_ids"]
    assert len(set(ids)) == 2
    written = await session.call(
        "write_section", {"section_id": ids[1], "content": "second body"}
    )
    assert written["success"] is True
    assert session.workspace["plan"]["sections"][1]["content"] == "second body"
    assert session.workspace["plan"]["sections"][0]["content"] is None


async def test_plan_caps_sections_and_field_lengths():
    session = _Session()
    many = [{"id": f"s{i}", "title": "T" * 500} for i in range(40)]

    result = await session.call(
        "plan_document",
        {"title": "R" * 1000, "abstract": "a" * 5000, "sections": many},
    )

    assert result["data"]["sections_count"] == 30
    assert result["data"]["section_ids"] == [f"s{i}" for i in range(30)]
    plan = session.workspace["plan"]
    assert len(plan["title"]) == 300
    assert len(plan["abstract"]) == 2000
    assert all(len(s["title"]) == 200 for s in plan["sections"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "plan_document slices sections[:30] and reports sections_count=30 with "
        "no word that ten were dropped; the model then writes to 's35' and is "
        "told it is 'not found in document plan'"
    ),
)
async def test_plan_says_when_it_dropped_sections_over_the_cap():
    session = _Session()
    many = [{"id": f"s{i}", "title": f"Section {i}"} for i in range(40)]

    result = await session.call("plan_document", {"title": "R", "sections": many})

    assert "error" in result or "30" in json.dumps(
        {k: v for k, v in result.items() if k != "data"}
        | {
            k: v
            for k, v in result.get("data", {}).items()
            if k not in ("sections_count", "section_ids")
        }
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "plan_document skips every non-dict section and still reports success: "
        "sections=['Intro','Methods'] yields a plan with zero sections that "
        "nothing can be written into"
    ),
)
async def test_plan_refuses_when_no_section_is_usable():
    session = _Session()

    result = await session.call(
        "plan_document", {"title": "Report", "sections": ["Intro", "Methods"]}
    )

    _refused(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "plan_document accepts duplicate section ids; write_section matches the "
        "first, so the second section of that id can never be written and the "
        "assembled document carries '[Section not yet written]' for it"
    ),
)
async def test_plan_does_not_accept_two_sections_with_one_id():
    session = _Session()

    result = await session.call(
        "plan_document",
        {
            "title": "Report",
            "sections": [
                {"id": "s", "title": "Background"},
                {"id": "s", "title": "Evaluation"},
            ],
        },
    )

    ids = result.get("data", {}).get("section_ids", [])
    assert "error" in result or len(set(ids)) == 2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "plan_document stores the id unstripped while write_section strips the "
        "id it is given, so a section planned as ' intro ' is unreachable"
    ),
)
async def test_a_planned_section_id_with_padding_can_still_be_written():
    session = _Session()
    await session.call(
        "plan_document",
        {"title": "Report", "sections": [{"id": " intro ", "title": "Intro"}]},
    )
    planned_id = session.workspace["plan"]["sections"][0]["id"]

    result = await session.call(
        "write_section", {"section_id": planned_id, "content": "body"}
    )

    assert result.get("success") is True


async def test_a_new_plan_replaces_the_old_document_entirely():
    session = _Session()
    await session.plan()
    await session.write_all()
    await session.assembled()

    await session.call(
        "plan_document",
        {"title": "Second", "sections": [{"id": "only", "title": "Only"}]},
    )

    assert session.workspace["plan"]["title"] == "Second"
    assert session.workspace["assembled_markdown"] is None
    assert session.workspace["citations_registry"] == {}
    _refused(
        await session.call("write_section", {"section_id": "intro", "content": "x"})
    )
    _refused(await session.call("export_document", {"format": "docx"}))


async def test_the_workspace_survives_a_checkpoint_round_trip():
    session = _Session()
    await session.plan()
    await session.write_all()
    await session.call(
        "insert_figure",
        {"section_id": "results", "figure_type": "chart", "caption": "Speedup"},
    )

    restored = _Session()
    restored.state.update(json.loads(json.dumps(session.state)))

    markdown = await restored.assembled()
    assert BODY["methods"] in markdown


# --------------------------------------------------------------------------
# write_section
# --------------------------------------------------------------------------


async def test_writing_before_planning_is_refused():
    session = _Session()

    error = _refused(
        await session.call("write_section", {"section_id": "intro", "content": "x"})
    )

    assert "plan_document" in error


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"section_id": "intro"},
        {"content": "body"},
        {"section_id": "  ", "content": "body"},
        {"section_id": "intro", "content": ""},
    ],
)
async def test_write_refuses_without_section_and_content(params):
    session = _Session()
    await session.plan()

    _refused(await session.call("write_section", params))

    assert all(s["content"] is None for s in session.workspace["plan"]["sections"])


async def test_writing_to_a_section_the_plan_does_not_have_is_refused():
    session = _Session()
    await session.plan()

    error = _refused(
        await session.call(
            "write_section", {"section_id": "conclusion", "content": "The end."}
        )
    )

    assert "conclusion" in error
    assert all(s["content"] is None for s in session.workspace["plan"]["sections"])
    assert "The end." not in await session.assembled()


async def test_write_stores_the_content_on_the_named_section_only():
    session = _Session()
    await session.plan()

    result = await session.call(
        "write_section", {"section_id": "methods", "content": BODY["methods"]}
    )

    assert result["success"] is True
    assert result["data"] == {
        "section_id": "methods",
        "content_length": len(BODY["methods"]),
        "citations_count": 0,
    }
    contents = {s["id"]: s["content"] for s in session.workspace["plan"]["sections"]}
    assert contents == {"intro": None, "methods": BODY["methods"], "results": None}


async def test_write_keeps_long_content_whole():
    session = _Session()
    await session.plan()
    long_body = "\n\n".join(f"Paragraph {i} " + "word " * 200 for i in range(100))

    result = await session.call(
        "write_section", {"section_id": "intro", "content": long_body}
    )

    assert result["data"]["content_length"] == len(long_body)
    assert long_body in await session.assembled()


async def test_citations_are_registered_and_appear_in_the_references():
    session = _Session()
    await session.plan()

    result = await session.call(
        "write_section",
        {
            "section_id": "methods",
            "content": "We follow the stride design [2] and ISB [1].",
            "citations": [
                {"ref_id": "[2]", "document_id": "doc-2", "title": "Stride Paper"},
                {
                    "ref_id": "[1]",
                    "document_id": "doc-1",
                    "title": "ISB Paper",
                    "excerpt": "irregular streams",
                },
                {"document_id": "doc-3", "title": "No ref id"},
                "not a citation",
            ],
        },
    )

    assert result["data"]["citations_count"] == 2
    registry = session.workspace["citations_registry"]
    assert registry == {
        "[2]": {"document_id": "doc-2", "title": "Stride Paper", "excerpt": ""},
        "[1]": {
            "document_id": "doc-1",
            "title": "ISB Paper",
            "excerpt": "irregular streams",
        },
    }
    markdown = await session.assembled()
    assert "## References" in markdown
    assert _in_order(markdown, "## References", "ISB Paper", "Stride Paper")
    assert "No ref id" not in markdown


async def test_at_most_twenty_citations_are_kept_per_call():
    session = _Session()
    await session.plan()
    citations = [
        {"ref_id": f"r{i:02d}", "document_id": f"d{i}", "title": f"Paper {i}"}
        for i in range(25)
    ]

    result = await session.call(
        "write_section",
        {"section_id": "intro", "content": "body", "citations": citations},
    )

    assert result["data"]["citations_count"] == 20
    assert len(session.workspace["citations_registry"]) == 20


@pytest.mark.xfail(
    strict=True,
    reason=(
        "write_section appends citations and never clears them: writing the "
        "same section twice with the same citation reports citations_count=2, "
        "and a citation dropped by the rewrite stays in the References"
    ),
)
async def test_rewriting_a_section_replaces_its_citations():
    session = _Session()
    await session.plan()
    old = {"ref_id": "[1]", "document_id": "d1", "title": "Withdrawn Paper"}
    new = {"ref_id": "[2]", "document_id": "d2", "title": "Better Paper"}
    await session.call(
        "write_section",
        {"section_id": "intro", "content": "See [1].", "citations": [old]},
    )

    result = await session.call(
        "write_section",
        {"section_id": "intro", "content": "See [2].", "citations": [new]},
    )

    assert result["data"]["citations_count"] == 1
    assert "Withdrawn Paper" not in await session.assembled()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "write_section declares search_query ('query to search KB for relevant "
        "context before writing') and its description promises RAG context; "
        "the handler never reads the parameter and touches no service"
    ),
)
async def test_search_query_searches_the_knowledge_base():
    executor = _Recorder()
    session = _Session(executor=executor)
    await session.plan()

    result = await session.call(
        "write_section",
        {
            "section_id": "methods",
            "content": BODY["methods"],
            "search_query": "stride prefetcher L2",
        },
    )

    assert executor.touched or set(result.get("data", {})) - {
        "section_id",
        "content_length",
        "citations_count",
    }


# --------------------------------------------------------------------------
# revise_section
# --------------------------------------------------------------------------


async def test_revising_before_planning_is_refused():
    session = _Session()

    _refused(
        await session.call(
            "revise_section", {"section_id": "intro", "new_content": "x"}
        )
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"section_id": "intro"},
        {"new_content": "x"},
        {"section_id": "intro", "new_content": ""},
    ],
)
async def test_revise_refuses_without_section_and_new_content(params):
    session = _Session()
    await session.plan()
    await session.write_all()

    _refused(await session.call("revise_section", params))

    assert session.workspace["plan"]["sections"][0]["content"] == BODY["intro"]
    assert session.workspace["plan"]["sections"][0]["revision_count"] == 0


async def test_revising_an_unknown_section_is_refused():
    session = _Session()
    await session.plan()

    error = _refused(
        await session.call(
            "revise_section", {"section_id": "appendix", "new_content": "x"}
        )
    )

    assert "appendix" in error


async def test_a_revision_replaces_the_text_in_the_final_document():
    session = _Session()
    await session.plan()
    await session.write_all()

    first = await session.call(
        "revise_section",
        {
            "section_id": "results",
            "feedback": "State the worst case too.",
            "new_content": "Geomean 2.1x; worst kernel 1.02x.",
            "additional_citations": [
                {"ref_id": "[9]", "document_id": "d9", "title": "Kernel Suite"}
            ],
        },
    )
    second = await session.call(
        "revise_section",
        {"section_id": "results", "new_content": "Geomean 2.1x; worst 1.02x; n=4."},
    )

    assert first["data"]["revision_count"] == 1
    assert second["data"] == {
        "section_id": "results",
        "revision_count": 2,
        "content_length": len("Geomean 2.1x; worst 1.02x; n=4."),
    }
    markdown = await session.assembled()
    assert "Geomean 2.1x; worst 1.02x; n=4." in markdown
    assert BODY["results"] not in markdown
    assert "worst kernel 1.02x." not in markdown
    assert BODY["intro"] in markdown and BODY["methods"] in markdown
    assert "Kernel Suite" in markdown


@pytest.mark.xfail(
    strict=True,
    reason=(
        "assemble_document stores a snapshot and revise_section does not "
        "invalidate it, so export_document after a revision exports the text "
        "from before the revision and reports success"
    ),
)
async def test_export_after_a_revision_does_not_ship_the_old_text(monkeypatch):
    from app.services.docx_builder import DOCXBuilder

    calls = _spy_on_build(monkeypatch, DOCXBuilder)
    session = _Session()
    await session.plan()
    await session.write_all()
    await session.assembled()
    await session.call(
        "revise_section",
        {"section_id": "results", "new_content": "Corrected: the geomean is 1.4x."},
    )

    result = await session.call("export_document", {"format": "docx"})

    if "error" in result:
        return
    text = _docx_text(calls[-1]["bytes"])
    assert "Corrected: the geomean is 1.4x." in text
    assert BODY["results"] not in text


# --------------------------------------------------------------------------
# insert_figure
# --------------------------------------------------------------------------


async def test_figures_need_a_plan():
    session = _Session()

    _refused(
        await session.call(
            "insert_figure",
            {"section_id": "results", "figure_type": "chart", "caption": "c"},
        )
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"figure_type": "chart", "caption": "c"},
        {"section_id": "results", "caption": "c"},
    ],
)
async def test_figure_refuses_without_section_and_type(params):
    session = _Session()
    await session.plan()

    _refused(await session.call("insert_figure", params))

    assert all(s["figures"] == [] for s in session.workspace["plan"]["sections"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the spec requires caption, but insert_figure only checks section_id "
        "and figure_type; a figure with no caption is inserted as '*[Figure: ]*'"
    ),
)
async def test_figure_refuses_without_a_caption():
    session = _Session()
    await session.plan()
    await session.write_all()

    result = await session.call(
        "insert_figure", {"section_id": "results", "figure_type": "chart"}
    )

    _refused(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the spec's figure_type enum is chart/table/diagram/flowchart; "
        "insert_figure accepts any string"
    ),
)
async def test_figure_refuses_a_type_outside_the_declared_set():
    session = _Session()
    await session.plan()
    await session.write_all()

    result = await session.call(
        "insert_figure",
        {"section_id": "results", "figure_type": "hologram", "caption": "c"},
    )

    _refused(result)


async def test_figure_for_an_unknown_section_is_refused():
    session = _Session()
    await session.plan()

    error = _refused(
        await session.call(
            "insert_figure",
            {"section_id": "appendix", "figure_type": "chart", "caption": "c"},
        )
    )

    assert "appendix" in error


async def test_a_figure_in_a_written_section_reaches_the_document():
    session = _Session()
    await session.plan()
    await session.write_all()

    first = await session.call(
        "insert_figure",
        {
            "section_id": "results",
            "figure_type": "chart",
            "caption": "Speedup per kernel",
            "data": {"labels": ["a", "b"], "values": [1.0, 2.1]},
        },
    )
    second = await session.call(
        "insert_figure",
        {
            "section_id": "results",
            "figure_type": "diagram",
            "caption": "C" * 500,
            "diagram_spec": "graph TD; A-->B",
        },
    )

    assert first["data"] == {
        "section_id": "results",
        "figure_type": "chart",
        "figures_count": 1,
    }
    assert second["data"]["figures_count"] == 2
    figures = session.workspace["plan"]["sections"][2]["figures"]
    assert figures[0]["type"] == "chart"
    assert figures[0]["data"] == {"labels": ["a", "b"], "values": [1.0, 2.1]}
    assert figures[1]["diagram_spec"] == "graph TD; A-->B"
    assert len(figures[1]["caption"]) == 300
    markdown = await session.assembled()
    assert _in_order(markdown, BODY["results"], "Speedup per kernel")
    assert _in_order(markdown, "## Methods", "## Results", "Speedup per kernel")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "insert_figure only appends its marker to content that already exists "
        "and assemble_document never reads section['figures']: a figure added "
        "before the section is written is reported inserted (figures_count=1) "
        "and is absent from the document"
    ),
)
async def test_a_figure_inserted_before_the_section_is_written_is_not_lost():
    session = _Session()
    await session.plan()

    result = await session.call(
        "insert_figure",
        {
            "section_id": "results",
            "figure_type": "chart",
            "caption": "Speedup per kernel",
        },
    )
    assert result["success"] is True
    await session.write_all()

    assert "Speedup per kernel" in await session.assembled()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the figure lives only as a marker appended to section content, so "
        "revise_section (which replaces the content) silently removes every "
        "figure in the section while section['figures'] still lists it"
    ),
)
async def test_a_figure_survives_a_revision_of_its_section():
    session = _Session()
    await session.plan()
    await session.write_all()
    await session.call(
        "insert_figure",
        {
            "section_id": "results",
            "figure_type": "chart",
            "caption": "Speedup per kernel",
        },
    )

    await session.call(
        "revise_section", {"section_id": "results", "new_content": "Tighter text."}
    )

    assert "Speedup per kernel" in await session.assembled()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "insert_figure stores data and diagram_spec and nothing renders them: "
        "a table's cells and a diagram's source never reach the assembled "
        "document, which gets only '*[Figure: <caption>]*'"
    ),
)
async def test_a_table_figure_puts_its_data_in_the_document():
    session = _Session()
    await session.plan()
    await session.write_all()

    await session.call(
        "insert_figure",
        {
            "section_id": "results",
            "figure_type": "table",
            "caption": "Per-kernel speedup",
            "data": {
                "headers": ["kernel", "speedup"],
                "rows": [["pointer_chase", "1.02"], ["stream_add", "3.87"]],
            },
        },
    )

    markdown = await session.assembled()
    assert "pointer_chase" in markdown
    assert "3.87" in markdown


# --------------------------------------------------------------------------
# assemble_document
# --------------------------------------------------------------------------


async def test_assembling_before_planning_is_refused():
    session = _Session()

    _refused(await session.call("assemble_document"))


async def test_every_written_section_is_in_the_document_in_plan_order():
    session = _Session()
    await session.plan()
    # Written out of order on purpose: the plan decides the order.
    for section_id in ("results", "intro", "methods"):
        await session.call(
            "write_section", {"section_id": section_id, "content": BODY[section_id]}
        )

    result = await session.call("assemble_document")

    assert result["success"] is True
    markdown = session.markdown
    assert result["data"] == {
        "total_sections": 3,
        "sections_written": 3,
        "sections_skipped": 0,
        "total_length": len(markdown),
        "citations_count": 0,
    }
    assert markdown.startswith("# Prefetcher Study\n")
    assert _in_order(
        markdown,
        "# Prefetcher Study",
        "## Abstract",
        "What a prefetcher buys on four kernels.",
        "## Table of Contents",
        "## Introduction\n",
        BODY["intro"],
        "## Methods\n",
        BODY["methods"],
        "## Results\n",
        BODY["results"],
    )
    for body in BODY.values():
        assert markdown.count(body) == 1
    assert "not yet written" not in markdown
    assert "## References" not in markdown


async def test_the_table_of_contents_lists_every_section_in_order():
    session = _Session()
    await session.plan()
    await session.write_all()

    markdown = await session.assembled()

    toc = markdown.split("## Table of Contents")[1].split("---")[0]
    assert _in_order(toc, "1. ", "Introduction", "2. ", "Methods", "3. ", "Results")


async def test_optional_parts_can_be_left_out():
    session = _Session()
    await session.plan()
    await session.call(
        "write_section",
        {
            "section_id": "intro",
            "content": BODY["intro"],
            "citations": [{"ref_id": "[1]", "document_id": "d", "title": "Cited"}],
        },
    )

    markdown = await session.assembled(
        include_toc=False, include_references=False, include_abstract=False
    )

    assert "Table of Contents" not in markdown
    assert "## References" not in markdown
    assert "## Abstract" not in markdown
    assert "What a prefetcher buys" not in markdown
    assert BODY["intro"] in markdown


async def test_a_custom_order_reorders_sections_and_the_contents_list():
    session = _Session()
    await session.plan()
    await session.write_all()

    markdown = await session.assembled(section_order=["results", "intro", "methods"])

    assert _in_order(markdown, BODY["results"], BODY["intro"], BODY["methods"])
    toc = markdown.split("## Table of Contents")[1].split("---")[0]
    assert _in_order(toc, "Results", "Introduction", "Methods")


async def test_a_partial_order_keeps_every_section():
    session = _Session()
    await session.plan()
    await session.write_all()

    result = await session.call(
        "assemble_document", {"section_order": ["results", "no_such_section"]}
    )

    markdown = session.markdown
    assert result["data"]["sections_written"] == 3
    assert _in_order(markdown, BODY["results"], BODY["intro"], BODY["methods"])


async def test_unwritten_sections_are_counted_and_visibly_marked():
    session = _Session()
    await session.plan()
    await session.call(
        "write_section", {"section_id": "methods", "content": BODY["methods"]}
    )

    result = await session.call("assemble_document")

    assert result["data"]["sections_written"] == 1
    assert result["data"]["sections_skipped"] == 2
    markdown = session.markdown
    assert BODY["methods"] in markdown
    assert markdown.count("not yet written") == 2


# --------------------------------------------------------------------------
# export_document
# --------------------------------------------------------------------------


def _docx_text(data):
    import docx

    document = docx.Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.extend(cell.text for cell in row.cells)
    return "\n".join(lines)


async def _ready(db=None, user=None):
    session = _Session(db=db, user=user)
    await session.plan()
    await session.write_all()
    await session.assembled()
    return session


async def test_exporting_needs_an_assembled_document():
    unplanned = _Session()
    _refused(await unplanned.call("export_document", {"format": "docx"}))

    session = _Session()
    await session.plan()
    await session.write_all()
    error = _refused(await session.call("export_document", {"format": "docx"}))

    assert "assemble_document" in error
    assert session.workspace["export_artifacts"] == []


@pytest.mark.parametrize("fmt", [None, "", "markdown", "html", "txt"])
async def test_export_refuses_a_format_it_does_not_declare(fmt):
    session = await _ready()

    params = {} if fmt is None else {"format": fmt}
    _refused(await session.call("export_document", params))

    assert session.workspace["export_artifacts"] == []


async def test_export_refuses_a_document_over_the_size_limit():
    session = _Session()
    await session.plan()
    await session.call(
        "write_section", {"section_id": "intro", "content": "x" * 500_001}
    )
    await session.assembled()

    error = _refused(await session.call("export_document", {"format": "docx"}))

    assert "large" in error.lower()
    assert session.workspace["export_artifacts"] == []


async def test_docx_export_contains_every_section_in_order(monkeypatch):
    from app.services.docx_builder import DOCXBuilder

    calls = _spy_on_build(monkeypatch, DOCXBuilder)
    session = await _ready()

    result = await session.call("export_document", {"format": "DOCX"})

    assert result["success"] is True
    artifact = result["data"]
    assert artifact["format"] == "docx"
    assert artifact["title"] == "Prefetcher Study"
    assert artifact["mime_type"].endswith("wordprocessingml.document")
    assert len(calls) == 1
    data = calls[0]["bytes"]
    assert data[:2] == b"PK"
    assert artifact["size_bytes"] == len(data)
    text = _docx_text(data)
    assert "Prefetcher Study" in text
    assert _in_order(
        text,
        "Introduction",
        BODY["intro"],
        BODY["methods"],
        BODY["results"],
    )
    assert session.workspace["export_artifacts"] == [artifact]


async def test_pdf_export_builds_a_pdf(monkeypatch):
    from app.services.pdf_builder import PDFBuilder

    calls = _spy_on_build(monkeypatch, PDFBuilder)
    session = await _ready()

    result = await session.call("export_document", {"format": "pdf"})

    assert result["success"] is True
    assert result["data"]["mime_type"] == "application/pdf"
    assert calls[0]["bytes"][:5] == b"%PDF-"
    assert result["data"]["size_bytes"] == len(calls[0]["bytes"])
    items = calls[0]["kwargs"]["content_items"]
    flattened = json.dumps(items)
    assert _in_order(flattened, BODY["intro"], BODY["methods"], BODY["results"])


async def test_pptx_export_has_a_slide_per_section_in_order(monkeypatch):
    from app.services.pptx_builder import PPTXBuilder

    calls = _record_pptx(monkeypatch, PPTXBuilder)
    session = await _ready()

    result = await session.call("export_document", {"format": "pptx"})

    assert result["success"] is True, result
    assert result["data"]["mime_type"].endswith("presentationml.presentation")
    outline = calls[0]["kwargs"]["outline"]
    assert outline.title == "Prefetcher Study"
    titles = [slide.title for slide in outline.slides]
    assert [t for t in titles if t in ("Introduction", "Methods", "Results")] == [
        "Introduction",
        "Methods",
        "Results",
    ]
    by_title = {slide.title: slide.content for slide in outline.slides}
    assert BODY["intro"] in by_title["Introduction"]
    assert BODY["methods"] in by_title["Methods"]
    assert BODY["results"] in by_title["Results"]
    assert [slide.slide_number for slide in outline.slides] == list(
        range(1, len(outline.slides) + 1)
    )
    assert calls[0]["bytes"][:2] == b"PK"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the pptx branch turns a section into one slide and keeps bullets[:10]: "
        "every line of a section past its tenth is dropped from the deck "
        "without a word, and the export reports success"
    ),
)
async def test_pptx_export_keeps_all_of_a_long_section(monkeypatch):
    from app.services.pptx_builder import PPTXBuilder

    calls = _record_pptx(monkeypatch, PPTXBuilder)
    session = _Session()
    await session.plan()
    lines = [f"- finding number {i}" for i in range(1, 16)]
    await session.call(
        "write_section", {"section_id": "results", "content": "\n".join(lines)}
    )
    await session.assembled()

    result = await session.call("export_document", {"format": "pptx"})

    assert result["success"] is True, result
    shown = json.dumps(
        [slide.content for slide in calls[0]["kwargs"]["outline"].slides]
    )
    assert "finding number 15" in shown


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the latex branch calls LatexCompilerService.compile_to_pdf on the "
        "class; it is an instance method, so every latex export fails with "
        "\"missing 1 required positional argument: 'self'\""
    ),
)
async def test_latex_export_reaches_the_compiler(monkeypatch):
    from app.services import latex_compiler_service as module

    seen = []

    def compile_to_pdf(self, *, tex_source, timeout_seconds, max_source_chars, **kw):
        seen.append(tex_source)
        return module.LatexCompileResult(
            success=True,
            engine="tectonic",
            pdf_bytes=b"%PDF-1.5",
            log="",
            violations=[],
        )

    monkeypatch.setattr(module.LatexCompilerService, "compile_to_pdf", compile_to_pdf)
    session = await _ready()

    result = await session.call("export_document", {"format": "latex"})

    assert result.get("success") is True, result
    assert BODY["methods"] in seen[0]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the latex branch hands the assembled *markdown* to the LaTeX compiler "
        "as tex_source: no \\documentclass, '# Title' headings and '---' "
        "rules, so even with the call fixed nothing compilable is produced"
    ),
)
async def test_latex_export_compiles_latex_not_markdown(monkeypatch):
    from app.services import latex_compiler_service as module

    seen = []

    def compile_to_pdf(*args, tex_source, **kw):
        seen.append(tex_source)
        return module.LatexCompileResult(
            success=True,
            engine="tectonic",
            pdf_bytes=b"%PDF-1.5",
            log="",
            violations=[],
        )

    # A staticmethod, so the handler's class-level call reaches it and this
    # test is about what is sent rather than about how it is called.
    monkeypatch.setattr(
        module.LatexCompilerService, "compile_to_pdf", staticmethod(compile_to_pdf)
    )
    session = await _ready()

    await session.call("export_document", {"format": "latex"})

    assert seen, "compiler was not called"
    assert "\\documentclass" in seen[0]
    assert "## Introduction" not in seen[0]


async def test_a_failed_latex_compile_is_an_error_not_an_artifact(monkeypatch):
    from app.services import latex_compiler_service as module

    def compile_to_pdf(*args, **kw):
        return module.LatexCompileResult(
            success=False,
            engine=None,
            pdf_bytes=None,
            log="! Undefined control sequence.",
            violations=[],
        )

    monkeypatch.setattr(
        module.LatexCompilerService, "compile_to_pdf", staticmethod(compile_to_pdf)
    )
    session = await _ready()

    error = _refused(await session.call("export_document", {"format": "latex"}))

    assert "Undefined control sequence" in error
    assert session.workspace["export_artifacts"] == []


async def test_a_builder_failure_is_reported_and_records_no_artifact(monkeypatch):
    from app.services.docx_builder import DOCXBuilder

    def build(self, *args, **kwargs):
        raise RuntimeError("template missing")

    monkeypatch.setattr(DOCXBuilder, "build", build)
    session = await _ready()

    error = _refused(await session.call("export_document", {"format": "docx"}))

    assert "template missing" in error
    assert session.workspace["export_artifacts"] == []


async def test_export_without_persist_writes_no_document_row(db_session, test_user):
    session = await _ready(db_session, test_user)

    result = await session.call("export_document", {"format": "docx"})

    assert result["success"] is True
    assert "document_id" not in result["data"]
    assert (await db_session.execute(select(Document))).scalars().all() == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "persist_to_kb builds Document without source_id or source_identifier "
        "(both NOT NULL), so the flush raises IntegrityError; the handler "
        "swallows it and returns success with no document_id -- nothing is "
        "ever saved to the knowledge base"
    ),
)
async def test_persist_to_kb_stores_the_whole_document(db_session, test_user):
    session = await _ready(db_session, test_user)

    result = await session.call(
        "export_document", {"format": "docx", "persist_to_kb": True}
    )

    assert result.get("success") is True, result
    document_id = result["data"]["document_id"]
    await db_session.commit()
    row = (
        await db_session.execute(
            select(Document).where(Document.id == UUID(document_id))
        )
    ).scalar_one()
    assert "Prefetcher Study" in row.title
    assert _in_order(row.content, BODY["intro"], BODY["methods"], BODY["results"])
    assert row.source_id is not None
    assert row.extra_metadata["job_id"] == str(session.job.id)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "when the persist_to_kb insert fails the handler reports plain success: "
        "the caller asked for the document to be saved and is not told it was "
        "not (no error, no warning, no document_id)"
    ),
)
async def test_a_failed_persist_is_not_reported_as_plain_success(test_user):
    class _BrokenDb:
        def add(self, obj):
            pass

        async def flush(self):
            raise RuntimeError("database unavailable")

    session = await _ready(_BrokenDb(), test_user)

    result = await session.call(
        "export_document", {"format": "docx", "persist_to_kb": True}
    )

    said_so = "error" in result or any(
        key in result.get("data", {})
        for key in ("warning", "warnings", "persist_error", "persisted")
    )
    assert said_so


@pytest.mark.xfail(
    strict=True,
    reason=(
        "export_document builds the file, records len(file_bytes) and discards "
        "the bytes: nothing is written to object storage or the workspace, so "
        "the artifact names a DOCX nobody can download"
    ),
)
async def test_the_exported_file_can_be_found_afterwards(db_session, test_user):
    session = await _ready(db_session, test_user)

    result = await session.call("export_document", {"format": "docx"})

    artifact = result["data"]
    locators = set(artifact) - {"type", "format", "title", "size_bytes", "mime_type"}
    assert locators, "artifact carries nothing that locates the exported file"
