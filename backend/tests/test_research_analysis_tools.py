"""The research tools that used to fabricate success now do real work.

Eight dispatch handlers returned success-shaped payloads while doing nothing:
link_entities reported "relationship_created": True from a function that created
no relationship, compare_methodologies returned the literal string "Comparison
would be generated here" as its result, create_knowledge_base_entry claimed
"entry_created": True and emitted an artifact it never persisted.

Every one produced a *conclusion* rather than fetching data, which is why the
fabrication was hard to notice — nothing downstream could contradict them. These
tests check the two properties that matter now: no tool reports success from a
hardcoded literal, and each refuses honestly when it cannot do the work.
"""

import inspect

import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_kg_provider,
    build_autonomous_research_provider,
)

FORMERLY_FABRICATED = [
    "link_entities",
    "build_research_graph",
    "compare_methodologies",
    "identify_research_gaps",
    "create_knowledge_base_entry",
    "generate_research_presentation",
    "analyze_document_cluster",
    "compare_documents",
]


class _RecordingLLM:
    """Stands in for the provider and records that the tool actually asked."""

    def __init__(self, structured=None):
        self._structured = structured
        self.calls = 0

    async def generate_structured(self, **kwargs):
        self.calls += 1
        completion = type("Completion", (), {})()
        completion.structured = self._structured
        completion.text = ""
        return completion

    async def generate_response(self, **kwargs):
        self.calls += 1
        return ""


class _Executor:
    def __init__(self, structured=None):
        self.llm_service = _RecordingLLM(structured)
        self._job_findings: dict = {}


def _context(db=None, user_id=None, job=None) -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        mode="autonomous", db=db, service=None, user_id=user_id, job=job, state={}
    )


def _handler(tool: str, executor=None):
    """Find a tool's handler across the providers that define these tools.

    They are split between the kg and research providers, so a test that
    assumed one of them silently skipped half the surface.
    """
    executor = executor or _Executor()
    for build in (build_autonomous_kg_provider, build_autonomous_research_provider):
        provider = build(executor=executor)
        if tool in provider.supported_tools:
            return provider._handlers[tool]
    raise AssertionError(f"no provider defines {tool}")


@pytest.mark.parametrize("tool", FORMERLY_FABRICATED)
def test_no_tool_reports_success_from_a_literal(tool):
    if tool == "compare_documents":
        pytest.skip("defined in two providers; covered by its own test")

    source = inspect.getsource(_handler(tool))
    assert "_unimplemented_tool" not in source, f"{tool} is still a stub"
    if '"success": True' in source:
        assert (
            "await " in source
        ), f"{tool} claims success without awaiting any real work"


@pytest.mark.asyncio
async def test_link_entities_requires_a_relationship_type():
    result = await _handler("link_entities")({}, _context())
    assert "relationship_type" in result.get("error", "")


@pytest.mark.asyncio
async def test_link_entities_refuses_when_neither_side_is_identified():
    result = await _handler("link_entities")({"relationship_type": "uses"}, _context())
    assert result.get("error"), "must not link entities it cannot identify"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    ["analyze_document_cluster", "compare_methodologies", "build_research_graph"],
)
async def test_document_tools_reject_an_empty_document_list(tool):
    result = await _handler(tool)({"document_ids": []}, _context())
    assert "document_ids" in result.get("error", "")


@pytest.mark.asyncio
async def test_identify_research_gaps_refuses_without_evidence():
    """A gap analysis over nothing is exactly the fabrication this replaced."""
    job = type("Job", (), {"id": "job-1", "goal": "find gaps"})()
    result = await _handler("identify_research_gaps")({}, _context(job=job))

    assert result.get("error")
    assert "save_research_finding" in result["error"]


@pytest.mark.asyncio
async def test_knowledge_base_entry_requires_content_and_a_user():
    handler = _handler("create_knowledge_base_entry")

    missing_content = await handler({"title": "t"}, _context())
    assert "content" in missing_content.get("error", "")

    no_user = await handler({"title": "t", "content": "c"}, _context())
    assert "user" in no_user.get("error", "")


@pytest.mark.asyncio
async def test_presentation_requires_title_topic_and_user():
    handler = _handler("generate_research_presentation")

    missing = await handler({"title": "t"}, _context())
    assert missing.get("error")

    no_user = await handler({"title": "t", "topic": "x"}, _context())
    assert "user" in no_user.get("error", "")


@pytest.mark.asyncio
async def test_analysis_reports_failure_when_the_model_returns_nothing():
    """An empty model reply must surface as an error, not an empty success."""
    executor = _Executor(structured=None)
    executor._job_findings["job-1"] = [{"title": "a finding"}]
    job = type("Job", (), {"id": "job-1", "goal": "g"})()

    result = await _handler("identify_research_gaps", executor)(
        {"topic": "t"}, _context(job=job)
    )

    assert result.get("error")
    assert result.get("success") is not True
    assert executor.llm_service.calls > 0, "the tool must actually ask the model"
