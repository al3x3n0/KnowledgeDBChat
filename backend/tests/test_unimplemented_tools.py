"""Tools that are advertised but unimplemented must fail, not fake success.

Eight dispatch handlers returned a success-shaped payload while doing nothing:
link_entities reported "relationship_created": True from a function that created
no relationship, compare_methodologies returned the literal string "Comparison
would be generated here" as its result, create_knowledge_base_entry claimed
"entry_created": True and emitted an artifact it never persisted.

That is worse than a missing feature. The agent recorded the fiction as a
finding, and the evidence layer this system is built around inherited it.
"""

import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    _unimplemented_tool,
    build_autonomous_kg_provider,
)

UNIMPLEMENTED = [
    "link_entities",
    "build_research_graph",
    "compare_methodologies",
    "identify_research_gaps",
    "create_knowledge_base_entry",
    "generate_research_presentation",
    "analyze_document_cluster",
    "compare_documents",
]


def test_the_marker_reports_failure_in_the_shape_the_loop_reads():
    result = _unimplemented_tool("some_tool")

    # success is derived as `not result.get("error")`, so an error key is what
    # makes the loop treat this as a failed call.
    assert result.get("error")
    assert "some_tool" in result["error"]
    assert result.get("unimplemented") is True
    assert result.get("success") is not True


def test_the_message_tells_the_agent_not_to_retry():
    # A retry loop against a permanently missing capability burns the whole
    # iteration budget, so the text has to say the failure is terminal.
    message = _unimplemented_tool("x")["error"]
    assert "not implemented" in message
    assert "do not retry" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", UNIMPLEMENTED)
async def test_unimplemented_handlers_no_longer_claim_success(tool):
    """Invoke each tool for real and assert it reports failure."""
    provider = build_autonomous_kg_provider(executor=None)
    if tool not in provider.supported_tools:
        pytest.skip(f"{tool} is provided elsewhere")

    result = await provider._handlers[tool]({}, _context())

    assert result.get("error"), f"{tool} did not report an error"
    assert result.get("unimplemented") is True
    assert result.get("success") is not True, f"{tool} still claims success"


def _context() -> AgentToolExecutionContext:
    return AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, job=None, state={}
    )
