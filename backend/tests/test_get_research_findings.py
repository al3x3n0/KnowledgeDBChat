"""`get_research_findings` reads the run's findings, not only the ones saved by hand.

Found live: a self-planned pipeline's report stage held the two benchmark
measurements its parent stage made, was told "no findings" by this tool four
times, and blocked rather than write a report on nothing.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.agent_tool_providers.research import (
    build_autonomous_research_provider,
)

pytestmark = pytest.mark.unit

MEASURED = {
    "type": "benchmark_measurement",
    "subject": "fastmod",
    "fastest_ms": 68,
    "inherited": True,
    "inherited_from_job_id": "parent",
}


def tools(executor):
    handlers = build_autonomous_research_provider(executor)._handlers
    return handlers["get_research_findings"], handlers["save_research_finding"]


def context(findings):
    job = SimpleNamespace(id=uuid4(), user_id=None)
    return SimpleNamespace(job=job, state={"findings": findings}, db=None, user_id=None)


@pytest.fixture
def executor():
    return SimpleNamespace(_job_findings={})


async def test_a_stage_sees_what_it_inherited(executor):
    get, _ = tools(executor)

    result = await get({}, context([dict(MEASURED)]))

    assert result["data"]["total"] == 1
    assert result["data"]["by_type"] == {"benchmark_measurement": 1}
    assert result["data"]["inherited"] == 1


async def test_a_saved_finding_is_listed_once(executor):
    get, save = tools(executor)
    ctx = context([dict(MEASURED)])
    saved = await save(
        {"title": "fastmod is 2x", "content": "c", "category": "key_insight"}, ctx
    )
    # As the executor does with every tool's findings.
    ctx.state["findings"].extend(saved["findings"])

    result = await get({}, ctx)

    assert result["data"]["total"] == 2


async def test_after_a_restart_the_run_s_findings_are_still_there(executor):
    """The saved list is in memory; the run's findings are restored from state."""
    get, _ = tools(executor)
    restarted = SimpleNamespace(_job_findings={})

    result = await tools(restarted)[0]({}, context([dict(MEASURED)]))

    assert result["data"]["total"] == 1


async def test_filters_by_type_and_tags(executor):
    get, _ = tools(executor)
    ctx = context(
        [
            dict(MEASURED),
            {"type": "implementation_verified", "verified": True, "tags": ["c"]},
            {"title": "note", "category": "key_insight", "tags": ["c", "fast"]},
        ]
    )

    by_type = await get({"finding_type": "benchmark_measurement"}, ctx)
    by_tag = await get({"tags": ["fast"]}, ctx)

    assert by_type["data"]["total"] == 1
    assert [f.get("title") for f in by_tag["data"]["findings"]] == ["note"]


async def test_a_confidence_floor_does_not_hide_a_measurement(executor):
    get, _ = tools(executor)
    ctx = context(
        [dict(MEASURED), {"title": "guess", "confidence": 0.2, "category": "x"}]
    )

    result = await get({"min_confidence": 0.5}, ctx)

    assert [f.get("type") for f in result["data"]["findings"]] == [
        "benchmark_measurement"
    ]


async def test_a_nonsense_limit_is_bounded(executor):
    get, _ = tools(executor)

    three = [dict(MEASURED, fastest_ms=ms) for ms in (68, 70, 71)]

    result = await get({"limit": "-1"}, context(three))

    # -1 is "no limit" to a slice that drops the tail; it is clamped to 1,
    # and `total` still says how many there are.
    assert result["data"]["total"] == 3 and len(result["data"]["findings"]) == 1
