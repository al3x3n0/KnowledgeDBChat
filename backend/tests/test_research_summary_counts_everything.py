"""The run summary must reflect what the run actually produced.

It counted documents, papers and insights only. A compiler experiment that
recorded nine codegen measurements summarised itself as "0 KB docs, 0 papers,
0 saved insights" — a run that had measured a great deal reporting that it had
found nothing.
"""

import pytest

from app.services.agent_runtime_finalizer import finalize_job
from tests.test_agent_runtime_finalizer import (  # reuse the harness
    _DummyDb,
    _DummyExecutor,
    _DummyJob,
)


def _state(findings):
    return {
        "goal_progress": 100,
        "findings": findings,
        "actions_taken": [],
        "artifacts": [],
        "execution_plan": [],
        "step_events": [],
        "tool_stats": {},
        "tool_priors": {},
        "execution_graph_nodes": [],
        "execution_graph_edges": [],
        "scope_events": [],
        "scope_guard_events": [],
        "skill_profile": {},
        "skill_profile_metrics": {},
        "memory_runtime": {},
        "memory_extraction_policy": {"extract_on_statuses": []},
        "memory_extraction": {},
        "injected_memories": [],
    }


async def _run(findings):
    executor = _DummyExecutor()
    job = _DummyJob(status="running")
    job.job_type = "research"
    await finalize_job(executor, job, _state(findings), _DummyDb())
    return job


@pytest.mark.asyncio
async def test_measurements_are_counted_not_ignored():
    findings = [
        {"type": "codegen_measurement", "title": f"kernel {i} @ -O3"} for i in range(9)
    ]

    job = await _run(findings)

    research = job.results["research"]
    assert research["other_findings"] == 9
    assert research["other_findings_by_type"] == {"codegen_measurement": 9}
    assert "9 codegen measurement" in job.results["summary"]
    assert "0 KB docs" not in job.results["summary"]


@pytest.mark.asyncio
async def test_documents_and_papers_still_reported():
    findings = [
        {"type": "document", "title": "A doc"},
        {"type": "paper", "title": "A paper"},
    ]

    job = await _run(findings)

    assert "1 KB docs" in job.results["summary"]
    assert "1 papers" in job.results["summary"]
    assert job.results["research"]["other_findings"] == 0


@pytest.mark.asyncio
async def test_mixed_findings_are_all_mentioned():
    findings = [
        {"type": "document", "title": "A doc"},
        {"type": "codegen_measurement", "title": "kernel @ -O3"},
        {"type": "benchmark_measurement", "title": "kernel @ -O2"},
    ]

    job = await _run(findings)

    summary = job.results["summary"]
    assert "1 KB docs" in summary
    assert "1 codegen measurement" in summary
    assert "1 benchmark measurement" in summary


@pytest.mark.asyncio
async def test_a_run_with_nothing_says_so_plainly():
    job = await _run([])

    assert (
        job.results["summary"] == "Research run completed: no findings were recorded."
    )
