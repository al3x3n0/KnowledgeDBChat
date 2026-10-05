"""create_synthesis_document must record the evidence its spec promises.

A goal contract counts *finding* types. This tool declares
``produces=("synthesis_document",)`` and recorded only an artifact typed
``document``, so a contract requiring synthesis_document could not be satisfied
by the one tool meant to satisfy it. Measured on a live pipeline: the writeup
stage called it once, the call succeeded, and the stage still ended
`completed_contract_unmet` on finding_type:synthesis_document — and across 364
jobs no finding of that type had ever been recorded.

This is a third way a contract can be unsatisfiable, and the one the other two
guards cannot see: the tool is reachable and the guidance names it correctly,
but the evidence never arrives under the name the contract counts.
"""

import pytest

from app.agent_core import tool_specs
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_research_provider,
)


class _Source:
    id = "source-1"


class _DB:
    def add(self, _obj):
        return None

    async def commit(self):
        return None

    async def refresh(self, _obj):
        return None


class _DocumentService:
    async def reprocess_document(self, *_args, **_kwargs):
        return None

    async def _get_or_create_agent_notes_source(self, _db):
        return _Source()


class _Job:
    id = "job-1"
    user_id = "user-1"
    job_type = "research"
    name = "writeup"
    goal = "Write it up"
    config = {}
    results = {}


class _Executor:
    document_service = _DocumentService()

    def __init__(self):
        self._job_findings = {
            "job-1": [{"title": "A finding", "content": "body", "category": "x"}]
        }


def test_the_spec_promises_synthesis_document():
    spec = tool_specs.STATIC_CATALOG.spec_for("create_synthesis_document")
    assert "synthesis_document" in (spec.produces or ())


@pytest.mark.asyncio
async def test_it_records_a_finding_of_the_type_the_contract_counts():
    # The real Document model is constructed, which needs no database until it
    # is flushed; the stub session makes add/commit/refresh no-ops. Note the
    # handler imports Document *inside* the function, so patching the module
    # attribute would do nothing.
    provider = build_autonomous_research_provider(_Executor())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=_DB(),
        service=None,
        user_id="user-1",
        job=_Job(),
        state={},
    )
    result = await provider.execute(
        "create_synthesis_document", {"topic": "attention"}, ctx
    )

    findings = (result or {}).get("findings") or []
    assert [f.get("type") for f in findings] == [
        "synthesis_document"
    ], "the contract counts finding types; an artifact alone cannot satisfy it"


@pytest.mark.asyncio
async def test_a_persisted_synthesis_points_at_the_document_it_wrote():
    # So a later stage can read what this one produced, rather than only being
    # told that something was produced.
    provider = build_autonomous_research_provider(_Executor())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=_DB(),
        service=None,
        user_id="user-1",
        job=_Job(),
        state={},
    )
    result = await provider.execute(
        "create_synthesis_document",
        {"topic": "attention", "persist": True, "title": "Attention synthesis"},
        ctx,
    )

    (finding,) = result["findings"]
    assert finding["type"] == "synthesis_document"
    assert "document_id" in finding


@pytest.mark.asyncio
async def test_the_artifact_is_still_produced():
    # The artifact is what a person downloads; the finding is what the contract
    # counts. Adding the second must not cost the first.
    provider = build_autonomous_research_provider(_Executor())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=_DB(),
        service=None,
        user_id="user-1",
        job=_Job(),
        state={},
    )
    result = await provider.execute(
        "create_synthesis_document", {"topic": "attention"}, ctx
    )

    assert any(a.get("type") == "synthesis_document" for a in result["artifacts"])
