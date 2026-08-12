"""A failing tool must report its failure.

Substituting a different tool and rewriting the result to success turns "this
tool is broken" into "the agent got an answer". A broken arXiv search became a
knowledge-base search whose unrelated documents were recorded as research
findings, and nothing in the run said otherwise.
"""

import pytest

from app.services.agent_action_service import AgentActionService


class _FailingRegistry:
    """Every tool fails the way the real arXiv tool did."""

    async def try_execute(self, *args, **kwargs):
        raise RuntimeError("'ArxivSearchResult' object is not subscriptable")


def _executor():
    """The real executor, with only tool dispatch made to fail.

    Faking the executor means reimplementing the helpers act() relies on, and a
    fake that drifts from the real one would test nothing.
    """
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    executor = AutonomousAgentExecutor()
    executor.tool_registry = _FailingRegistry()
    return executor


class _Job:
    id = "job-1"
    goal = "find papers on speculative decoding"
    job_type = "research"
    iteration = 1
    user_id = "user-1"
    name = "fallback test"

    def __init__(self, **config):
        self.config = config


@pytest.fixture
def service(monkeypatch):
    """Exercise the fallback decision with the tool itself failing."""
    from app.services import agent_action_service as module

    svc = AgentActionService()

    async def _noop(**kwargs):
        return None

    monkeypatch.setattr(
        module.agent_execution_journal_service, "begin_tool_call", _noop
    )
    monkeypatch.setattr(
        module.agent_execution_journal_service, "complete_tool_call", _noop
    )

    return svc


async def _act(service, job):
    """Drive act() far enough to exercise the fallback decision."""
    return await service.act(
        _executor(),
        job,
        {"tool": "search_arxiv", "params": {"query": "speculative decoding"}},
        {},
        None,
    )


@pytest.mark.asyncio
async def test_a_failing_tool_reports_failure_by_default(service):
    result = await _act(service, _Job())

    assert not result.get("success"), "a broken tool must not report success"
    assert result.get("error")
    assert result.get("tool") in (None, "", "search_arxiv")


@pytest.mark.asyncio
async def test_the_alternative_is_suggested_not_executed(service):
    result = await _act(service, _Job())

    assert result.get("suggested_alternative_tool") == "search_documents"
    assert "fallback" not in result, "the alternative must not have been run"


@pytest.mark.asyncio
async def test_substitution_still_available_when_asked_for_explicitly(service):
    result = await _act(service, _Job(tool_fallback_enabled=True))

    # Opting in is allowed; it is no longer silent, and no longer the default.
    assert "suggested_alternative_tool" not in result
