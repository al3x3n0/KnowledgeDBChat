"""A failing tool must report its failure, and nothing else may run instead.

Substituting a different tool and rewriting the result to success turned "this
tool is broken" into "the agent got an answer". A broken arXiv search became a
knowledge-base search whose unrelated documents were recorded as research
findings, and nothing in the run said otherwise. The substitution machinery is
gone rather than disabled, so no configuration can bring it back.
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
async def test_nothing_runs_in_place_of_the_failed_tool(service):
    result = await _act(service, _Job())

    assert "fallback" not in result
    assert "suggested_alternative_tool" not in result


@pytest.mark.asyncio
async def test_no_configuration_can_re_enable_substitution(service):
    """The old opt-in key must do nothing: the code path no longer exists."""
    result = await _act(service, _Job(tool_fallback_enabled=True))

    assert not result.get("success")
    assert "fallback" not in result
    assert result.get("tool") in (None, "", "search_arxiv")


def test_the_substitution_machinery_is_gone():
    import app.services.agent_action_service as module

    source = open(module.__file__).read()
    assert "_fallback_action_for" not in source
    assert "tool_fallback_enabled" not in source
