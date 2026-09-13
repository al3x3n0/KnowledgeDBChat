"""Regressions for tool failures observed in a real agent run.

Each of these cost the agent an iteration and produced a failed action, and
none of them were the model's fault.
"""

import pytest

from app.services.autonomous_agent_executor import (
    AutonomousAgentExecutor,
    _tool_requires_params,
)
from app.services.custom_tool_service import CustomToolService


class _Job:
    id = "job-1"
    goal = "Investigate why a compiler removes a branch"
    job_type = "research"
    config: dict = {}
    progress = 0


def test_catalog_required_params_are_recognized():
    assert _tool_requires_params("search_web") is True
    assert _tool_requires_params("summarize_url") is True
    assert _tool_requires_params("not_a_real_tool") is False


def test_recovery_builds_a_usable_search_web_call():
    """It used to emit search_web with no params, which the tool rejects."""
    executor = AutonomousAgentExecutor()

    action = executor._build_action_for_tool(
        tool="search_web", job=_Job(), purpose="recover"
    )

    assert action["tool"] == "search_web"
    assert action["params"]["query"], "search_web without a query always fails"


def test_recovery_declines_tools_it_cannot_fill():
    """Better to try the next candidate than spend a turn being rejected."""
    executor = AutonomousAgentExecutor()

    action = executor._build_action_for_tool(
        tool="summarize_url", job=_Job(), purpose="recover"
    )

    assert action is None


def test_recovery_still_builds_argument_free_tools():
    executor = AutonomousAgentExecutor()

    action = executor._build_action_for_tool(
        tool="some_tool_with_no_required_args", job=_Job(), purpose="recover"
    )

    assert action == {
        "tool": "some_tool_with_no_required_args",
        "params": {},
        "purpose": "recover",
    }


@pytest.mark.asyncio
async def test_execute_python_runs_valid_code():
    """compile_restricted returns a code object; reading .errors off it raised
    AttributeError for every input, so this tool never worked."""
    service = CustomToolService()

    result = await service._execute_python(
        config={"code": "output = {'value': 6 * 7}", "timeout_seconds": 5},
        inputs={},
        user=None,
    )

    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_execute_python_reports_a_syntax_error_as_a_compile_error():
    from app.services.custom_tool_service import ToolExecutionError

    service = CustomToolService()

    with pytest.raises(ToolExecutionError) as excinfo:
        await service._execute_python(
            config={"code": "def broken(:", "timeout_seconds": 5},
            inputs={},
            user=None,
        )

    assert "compilation errors" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_blocked_import_is_not_reported_as_a_missing_sandbox():
    """The sandbox correctly refuses subprocess; the message used to blame a
    missing RestrictedPython install, sending a reader after the wrong fix."""
    from app.services.custom_tool_service import ToolExecutionError

    service = CustomToolService()

    with pytest.raises(ToolExecutionError) as excinfo:
        await service._execute_python(
            config={"code": "import subprocess", "timeout_seconds": 5},
            inputs={},
            user=None,
        )

    message = str(excinfo.value)
    assert "does not allow" in message
    assert "Install the RestrictedPython package" not in message
    assert "Allowed modules:" in message
