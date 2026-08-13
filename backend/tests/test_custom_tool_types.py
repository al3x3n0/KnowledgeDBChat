"""One rule for which custom tool types may be created.

The docker gate lived twice, once in POST /user-tools and once in the agent's
create_custom_tool, so turning CUSTOM_TOOL_DOCKER_ENABLED on or off had to be
honoured in two places and could drift in one.
"""

import pytest

from app.services.custom_tool_types import (
    DOCKER_TOOL_TYPE,
    WORKFLOW_RUNNER_TYPE,
    allowed_custom_tool_types,
    reject_custom_tool_type,
)


@pytest.fixture
def docker_off(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False, raising=False)


@pytest.fixture
def docker_on(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", True, raising=False)


def test_the_ordinary_types_are_always_creatable(docker_off):
    allowed = allowed_custom_tool_types()

    assert {"webhook", "transform", "python", "llm_prompt"} <= allowed


def test_docker_is_gated_by_its_flag(docker_off):
    assert DOCKER_TOOL_TYPE not in allowed_custom_tool_types()
    assert reject_custom_tool_type(DOCKER_TOOL_TYPE) is not None


def test_docker_appears_when_the_flag_is_on(docker_on):
    assert DOCKER_TOOL_TYPE in allowed_custom_tool_types()
    assert reject_custom_tool_type(DOCKER_TOOL_TYPE) is None


def test_the_gate_applies_to_both_callers(docker_off):
    """Whatever the flag says, it says to the operator path and the agent path
    alike."""
    for include_runner in (True, False):
        assert DOCKER_TOOL_TYPE not in allowed_custom_tool_types(
            include_workflow_runner=include_runner
        )


def test_workflow_runner_is_the_operator_path_only(docker_off):
    assert WORKFLOW_RUNNER_TYPE in allowed_custom_tool_types(
        include_workflow_runner=True
    )
    assert WORKFLOW_RUNNER_TYPE not in allowed_custom_tool_types(
        include_workflow_runner=False
    )
    assert reject_custom_tool_type(WORKFLOW_RUNNER_TYPE) is not None


def test_a_rejection_names_what_is_allowed(docker_off):
    message = reject_custom_tool_type("nonsense")

    assert "transform" in message and "llm_prompt" in message
    assert "nonsense" in message


def test_case_and_padding_are_tolerated(docker_off):
    assert reject_custom_tool_type("  TRANSFORM  ") is None
