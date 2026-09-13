"""Which custom tool types may be created, and by whom.

The rule lived twice: once in POST /user-tools and once in the agent's
create_custom_tool. Both had their own copy of the docker gate, so enabling or
disabling CUSTOM_TOOL_DOCKER_ENABLED had to be honoured in two places and could
drift in one. The gate belongs in one function.

The two callers differ on purpose and that difference is a parameter rather
than a second copy: an operator posting to the API may create a
workflow_runner, while an agent may not, because a workflow_runner points at a
workflow id that workflow synthesis fills in.
"""

from __future__ import annotations

from typing import Optional, Set

# Types whose executor needs nothing beyond the process it runs in.
BASE_CUSTOM_TOOL_TYPES: Set[str] = {
    "webhook",
    "transform",
    "python",
    "llm_prompt",
}

# Reserved for workflow synthesis, which supplies the workflow id it targets.
WORKFLOW_RUNNER_TYPE = "workflow_runner"

# Runs a container on the host daemon, so it stays behind its own flag.
DOCKER_TOOL_TYPE = "docker_container"


def allowed_custom_tool_types(*, include_workflow_runner: bool = False) -> Set[str]:
    """The tool types creatable right now, given deployment settings."""
    from app.core.config import settings

    allowed = set(BASE_CUSTOM_TOOL_TYPES)
    if include_workflow_runner:
        allowed.add(WORKFLOW_RUNNER_TYPE)
    if bool(getattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False)):
        allowed.add(DOCKER_TOOL_TYPE)
    return allowed


def reject_custom_tool_type(
    tool_type: str, *, include_workflow_runner: bool = False
) -> Optional[str]:
    """Return why this type is not creatable, or None if it is.

    Naming the allowed set in the message matters: a caller told only that its
    choice was invalid will guess again.
    """
    allowed = allowed_custom_tool_types(include_workflow_runner=include_workflow_runner)
    normalized = str(tool_type or "").strip().lower()
    if normalized in allowed:
        return None
    return (
        f"tool_type must be one of: {', '.join(sorted(allowed))}. "
        f"Got {tool_type!r}."
    )
