"""Pure normalization and response mapping for agent learning feedback."""

import re
from typing import Any, Optional

from app.schemas.agent_job import AgentJobFeedbackResponse


def sanitize_tool_names(
    values: Optional[list[str]],
    *,
    limit: int = 12,
) -> list[str]:
    """Return unique, valid tool identifiers within a bounded limit."""
    output: list[str] = []
    if not isinstance(values, list):
        return output
    normalized_limit = max(1, min(limit, 40))
    for raw in values:
        tool = str(raw or "").strip()
        if not tool or not re.match(r"^[a-zA-Z0-9_:\\-]{2,80}$", tool):
            continue
        if tool not in output:
            output.append(tool)
        if len(output) >= normalized_limit:
            break
    return output


def memory_to_feedback_response(memory: Any) -> AgentJobFeedbackResponse:
    """Map a ConversationMemory-like object to public feedback output."""
    context = memory.context if isinstance(memory.context, dict) else {}
    preferred = (
        context.get("preferred_tools")
        if isinstance(context.get("preferred_tools"), list)
        else []
    )
    discouraged = (
        context.get("discouraged_tools")
        if isinstance(context.get("discouraged_tools"), list)
        else []
    )
    try:
        rating = int(context.get("rating", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        rating = 0
    rating = max(1, min(5, rating)) if rating else 3
    return AgentJobFeedbackResponse(
        id=memory.id,
        job_id=memory.job_id,
        rating=rating,
        feedback=str(context.get("feedback_text") or memory.content or "").strip()
        or None,
        target_type=str(context.get("target_type") or "job"),
        target_id=str(context.get("target_id") or "").strip() or None,
        scope=str(context.get("scope") or "user"),
        preferred_tools=[str(item) for item in preferred[:20]],
        discouraged_tools=[str(item) for item in discouraged[:20]],
        checkpoint=str(context.get("checkpoint") or "").strip() or None,
        created_at=memory.created_at,
    )
