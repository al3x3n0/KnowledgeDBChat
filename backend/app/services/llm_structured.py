"""One way to ask a model for JSON.

Two paths to structured output grew side by side: ``generate_structured``, which
uses each provider's schema-constrained output and native tool calling, and
``generate_response``, which asks for JSON in the prompt and leaves the caller
to parse prose. The second is used in 46 files and the first in 4, which is why
so many subsystems grew their own parser — and why a fenced reply could fail a
job in one place and be handled in another.

``ask_for_json`` collapses that into a single call: constrain the schema at the
provider when it can, fall back to prompted text and shared parsing when it
cannot, and return a dict either way. Callers stop caring which path ran.

The fallback is what makes this adoptable — providers and models differ in
schema support, and a helper that only worked on some of them would just become
a third path.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from app.services import llm_json


async def ask_for_json(
    llm_service: Any,
    *,
    schema: dict[str, Any],
    system_prompt: str | None = None,
    user_message: str | None = None,
    task_type: str = "chat",
    user_settings: Any = None,
    routing: dict[str, Any] | None = None,
    snapshot_context: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    user_id: Any = None,
    db: Any = None,
) -> dict[str, Any] | None:
    """Ask for a JSON object matching ``schema``. Returns None if none arrives.

    ``schema`` is a JSON Schema dict — ``SomeModel.model_json_schema()`` for a
    pydantic model.
    """
    structured = await _try_structured(
        llm_service,
        schema=schema,
        system_prompt=system_prompt,
        user_message=user_message,
        task_type=task_type,
        user_settings=user_settings,
        routing=routing,
        snapshot_context=snapshot_context,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        user_id=user_id,
        db=db,
    )
    if structured is not None:
        return structured

    response = await llm_service.generate_response(
        system_prompt=system_prompt,
        user_message=user_message,
        task_type=task_type,
        user_settings=user_settings,
        routing=routing,
        snapshot_context=snapshot_context,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        user_id=user_id,
        db=db,
    )
    return llm_json.extract_json_object(response)


async def _try_structured(
    llm_service: Any,
    *,
    schema: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Native schema-constrained path. Returns None so the caller can fall back."""
    try:
        completion = await llm_service.generate_structured(
            response_schema=schema, **kwargs
        )
    except Exception as exc:
        # Provider lacks schema support, is misconfigured, or failed the call.
        # Not fatal: the prompted path still produces an answer.
        logger.debug(f"Structured output unavailable, falling back to prompt: {exc}")
        return None

    if completion is None:
        return None

    payload = getattr(completion, "structured", None)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return llm_json.extract_json_object(payload)

    # Some providers honour the schema but still answer as text.
    return llm_json.extract_json_object(str(getattr(completion, "text", "") or ""))


def schema_hint(schema: dict[str, Any], *, indent: int = 2) -> str:
    """Render a schema for embedding in a prompt.

    The prompted fallback only produces the right shape if the prompt says what
    the shape is, so callers migrating to ``ask_for_json`` should keep telling
    the model what they want rather than relying on the schema argument alone.
    """
    return json.dumps(schema, indent=indent, default=str)
