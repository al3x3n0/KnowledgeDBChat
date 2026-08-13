"""Check a tool call against the tool's own schema before running it.

Tools validate their own arguments to wildly different standards. Some return
"query is required", some raise, and some accept nonsense and fail obscurely
somewhere inside. The catalog already carries each tool's JSON Schema, so the
same check can be applied to every tool before dispatch.

The point is the message as much as the rejection: an agent told "params
missing required field: query (expected string)" fixes its next call, while one
told "invalid input" guesses again and spends another iteration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# JSON Schema types mapped to what Python actually arrives as. bool is checked
# before int deliberately: bool is a subclass of int, so True would otherwise
# satisfy an "integer" field.
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _describe(value: Any) -> str:
    return type(value).__name__


def validate_tool_params(
    tool_name: str, params: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Return a message describing why this call is malformed, or None.

    Unknown tools and tools without a schema are not rejected: this guards
    against malformed calls, not against tools the catalog has yet to describe.
    """
    from app.agent_core.tool_catalog import get_tool_metadata

    metadata = get_tool_metadata(tool_name)
    schema = getattr(metadata, "input_schema", None) or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None

    supplied = params if isinstance(params, dict) else {}
    problems: List[str] = []

    for field in schema.get("required") or []:
        name = str(field)
        if name not in supplied:
            expected = (properties.get(name) or {}).get("type")
            problems.append(
                f"missing required field: {name}"
                + (f" (expected {expected})" if expected else "")
            )
            continue
        if supplied[name] in (None, ""):
            problems.append(f"required field {name} is empty")

    for name, value in supplied.items():
        if name.startswith("_"):  # runtime bookkeeping, not part of the schema
            continue
        spec = properties.get(name)
        if not isinstance(spec, dict):
            continue
        expected = spec.get("type")
        check = _TYPE_CHECKS.get(str(expected))
        if check and value is not None and not check(value):
            problems.append(
                f"field {name} should be {expected}, got {_describe(value)}"
            )
        choices = spec.get("enum")
        if isinstance(choices, list) and choices and value not in choices:
            problems.append(
                f"field {name} should be one of {', '.join(map(str, choices))}, "
                f"got {value!r}"
            )

    if not problems:
        return None

    known = ", ".join(sorted(properties)) or "none"
    return (
        f"{tool_name} was called with invalid parameters: "
        + "; ".join(problems)
        + f". Accepted parameters: {known}."
    )
