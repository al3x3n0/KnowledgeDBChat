"""Parsing JSON out of model output, in one place.

Models asked for JSON reply with JSON, JSON wrapped in prose, JSON in a markdown
fence, or prose alone. Fifteen services grew their own tolerant parser for this,
and they did not agree: the decision parser scans balanced braces and recovers a
payload from ``{"a":1} and {"b":2}``, while the runners took the widest ``{``-to-
``}`` span and returned nothing for the same reply. The same malformed answer
therefore succeeded in one subsystem and failed in another.

This is the single implementation, using the stronger of the two algorithms. It
tries, in order: the whole string, the first fenced block, then each balanced
brace span from left to right — tracking string literals and escapes so a ``}``
inside a string does not end the object early.

Tolerance is a fallback, not a strategy. Callers that need a guarantee should ask
the provider for schema-constrained output (``LLMService.generate_structured``),
which removes the guessing instead of tuning it.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _loads_object(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_spans(text: str) -> dict[str, Any] | None:
    """Return the first balanced ``{...}`` span that parses as an object."""
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    parsed = _loads_object(text[start : idx + 1])
                    if parsed is not None:
                        return parsed
                    break
        # Only the first unbalanced opening brace is worth retrying; scanning
        # every brace in a long reply is not worth the time.
        if depth != 0:
            continue
    return None


def _loads_array(candidate: str) -> list[Any] | None:
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def extract_json_array(value: Any) -> list[Any] | None:
    """Return the first JSON array in model output, or None.

    Separate from ``extract_json_object`` rather than a general "first JSON
    value": a reply containing both must still yield the object to callers that
    asked for one, and the array to callers that asked for an array.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return None

    direct = _loads_array(value.strip())
    if direct is not None:
        return direct

    fenced = FENCE_PATTERN.search(value)
    if fenced:
        parsed = _loads_array(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    start = value.find("[")
    end = value.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return _loads_array(value[start : end + 1])


def extract_json_object(value: Any) -> dict[str, Any] | None:
    """Return the first JSON object in model output, or None.

    Accepts an already-parsed dict and passes it through, so callers that may
    receive either a string or a decoded payload need no special case.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return None

    direct = _loads_object(value.strip())
    if direct is not None:
        return direct

    fenced = FENCE_PATTERN.search(value)
    if fenced:
        parsed = _loads_object(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    return _balanced_spans(value)
