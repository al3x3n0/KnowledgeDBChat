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

# A reply with hundreds of brace pairs is malformed by any reasonable reading;
# parsing every one of them is wasted work on untrusted input.
MAX_SPAN_ATTEMPTS = 200


def _loads_object(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_spans(text: str) -> dict[str, Any] | None:
    """Return the first balanced ``{...}`` span that parses as an object.

    One left-to-right pass collecting every balanced span, then attempts in
    order of opening brace. The previous implementation restarted a scan from
    every ``{``, which is quadratic: 28KB of unbalanced braces — an entirely
    plausible malformed reply — took 37 seconds, in a code path that parses
    untrusted model output.
    """
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
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
            stack.append(idx)
        elif ch == "}" and stack:
            spans.append((stack.pop(), idx))

    # Attempt by opening position so the outermost, earliest object wins, which
    # is what callers expect when a reply nests or repeats objects.
    for span_start, span_end in sorted(spans)[:MAX_SPAN_ATTEMPTS]:
        parsed = _loads_object(text[span_start : span_end + 1])
        if parsed is not None:
            return parsed
    return None


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
