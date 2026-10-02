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

import functools
import json
import re
from typing import Any, Mapping

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

# A reply with hundreds of brace pairs is malformed by any reasonable reading;
# parsing every one of them is wasted work on untrusted input.
MAX_SPAN_ATTEMPTS = 200


# RecursionError as well as a decode error: a reply of thousands of unclosed
# brackets is not malformed JSON as far as the decoder is concerned until it
# has recursed into every one of them, and it raises this instead.
def _loads_object(candidate: str, strict: bool = True) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate, strict=strict)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _loads_array(candidate: str, strict: bool = True) -> list[Any] | None:
    try:
        parsed = json.loads(candidate, strict=strict)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, list) else None


def _balanced_spans(
    text: str, opener: str = "{", closer: str = "}", loads: Any = _loads_object
) -> Any:
    """Return the first balanced ``{...}`` span that parses as an object.

    With ``opener``/``closer``/``loads`` given, the same for ``[...]`` and an
    array: one scanner, so the two cannot come to disagree about what a string
    literal is.

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
        elif ch == opener:
            stack.append(idx)
        elif ch == closer and stack:
            spans.append((stack.pop(), idx))

    # Attempt by opening position so the outermost, earliest object wins, which
    # is what callers expect when a reply nests or repeats objects.
    for span_start, span_end in sorted(spans)[:MAX_SPAN_ATTEMPTS]:
        parsed = loads(text[span_start : span_end + 1])
        if parsed is not None:
            return parsed
    return None


def extract_json_object(value: Any, *, strict: bool = True) -> dict[str, Any] | None:
    """Return the first JSON object in model output, or None.

    Accepts an already-parsed dict and passes it through, so callers that may
    receive either a string or a decoded payload need no special case.

    ``strict=False`` accepts raw newlines and tabs inside strings. A reply that
    carries a whole source file in a JSON string is written with literal
    newlines as often as with ``\\n``, and strict JSON rejects those: a
    4,117-character reply full of usable proposals once parsed as nothing.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return None

    loads = functools.partial(_loads_object, strict=strict)
    direct = loads(value.strip())
    if direct is not None:
        return direct

    fenced = FENCE_PATTERN.search(value)
    if fenced:
        parsed = loads(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    return _balanced_spans(value, loads=loads)


def require_json_object(
    value: Any, message: str = "No JSON object found in response"
) -> dict[str, Any]:
    """The first JSON object in model output, or a ``ValueError`` saying so.

    For callers that treat an unparseable reply as a failure of the request
    rather than as an empty answer.
    """
    parsed = extract_json_object(value)
    if parsed is None:
        raise ValueError(message)
    return parsed


def completion_object(completion: Any, *, strict: bool = True) -> dict[str, Any]:
    """The object out of a completion, whichever way the provider returned it.

    ``generate_structured`` hands back an ``LLMCompletion``, not a dict:
    providers with native schema output fill ``.structured``, the rest leave
    JSON in ``.text``, sometimes inside a fence or a sentence. Treating the
    completion itself as a mapping is the quiet failure -- every field reads
    as missing, so the caller sees a model that cannot follow instructions.
    Measured: a drafter reported "the reply was not JSON" three times against
    a model that had answered correctly each time.

    Returns ``{}`` when there is no object. Five services each carried a copy
    of this, differing in which malformed replies they survived.
    """
    structured = getattr(completion, "structured", None)
    if isinstance(structured, Mapping) and structured:
        return dict(structured)
    if isinstance(completion, Mapping):
        return dict(completion)
    text = getattr(completion, "text", None)
    if text is None and isinstance(completion, str):
        text = completion
    return extract_json_object(str(text or ""), strict=strict) or {}


def extract_json_array(value: Any, *, strict: bool = True) -> list[Any] | None:
    """Return the first JSON array in model output, or None.

    Separate from ``extract_json_object`` rather than a general "first JSON
    value": a reply containing both must still yield the object to callers that
    asked for one, and the array to callers that asked for an array.

    This was removed while two callers still used it, and both caught the
    resulting ``AttributeError`` as an ordinary parse failure. The chat planner
    therefore discarded every tool call the model made, for eight weeks, while
    logging one line per turn: chat answered as though no tool had been needed.
    ``tests/test_llm_json.py`` now fails if a name a caller uses is missing.
    """
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return None

    loads = functools.partial(_loads_array, strict=strict)
    direct = loads(value.strip())
    if direct is not None:
        return direct

    fenced = FENCE_PATTERN.search(value)
    if fenced:
        parsed = loads(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    return _balanced_spans(value, "[", "]", loads)
