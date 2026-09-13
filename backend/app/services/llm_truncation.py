"""Recovering from a response the model ran out of room to finish.

Truncation is not the same failure as malformed output, and treating it as one
wastes the recovery. Malformed output means the model wrote something wrong and
asking it again may produce something right. Truncation means it was writing
the correct thing and the budget ended mid-sentence: the prefix is good, and
either closing it or asking for more room recovers the work. Sending it back
with "your JSON was invalid" invites it to rewrite a plan it had already got
right, on a budget that is still too small.

Two shapes, and they need different answers.

    Empty content. The whole budget went to reasoning and nothing was emitted.
    Nothing to repair -- the only recovery is more room. Measured on a research
    job: six calls in one run came back with finish_reason='length' and
    completion_tokens exactly at the cap, having emitted no answer at all, and
    the run could not advance past iteration zero.

    Partial content. The model was part-way through a JSON object when the
    budget ran out, so the text is a valid prefix of valid JSON. Closing the
    open braces recovers it without another call, which matters because the
    retry is on the same budget that just proved too small.

The closing is deliberately conservative. It refuses anything it cannot
finish honestly rather than guessing a value the model never wrote: a repaired
object that invents a field is worse than a failed parse, because the parse
failure is visible and the invention is not.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

#: What a provider reports when the budget, not the model, ended the response.
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})

#: How much more room to ask for after a truncated attempt. Not a small bump:
#: the budget did not miss by a little, it was spent entirely on reasoning
#: before the answer began, and a 10% increase buys another failure.
RETRY_MULTIPLIER = 2

#: A ceiling on that growth, so a pathological prompt cannot escalate without
#: bound across retries.
MAX_RETRY_TOKENS = 32000


def is_truncated(finish_reason: Optional[str], content: Optional[str]) -> bool:
    """Did the budget end this response rather than the model?"""
    return str(finish_reason or "") in TRUNCATION_REASONS


def next_budget(
    current: Optional[int], *, cap: int = MAX_RETRY_TOKENS
) -> Optional[int]:
    """A budget worth retrying on, or None when there is no room left to give.

    None rather than the cap: a caller already at the ceiling should report the
    truncation rather than repeat the same call and call it a retry.
    """
    try:
        value = int(current or 0)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value >= cap:
        return None
    return min(value * RETRY_MULTIPLIER, cap)


def _safe_cut(text: str) -> Tuple[Optional[int], List[str]]:
    """Where the JSON could honestly be cut, and what is left open there.

    Completeness is tracked per CONTAINER, not per token, because those differ
    in the case that matters: in `{"tool": "x", "arguments":` the key is a
    finished string but an unfinished member, and cutting after it produces a
    key with no value. Each frame therefore remembers the offset just past its
    last complete member -- a value that ended, not merely a token that did.

    Returns (cut, closers) or (None, []) when the text is broken rather than
    unfinished, so a caller can tell "cannot finish this" from "nothing to do".
    """
    frames: List[dict] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == '"':
            end = index + 1
            while end < length:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == '"':
                    break
                end += 1
            if end >= length:
                break  # the string was still being written
            if frames and frames[-1]["expect"] == "key":
                frames[-1]["expect"] = "colon"
            elif frames and frames[-1]["expect"] == "value":
                frames[-1]["safe"] = end + 1
                frames[-1]["expect"] = "comma"
            index = end + 1
            continue

        if char in "{[":
            frames.append(
                {
                    "close": "}" if char == "{" else "]",
                    "expect": "key" if char == "{" else "value",
                    "safe": index + 1,
                }
            )
            index += 1
            continue

        if char in "}]":
            if not frames or frames[-1]["close"] != char:
                return None, []  # mismatched: broken, not unfinished
            frames.pop()
            if frames:
                frames[-1]["safe"] = index + 1
                frames[-1]["expect"] = "comma"
            index += 1
            continue

        if char == ":":
            if frames and frames[-1]["expect"] == "colon":
                frames[-1]["expect"] = "value"
            index += 1
            continue

        if char == ",":
            if frames and frames[-1]["expect"] == "comma":
                frames[-1]["expect"] = "key" if frames[-1]["close"] == "}" else "value"
            index += 1
            continue

        if char.isspace():
            index += 1
            continue

        # A bare literal: number, true, false, null. It only counts as a
        # complete value once a delimiter proves the model finished writing it
        # -- `12` may have been on its way to `125`.
        end = index
        while end < length and text[end] not in ",:{}[] \t\r\n":
            end += 1
        if end >= length:
            break  # the literal was still being written
        if frames and frames[-1]["expect"] == "value":
            frames[-1]["safe"] = end
            frames[-1]["expect"] = "comma"
        index = end
        continue

    if not frames:
        return None, []  # balanced: not a truncation
    return frames[-1]["safe"], [f["close"] for f in frames]


def repair_truncated_json(text: str) -> Optional[str]:
    """Close a JSON value the model was still writing, or None if it cannot be.

    Returns None when the text is already balanced (nothing to repair), when it
    is broken rather than unfinished, or when nothing survived the cut. None is
    a real answer: the caller should fall back to asking the model, not to a
    guess made up locally.
    """
    if not text:
        return None
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    body = text[min(starts) :]

    cut, closers = _safe_cut(body)
    if cut is None:
        return None

    repaired = body[:cut].rstrip()
    while repaired and repaired[-1] in ",:":
        repaired = repaired[:-1].rstrip()
    if not repaired:
        return None
    repaired += "".join(reversed(closers))

    try:
        parsed = json.loads(repaired)
    except ValueError:
        return None
    if parsed in ({}, []):
        # Structurally valid and empty: the budget ended before anything was
        # said. Reporting that as a repair hides the truncation behind a
        # decision nobody made.
        return None
    return repaired
