"""Notice when a chat message is asking to start a research campaign.

Chat answers questions. Some messages are not questions -- "run a campaign to
find out whether X" is a request to *start work*, and answering it with prose
is the wrong response to give. This turns such a message into a drafted
campaign the reply offers, which the person edits and launches, or ignores.

Deliberately a draft and not a launch. A campaign spends a job budget
autonomously, and inferring that from a sentence and acting on it is the kind
of helpfulness nobody asks for twice. The widget is the confirmation step.

Two stages, in this order for a reason:

1. A cheap textual filter. Chat is the busiest path in the product and most
   messages are plainly not campaign requests; sending every one to an LLM
   would put a model call and its latency in front of ordinary questions.
2. The model, only for what survives the filter, and asked for a schema rather
   than prose so the result is either usable or absent.

The filter is intentionally loose -- it decides who gets *asked*, not who gets
a campaign, so a false positive costs one structured call that returns
`is_campaign_request: false`, while a false negative silently loses the
feature. The model is the judge.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional

from loguru import logger

#: Verbs that propose work rather than ask about it.
_START = r"(?:start|launch|run|kick\s*off|set\s*up|create|begin|spin\s*up)"
#: What is being started. "study" and "investigation" carry the same intent as
#: the literal word, and a person asking for autonomous research rarely says
#: "campaign" unprompted.
_THING = r"(?:campaign|study|investigation|research\s+(?:effort|programme|program|push)|experiment\s+series)"

_PATTERNS = (
    re.compile(rf"\b{_START}\b[^.?!]{{0,40}}\b{_THING}\b", re.I),
    re.compile(rf"\b{_THING}\b[^.?!]{{0,30}}\b(?:to|that|which)\b", re.I),
    re.compile(
        r"\b(?:research|investigate|find\s+out)\b[^.?!]{0,30}\bautonomous", re.I
    ),
)

#: Below this a message is too short to carry a goal worth pursuing.
MIN_CHARS = 15


def looks_like_campaign_request(text: str) -> bool:
    """Whether this message is worth asking the model about.

    Cheap and loose on purpose; see the module docstring for why the cost of
    each kind of mistake is not symmetric.
    """
    body = str(text or "").strip()
    if len(body) < MIN_CHARS:
        return False
    return any(pattern.search(body) for pattern in _PATTERNS)


DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_campaign_request": {
            "type": "boolean",
            "description": (
                "True only if the message asks to START autonomous research, "
                "rather than asking a question about research."
            ),
        },
        "name": {"type": "string", "description": "Short title, under 60 characters."},
        "goal": {
            "type": "string",
            "description": (
                "What the campaign must establish, phrased so completion can "
                "be judged against it. Not a topic -- a claim to settle."
            ),
        },
        "items": {
            "type": "array",
            "description": "Seed questions to pursue first, 2 to 5 of them.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["title"],
            },
        },
    },
    "required": ["is_campaign_request"],
}

SYSTEM_PROMPT = """You decide whether a chat message is asking to start an autonomous research campaign, and if so, draft one.

A campaign spends a budget of agent jobs pursuing a goal without further input. Say yes only when the message asks for work to be STARTED. "How do campaigns work?", "what did the last campaign find?" and "should I run a campaign?" are all questions, not requests -- answer false.

When it is a request:
- `goal` is what completion is judged against. Write a claim to settle, not a topic to explore. "Establish whether X beats Y on Z" rather than "look into X".
- `items` are the first questions to pursue, 2 to 5. Each should be answerable by one agent job.
- Take the specifics from the message. Do not invent a subject the person did not mention.

If the message gestures at research but names nothing to settle, answer false: a campaign without a goal runs until it exhausts its budget."""


def _payload(completion: Any) -> Dict[str, Any]:
    """The object out of a completion, whichever way the provider returned it.

    `generate_structured` hands back an LLMCompletion, not a dict: providers
    with native schema output fill `.structured`, the rest leave JSON in
    `.text`, sometimes fenced. Treating the completion itself as a mapping is
    the quiet failure -- every field reads as missing and the draft silently
    becomes None.
    """
    structured = getattr(completion, "structured", None)
    if isinstance(structured, Mapping) and structured:
        return dict(structured)
    if isinstance(completion, Mapping):
        return dict(completion)
    text = str(getattr(completion, "text", "") or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean(draft: Mapping[str, Any], fallback_name: str) -> Optional[Dict[str, Any]]:
    """The draft as the API would accept it, or None if it is not usable."""
    # `is_campaign_request` is a veto, not a requirement. The schema marks it
    # required and DeepSeek's JSON mode does not enforce that: for a real
    # request it returns {goal, items} and omits the flag entirely, so
    # demanding it threw away every usable draft. Observed on the other side
    # too -- for "Should I run a campaign about prefetching?" it answered
    # {"start_campaign": ...} with no goal at all.
    #
    # So an explicit false is honoured, and otherwise the goal decides. That
    # is the discriminator the model actually expresses: it drafts a goal when
    # it reads the message as a request and omits one when it does not.
    if draft.get("is_campaign_request") is False:
        return None
    goal = str(draft.get("goal") or "").strip()
    if not goal:
        # The service refuses a campaign with no goal, and it is right to.
        # Offering a widget that cannot be submitted is worse than offering
        # nothing.
        return None
    items = []
    for row in draft.get("items") or []:
        # Asked for objects; models routinely answer with bare strings, and a
        # seed question is perfectly expressible as one. Refusing those would
        # drop a usable draft over its shape.
        if isinstance(row, str):
            row = {"title": row}
        if not isinstance(row, Mapping):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        item: Dict[str, Any] = {"title": title[:300]}
        detail = str(row.get("detail") or "").strip()
        if detail:
            item["detail"] = detail
        items.append(item)
        if len(items) >= 5:
            break
    # The model often omits `name`. Falling back to the raw message gives
    # titles like "Start a campaign to find out whether a stride pref..." --
    # the request, not the subject. The goal is already a cleaned statement of
    # what is being settled, so its first clause reads as a title.
    name = str(draft.get("name") or "").strip()
    if not name:
        name = re.split(r"[.;:]", goal)[0].strip()
        name = re.sub(
            r"^(?:establish|determine|find\s+out|assess)\s+(?:whether\s+)?",
            "",
            name,
            flags=re.I,
        ).strip()
        name = (name[:1].upper() + name[1:]) if name else fallback_name
    return {
        "name": name[:300],
        "goal": goal,
        "items": items,
        # The person chooses the real budget in the widget; this is a starting
        # point sized to the seeds rather than the service's default of 10.
        "max_jobs": max(2, min(len(items) * 2 or 4, 20)),
    }


async def draft_from_message(
    message: str,
    *,
    llm_service: Any,
    user_id: Any = None,
    db: Any = None,
) -> Optional[Dict[str, Any]]:
    """A campaign draft for this message, or None if it is not asking for one.

    Never raises: this runs inside the chat request, and a chat reply must not
    be lost because an optional offer could not be produced.
    """
    body = str(message or "").strip()
    if not looks_like_campaign_request(body):
        return None

    try:
        completion = await llm_service.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_message=body,
            response_schema=DRAFT_SCHEMA,
            task_type="fast",
            user_id=user_id,
            db=db,
            snapshot_context={"phase": "campaign_intent"},
        )
    except Exception as exc:
        logger.warning(f"Campaign intent detection failed: {exc}")
        return None

    payload = _payload(completion)
    try:
        return _clean(payload, fallback_name=body[:60])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Campaign draft could not be normalised: {exc}")
        return None
