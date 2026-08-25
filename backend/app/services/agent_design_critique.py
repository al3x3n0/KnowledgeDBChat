"""Attack an experiment's design before it is paid for.

The validity predicates in this package judge *results*: the instrument was
verified, the prediction was settled, the trace held one regime. They are
deterministic and they are the gate. What they cannot do is arrive in time --
each of them fires after a simulation has already run, and each exists because
some earlier trace defeated the set before it. That is a growing list of known
traps rather than a faculty for suspecting new ones.

This is the other half. A run that has written a workload and is about to spend
half an hour simulating it gets the design attacked first, by a model that did
not write it and is asked only to find what is wrong with it.

**Why a separate call rather than more instructions in the thinking prompt.**
The model that produced a design is reasoning from the assumptions that
produced it; asked to check its own work it tends to confirm it. A critic given
the artifact and none of the reasoning behind it is not defending anything. The
live case this was built from: asked for a workload whose phases *alternate*,
an agent wrote a hundred of one phase followed by a hundred of the other, which
is two experiments rather than an alternation -- and then spent the simulation
budget on it.

**Diverse lenses, not more critics.** Asking one prompt three times produces
one concern three times. Each lens below asks a different question, because the
ways a measurement goes wrong are not variations of each other: answering the
wrong question, being an artifact of the harness, and costing more than the
budget are unrelated failures.

**A critic that always finds something is noise.** Every lens is told that
finding nothing is a valid answer and is the expected one for a sound design.
The cost of a false concern is a run that rewrites a good workload.

**This advises; it does not gate.** Control flow stays deterministic -- a
campaign has to replay identically, and a model in the loop gives that up. The
concerns are recorded so a contract can require that the run *answered* them,
which is a deterministic question about a non-deterministic input.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from loguru import logger

#: What each critic is asked to look for. The names are stable because they end
#: up on findings and in contract remedies.
LENSES: Dict[str, str] = {
    "answers_the_question": (
        "Does this design measure what the goal asked for, or something "
        "adjacent to it? Compare the stated goal against what the artifact "
        "actually does, word by word. A design that is internally sound and "
        "answers a different question is the failure this lens exists for."
    ),
    "artifact_of_the_harness": (
        "What could make a result from this an artifact rather than a "
        "measurement? Consider: whether the quantity varies enough to be worth "
        "measuring at all, whether structure is planted by the design rather "
        "than found in the workload, whether the run has distinct regimes that "
        "would be spliced together, and whether the sample count supports the "
        "estimate the goal wants."
    ),
    "cost_and_feasibility": (
        "What will this cost to run, and will it finish? Estimate the work "
        "implied by the loops. An out-of-order simulator executes on the order "
        "of 100k instructions per second, and the tool that runs this times "
        "out. A design that cannot complete is not a design."
    ),
}

CRITIQUE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One sentence naming the specific defect.",
                    },
                    "why_it_matters": {
                        "type": "string",
                        "description": (
                            "What the result would be wrong about if this "
                            "stands, in terms of the goal."
                        ),
                    },
                    "remedy": {
                        "type": "string",
                        "description": "The concrete change to the artifact.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["blocking", "serious", "minor"],
                        "description": (
                            "blocking: the result would be meaningless. "
                            "serious: it would need caveating. "
                            "minor: worth knowing."
                        ),
                    },
                },
                "required": ["summary", "why_it_matters", "remedy", "severity"],
            },
        }
    },
    "required": ["concerns"],
}

_PROMPT = """You are reviewing an experiment before it is run. You did not \
write it and you are not defending it.

THE GOAL THE EXPERIMENT IS MEANT TO SERVE:
{goal}

WHAT THE RUN INTENDS TO DO WITH IT:
{intent}

THE ARTIFACT:
```
{artifact}
```

YOUR LENS -- look only for this:
{lens}

Finding nothing is a valid answer and is the expected one for a sound design. \
Do not manufacture concerns to appear useful: a false concern costs a rewrite \
of a workload that was already correct. Report only defects you can point at \
in the artifact above, and say where.

Return concerns as JSON matching the schema. An empty list means the design is \
sound under this lens."""


#: The schema is a request, not a guarantee. Asked for summary/why_it_matters/
#: remedy/severity, this provider returned {"concern": ..., "location": ...} --
#: a correct finding, precisely located, in the wrong shape. The first version
#: of this module required the declared keys, dropped it, and reported "0
#: concerns", which reads as a clean bill of health rather than as a parse
#: failure. Structured output enforced by the provider would not need this;
#: output that is merely requested does.
_SUMMARY_KEYS = ("summary", "concern", "issue", "problem", "title", "finding")
_WHY_KEYS = ("why_it_matters", "why", "impact", "rationale", "consequence")
_REMEDY_KEYS = ("remedy", "fix", "suggestion", "recommendation", "action")


def _first(concern: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = concern.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_concern_list(payload: Any) -> List[Any]:
    """Find the concerns whatever envelope they arrived in.

    The declared schema is an object with a `concerns` array. This provider has
    returned that, and also a bare array, and also objects keyed differently --
    all with correct content. Requiring the declared envelope raised
    AttributeError on the bare list, which is at least loud; requiring the
    declared field names failed silently and reported a clean design. Neither
    is a reason to trust the shape.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        listed = payload.get("concerns")
        if isinstance(listed, list):
            return listed
        for value in payload.values():
            if isinstance(value, list):
                return value
        # A lone concern returned unwrapped is still a concern.
        if any(isinstance(v, str) and v.strip() for v in payload.values()):
            return [payload]
    return []


def _clean(concern: Any, lens: str) -> Optional[Dict[str, Any]]:
    if not isinstance(concern, Mapping):
        return None
    summary = _first(concern, _SUMMARY_KEYS)
    if not summary:
        # Last resort before discarding: any string field at all. A concern
        # that cannot be read is still evidence the reviewer found something,
        # and losing it silently is worse than reporting it oddly.
        summary = next(
            (v.strip() for v in concern.values() if isinstance(v, str) and v.strip()),
            "",
        )
    if not summary:
        return None

    # An absent severity is NOT a middle one. Defaulting it to "serious" made
    # four unrated concerns read as four judged ones, and "0 blocking" read as
    # a reviewer declining to escalate when it had never rated anything. The
    # rating is the model's to give; its absence is reported, not filled in.
    severity = str(concern.get("severity") or "").strip().lower()
    if severity not in ("blocking", "serious", "minor"):
        severity = "unrated"
    record = {
        "lens": lens,
        "summary": summary[:600],
        "why_it_matters": _first(concern, _WHY_KEYS)[:600],
        "remedy": _first(concern, _REMEDY_KEYS)[:600],
        "severity": severity,
    }
    location = concern.get("location")
    if isinstance(location, str) and location.strip():
        record["location"] = location.strip()[:300]
    return record


#: How many times a lens is asked again when its answer comes back unusable.
#: The decision parser retries the same way and for the same reason: the model
#: is capable of the shape, it just did not produce it this time.
MAX_ATTEMPTS = 3

_CORRECTION = """Your previous reply could not be read as a critique.

What came back:
{received}

Reply with a JSON object of exactly this shape and nothing else:
{{"concerns": [{{"summary": "...", "why_it_matters": "...", "remedy": "...", "severity": "blocking|serious|minor"}}]}}

If the design is sound under your lens, reply {{"concerns": []}}. Do not change your judgement to fit the shape -- if you found a defect last time, it is still there; say it again in this form."""


async def _ask_lens(
    llm_service: Any, prompt: str, lens: str
) -> tuple[Optional[List[Any]], str]:
    """One lens, retried until its answer can be read.

    Returns (concerns, failure_reason). A `None` result means the lens never
    produced anything readable and must be reported as unreviewed -- retried
    and still unusable is not the same as reviewed and clean.

    Tolerant parsing comes first and retrying second, in that order. A correct
    finding in the wrong shape should not cost another call: the first version
    of this lost a precisely-located defect because it required the declared
    field names, and no number of retries fixes a parser that discards good
    answers.
    """
    last_seen = ""
    for attempt in range(MAX_ATTEMPTS):
        message = (
            "Report your concerns."
            if attempt == 0
            else _CORRECTION.format(received=last_seen[:500] or "(nothing)")
        )
        try:
            completion = await llm_service.generate_structured(
                system_prompt=prompt,
                user_message=message,
                response_schema=CRITIQUE_SCHEMA,
            )
            payload = getattr(completion, "structured", None)
            text = str(getattr(completion, "text", "") or "")
            if payload is None:
                payload = json.loads(text or "{}")
            last_seen = text or json.dumps(payload, default=str)
        except Exception as exc:
            last_seen = str(exc)
            logger.warning(
                f"Design critique lens {lens} attempt {attempt + 1} failed: {exc}"
            )
            continue

        raw = _as_concern_list(payload)
        if raw:
            readable = [c for c in raw if _clean(c, lens)]
            if readable:
                return raw, ""
            # Something was returned and none of it could be read. Ask again
            # rather than recording a design as clean on an unreadable answer.
            logger.warning(f"Design critique lens {lens}: unreadable concerns")
            continue

        # An explicitly empty list is an answer, not a failure.
        if isinstance(payload, Mapping) and isinstance(payload.get("concerns"), list):
            return [], ""
        if isinstance(payload, list):
            return [], ""

    return None, f"no readable answer in {MAX_ATTEMPTS} attempts"


async def critique(
    llm_service: Any,
    *,
    artifact: str,
    goal: str,
    intent: str = "",
    lenses: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Attack a design from several angles and report what survives.

    A lens that never answers readably is reported as unreviewed rather than as
    clean: "nothing was found" and "we could not look" are opposite statements
    and only one of them is reassuring.
    """
    chosen = [name for name in (lenses or LENSES) if name in LENSES]
    concerns: List[Dict[str, Any]] = []
    unreviewed: List[Dict[str, str]] = []

    for name in chosen:
        prompt = _PROMPT.format(
            goal=(goal or "").strip()[:2000],
            intent=(intent or "not stated").strip()[:1000],
            artifact=(artifact or "").strip()[:12000],
            lens=LENSES[name],
        )
        raw, reason = await _ask_lens(llm_service, prompt, name)
        if raw is None:
            unreviewed.append({"lens": name, "reason": reason})
            continue
        for item in raw:
            cleaned = _clean(item, name)
            if cleaned:
                concerns.append(cleaned)

    # Unrated sorts with serious rather than last: it is unknown, not mild.
    order = {"blocking": 0, "serious": 1, "unrated": 1, "minor": 2}
    concerns.sort(key=lambda c: order.get(c["severity"], 3))
    return {
        "reviewed": bool(chosen) and not unreviewed,
        "lenses": chosen,
        "unreviewed_lenses": unreviewed,
        "concerns": concerns,
        "blocking": [c for c in concerns if c["severity"] == "blocking"],
        "unrated": [c for c in concerns if c["severity"] == "unrated"],
    }


def describe() -> List[str]:
    return [
        "an experiment's design is attacked before it is simulated, by a "
        "reviewer that did not write it -- the model that produced a design "
        "reasons from the assumptions that produced it and tends to confirm it",
        "a critique advises and does not gate: control flow stays "
        "deterministic, and what a contract can require is that the run "
        "answered the concerns, not that a model approved of it",
    ]
