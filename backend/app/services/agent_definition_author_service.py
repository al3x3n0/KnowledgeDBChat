"""Write an agent definition from a description of what it should do.

The hard part of defining an agent is not the prose. It is the two closed sets
either side of it: the capabilities that decide whether the router ever reaches
this agent, and the tool whitelist that decides what it can do once reached.
Both are easy to write plausibly and wrong, and both fail *quietly* -- a
capability outside the router's vocabulary never matches, so the agent is
simply never chosen, and a whitelist naming a tool that does not exist leaves
the agent with fewer tools than its author believes, with nothing anywhere
saying so.

So this drafts against the real checks rather than a description of them. The
schema that refuses a draft is ``AgentDefinitionCreate``, the same one the
create endpoint uses; the capability vocabulary is read from
``CAPABILITY_KEYWORDS`` where the router reads it; the tool names are checked
against the catalog the runtime builds its menu from. A refusal goes back to
the model verbatim, because a refusal that names what is wrong is exactly what
the next attempt needs.

Drafting never creates. The definition comes back for review with ``notes``
saying what had to be repaired, because a definition that validates is not the
same as one that does what was meant.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import ValidationError
from sqlalchemy import select

logger = logging.getLogger(__name__)

#: Three attempts: enough for the model to act on a refusal and on a second
#: one, few enough that a hopeless request fails while someone is still
#: watching.
MAX_ATTEMPTS = 3

DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "display_name": {"type": "string"},
        "description": {"type": "string"},
        "system_prompt": {"type": "string"},
        "capabilities": {"type": "array", "items": {"type": "string"}},
        "tool_whitelist": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "integer"},
    },
    "required": ["name", "display_name", "system_prompt", "capabilities"],
}


def vocabulary() -> Dict[str, List[str]]:
    """The closed sets a definition may draw on, read from where they live.

    Restating either here would be a copy to keep in step, and the first to
    drift would produce drafts refused for a reason the author cannot act on.
    """
    from app.agent_core import tool_specs
    from app.agent_core.routing import CAPABILITY_KEYWORDS

    return {
        "capabilities": sorted(CAPABILITY_KEYWORDS),
        "tools": sorted(tool_specs.STATIC_CATALOG.spec_names()),
    }


def _system_prompt() -> str:
    vocab = vocabulary()
    return (
        "You write agent definitions for an autonomous research platform.\n\n"
        "Return one JSON object with: name, display_name, description, "
        "system_prompt, capabilities, tool_whitelist, priority.\n\n"
        "name: lowercase letters, digits and underscores, starting with a "
        "letter, 2-100 characters.\n"
        "system_prompt: at least 10 characters; write the instructions the "
        "agent itself will be given, in the second person.\n"
        "priority: 1-100, where a higher number wins when several agents "
        "match. Use 50 unless the request implies otherwise.\n\n"
        "capabilities decide whether the router ever reaches this agent. Use "
        "ONLY these, and choose the ones a person's request would actually "
        "mention:\n" + ", ".join(vocab["capabilities"]) + "\n\n"
        "tool_whitelist decides what the agent may call once reached. Use ONLY "
        "these names. Omit the field entirely to allow every tool, which is "
        "usually right for a general agent and wrong for a narrow one:\n"
        + ", ".join(vocab["tools"])
        + "\n"
    )


def _payload(completion: Any) -> Dict[str, Any]:
    """The object out of a completion, whichever way the provider returned it.

    ``generate_structured`` hands back an ``LLMCompletion``, not a dict:
    providers with native schema output fill ``.structured``, the rest leave
    JSON in ``.text``, sometimes inside a fence. Treating the completion itself
    as a mapping is the quiet failure -- every field reads as missing, so the
    draft looks like a model that cannot follow instructions. Measured: the
    first live run of this drafter reported "the reply was not JSON" three
    times against a model that had answered correctly each time.
    """
    structured = getattr(completion, "structured", None)
    if isinstance(structured, Mapping) and structured:
        return dict(structured)
    if isinstance(completion, Mapping):
        return dict(completion)

    text = str(getattr(completion, "text", "") or completion or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # A model that wrapped the object in a sentence still answered.
        start_brace, end_brace = text.find("{"), text.rfind("}")
        if start_brace < 0 or end_brace <= start_brace:
            return {}
        try:
            parsed = json.loads(text[start_brace : end_brace + 1])
        except json.JSONDecodeError:
            return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def check(payload: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate a drafted definition the way creating one would.

    Returns the cleaned definition, or ``None`` with the complaints. The
    complaints are written to be handed straight back to the model: each says
    what is wrong and what would be right.
    """
    from app.agent_core import tool_specs
    from app.agent_core.routing import CAPABILITY_KEYWORDS
    from app.schemas.agent import AgentDefinitionCreate

    complaints: List[str] = []
    try:
        model = AgentDefinitionCreate(**dict(payload))
    except ValidationError as exc:
        for error in exc.errors():
            field = ".".join(str(p) for p in error.get("loc", ())) or "(root)"
            complaints.append(f"{field}: {error.get('msg')}")
        return None, complaints

    known_capabilities = set(CAPABILITY_KEYWORDS)
    unknown = [c for c in model.capabilities if c not in known_capabilities]
    if unknown:
        # Silent failure otherwise: the router matches on this vocabulary, so a
        # capability outside it is never matched and the agent is never chosen.
        complaints.append(
            f"capabilities {unknown} are not in the router's vocabulary, so the "
            "agent would never be routed to. Use only: "
            + ", ".join(sorted(known_capabilities))
        )

    if model.tool_whitelist is not None:
        known_tools = set(tool_specs.STATIC_CATALOG.spec_names())
        missing = [t for t in model.tool_whitelist if t not in known_tools]
        if missing:
            # Also silent: a whitelist is a filter, and a name matching nothing
            # removes a tool the author thought they were granting.
            complaints.append(
                f"tool_whitelist names {missing}, which are not tools. A "
                "whitelist is a filter, so these grant nothing and the agent "
                "ends up with fewer tools than intended."
            )
        if not model.tool_whitelist:
            complaints.append(
                "tool_whitelist is empty, which allows no tools at all. Omit "
                "the field to allow every tool."
            )

    if complaints:
        return None, complaints
    return model.model_dump(), []


async def _name_taken(name: str, db: Any) -> bool:
    from app.models.agent_definition import AgentDefinition

    result = await db.execute(
        select(AgentDefinition.id).where(AgentDefinition.name == name).limit(1)
    )
    return result.scalar_one_or_none() is not None


def _revision_message(text: str, current: Mapping[str, Any]) -> str:
    """Ask for a revision of what the author has, not a fresh invention.

    ``current`` is whatever is on the form, which is not necessarily the last
    thing the model produced: a person may have edited a field by hand between
    drafts, and refining from the model's own last answer would silently throw
    that away. So the form is the source of truth and the instruction is
    applied to it.
    """
    shown = {
        key: current.get(key)
        for key in (
            "name",
            "display_name",
            "description",
            "system_prompt",
            "capabilities",
            "tool_whitelist",
            "priority",
        )
        if current.get(key) not in (None, "", [])
    }
    return (
        "Revise this agent definition:\n\n"
        + json.dumps(shown, indent=2)
        + f"\n\nApply exactly this change:\n\n{text}\n\n"
        "Keep everything the change does not touch as it is, including the "
        "name unless the change asks for a different one. Return the whole "
        "definition."
    )


async def draft_definition(
    description: str,
    *,
    current: Optional[Mapping[str, Any]] = None,
    user_id: Any = None,
    db: Any = None,
) -> Dict[str, Any]:
    """Draft an agent definition, repairing it against the real checks.

    With ``current``, this is a revision rather than a fresh draft: the
    description is read as an instruction to apply to what the author already
    has. The same checks apply either way -- a refinement that introduces a
    capability the router does not know is refused exactly as a first draft
    would be, because "make it narrower" is no reason to accept an agent
    nothing can route to.

    Never creates anything: the caller reviews what comes back. ``notes`` says
    what had to be repaired, which is the part worth reading -- a draft that
    needed two attempts to name a real capability is a draft worth looking at
    twice.
    """
    from app.services.llm_service import LLMService

    text = str(description or "").strip()
    if not text:
        return {
            "definition": None,
            "notes": [
                "No change was described." if current else "No description was given."
            ],
        }

    llm = LLMService()
    system = _system_prompt()
    message = (
        _revision_message(text, current)
        if isinstance(current, Mapping) and current
        else f"Write an agent definition for this request:\n\n{text}"
    )
    notes: List[str] = []
    definition: Optional[Dict[str, Any]] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            completion = await llm.generate_structured(
                system_prompt=system,
                user_message=message,
                response_schema=DRAFT_SCHEMA,
                task_type="balanced",
                user_id=user_id,
                db=db,
            )
        except Exception as exc:  # pragma: no cover - network/provider failure
            logger.warning(f"Agent draft call failed on attempt {attempt}: {exc}")
            notes.append(f"The model could not be reached: {exc}")
            break

        payload = _payload(completion)
        if not payload:
            notes.append(f"Attempt {attempt}: the reply was not JSON.")
            message = (
                f"{message}\n\nYour last reply was not a JSON object. Reply "
                "with one JSON object and nothing else."
            )
            continue

        candidate, complaints = check(payload)
        if complaints:
            notes.append(f"Attempt {attempt}: " + "; ".join(complaints))
            message = (
                f"{message}\n\nYour last definition was rejected:\n"
                + "\n".join(complaints)
                + "\n\nFix exactly that and return the whole definition again."
            )
            continue

        keeping_its_own_name = bool(
            isinstance(current, Mapping)
            and candidate
            and candidate["name"] == str(current.get("name") or "")
        )
        if (
            db is not None
            and candidate
            and not keeping_its_own_name
            and await _name_taken(candidate["name"], db)
        ):
            # Caught here rather than at create: `name` is unique, and a draft
            # that collides is refused at the end of the work rather than the
            # start of it.
            taken = candidate["name"]
            notes.append(f"Attempt {attempt}: the name {taken!r} is already taken.")
            message = (
                f"{message}\n\nThe name {taken!r} already exists. Choose a "
                "different name and return the whole definition again."
            )
            continue

        definition = candidate
        break

    if definition is None and not notes:
        notes.append("No usable definition was produced.")
    return {"definition": definition, "notes": notes}


__all__ = ["DRAFT_SCHEMA", "check", "draft_definition", "vocabulary"]
