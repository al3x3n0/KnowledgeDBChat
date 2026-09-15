"""Drafting a plugin manifest from a description, and checking it before anyone sees it.

Writing a manifest by hand means knowing five closed vocabularies -- executor
types, view kinds, doors, icons, panel slots -- and one open one, the shape a
tool's output happens to have. A model is good at the first five if it is told
them, and *cannot* know the sixth, because the output shape only exists once
the tool runs.

So this does two things rather than one.

**It repairs against the real validator.** A draft goes through
`validate_manifest`, and when that refuses, the reason goes back to the model
and it tries again. The messages were written to be actionable for a human
author -- they name the allowed set, they say which view a nav entry pointed
at and which ones exist -- which makes them an unusually good repair signal.
Nothing here re-implements a rule; the thing that decides at install is the
thing that decides here.

**It runs the tool it wrote.** A view declares a dotted `path` into its tool's
output, and a path that misses renders an empty table -- which is
indistinguishable from a tool that had nothing to say, and is the mistake I
made by hand the first time I wrote one of these. So a drafted `transform` is
executed and its declared path resolved against the real output; when it
misses, the actual shape goes back to the model.

Only `transform` tools are dry-run. They render a template over their inputs
and touch nothing else -- the one executor where running an *unreviewed* draft
costs nothing. A webhook would reach the network and an `llm_prompt` would
spend a model call, neither of which should happen because somebody typed a
sentence into a box.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from loguru import logger

from app.agent_core.plugin_specs import EXECUTOR_GOVERNANCE
from app.services.custom_tool_types import allowed_custom_tool_types
from app.services.plugin_manifest import (
    ManifestError,
    known_job_types,
    validate_manifest,
)
from app.services.plugin_ui import ICONS, NAV_DOORS, PANEL_SLOTS, VIEW_KINDS

#: How many times the model may be handed its own refusal and asked again.
#: Three is enough for the mistakes that actually happen -- a wrong door id, a
#: nav entry pointing at a renamed view, a path off by one level -- and a
#: fourth has never been observed to fix what a third could not.
MAX_ATTEMPTS = 3

#: The executor whose output can be discovered by running it, safely.
DRY_RUN_TYPES = frozenset({"transform"})


DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Short lowercase slug, letters/digits/underscore.",
        },
        "name": {"type": "string"},
        "version": {"type": "string"},
        "description": {"type": "string"},
        "contributes": {
            "type": "object",
            "properties": {
                "tools": {"type": "array", "items": {"type": "object"}},
                "views": {"type": "object"},
                "nav": {"type": "array", "items": {"type": "object"}},
                "panels": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
    "required": ["id", "name", "contributes"],
}


def vocabulary() -> Dict[str, List[str]]:
    """The closed sets a manifest may draw on, read from where they are defined.

    Restating these in the prompt would be a sixth copy to keep in step, and
    the first one to drift would produce drafts the validator refuses for a
    reason the author cannot act on.
    """
    return {
        "tool_types": sorted(allowed_custom_tool_types()),
        "read_only_tool_types": sorted(
            t
            for t in allowed_custom_tool_types()
            if EXECUTOR_GOVERNANCE.get(t, {}).get("effects") == "read"
        ),
        "view_kinds": list(VIEW_KINDS),
        "doors": list(NAV_DOORS),
        "icons": list(ICONS),
        "panel_slots": list(PANEL_SLOTS),
        "job_types": sorted(known_job_types()),
    }


def _system_prompt() -> str:
    vocab = vocabulary()
    return f"""You write plugin manifests for a research and knowledge platform.

A plugin is one installable unit. It may contribute tools an autonomous agent
job can call by name, and views rendered as pages or panels in the interface.

Return ONE JSON object with: id, name, version, description, contributes.

`contributes.tools` — each has name (lowercase, underscores), description,
tool_type, parameters_schema (a JSON Schema object), config, job_types.
  tool_type must be one of: {', '.join(vocab['tool_types'])}
  job_types must be from: {', '.join(vocab['job_types'])}
  - transform: renders a Jinja2 template over its inputs.
    config = {{"template": "..."}}. Touches nothing. Prefer this.
  - webhook: config = {{"url": "...", "method": "GET"}}. Reaches the network.
  - python: config = {{"code": "..."}}. Sandboxed.
  - llm_prompt: config = {{"prompt": "..."}}. Costs a model call.

`contributes.views` — a map of view id to a view.
  kind must be one of: {', '.join(vocab['view_kinds'])}
  A view with a `source` names one of THIS manifest's own tools by its short
  name, and that tool MUST be read-only: {', '.join(vocab['read_only_tool_types'])}.
  A view may NOT name a built-in tool or another plugin's tool.
  `path` is a dotted path into the tool's OUTPUT where the data lives.
  table and stats and detail need `columns` (or `fields`) and a `source`.
  markdown needs either `text` or a `source`.
  A view has no request to take arguments from, so `source.params` MUST supply
  every input the tool marks `required`. If a view would have nothing sensible
  to pass, give the tool no required inputs instead.

`contributes.nav` — [{{door, name, view, icon}}].
  door must be one of: {', '.join(vocab['doors'])}
  icon must be one of: {', '.join(vocab['icons'])}
  view must be a view id this manifest declares.

`contributes.panels` — [{{slot, view, title}}].
  slot must be one of: {', '.join(vocab['panel_slots'])}

Rules that will get a manifest rejected:
- an `id` with a hyphen or capital: lowercase letters, digits and underscore only
- naming a door by its label ("R&D") instead of its id ("rnd")
- a view whose `source.params` omits an input its tool requires
- a nav entry or panel pointing at a view id you did not declare
- backing a view with a webhook or python tool (those can write)
- a tool name with spaces or punctuation

Keep it small. One or two tools and one view is a good plugin. Output JSON
only, no prose and no code fences."""


def _payload(completion: Any) -> Dict[str, Any]:
    """The object out of a completion, whichever way the provider returned it.

    `generate_structured` hands back an LLMCompletion, not a dict: providers
    with native schema output fill `.structured`, the rest leave JSON in
    `.text`, sometimes fenced. Treating the completion itself as a mapping is
    the quiet failure -- every field reads as missing and the draft silently
    becomes empty.
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
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _resolve(data: Any, path: str) -> Tuple[bool, str]:
    """Walk a dotted path, and say where it stopped. Mirrors the renderer."""
    trimmed = (path or "").strip()
    if not trimmed:
        return True, ""
    current = data
    walked: List[str] = []
    for segment in trimmed.split("."):
        if not segment:
            continue
        if not isinstance(current, Mapping) or segment not in current:
            walked.append(segment)
            return False, ".".join(walked[:-1]) or "(root)"
        current = current[segment]
        walked.append(segment)
    return True, ""


def _shape(value: Any, depth: int = 0) -> str:
    """A compact description of what a tool actually returned.

    Three levels, not two. This string is the repair signal a model uses to fix
    a wrong path, and a row's *fields* usually sit exactly three deep --
    `{items: [{name: str}]}`. Truncating at two yields `{items: [{...}]}`,
    which says a list of objects exists and nothing about what to read from
    them, so the retry has to guess the very thing it got wrong.
    """
    if isinstance(value, Mapping):
        if depth >= 3:
            return "{...}"
        inner = ", ".join(
            f"{k}: {_shape(v, depth + 1)}" for k, v in list(value.items())[:8]
        )
        return "{" + inner + "}"
    if isinstance(value, list):
        return f"[{_shape(value[0], depth + 1)}]" if value else "[]"
    return type(value).__name__


async def _dry_run_paths(manifest: Mapping[str, Any], user: Any, db: Any) -> List[str]:
    """Run the drafted transforms and check each view's path resolves.

    Returns a list of complaints, empty when every path lands. Never raises:
    a tool that fails to run is a fact about the draft, not a reason to lose
    the draft.
    """
    from app.services.custom_tool_service import CustomToolService
    from app.services.plugin_registry import ContributedTool

    contributes = manifest.get("contributes") or {}
    by_name = {t["name"]: t for t in contributes.get("tools") or [] if t.get("name")}
    complaints: List[str] = []

    for view_id, view in (contributes.get("views") or {}).items():
        source = view.get("source")
        if not source:
            continue
        tool = by_name.get(source.get("tool"))
        if not tool or tool.get("tool_type") not in DRY_RUN_TYPES:
            continue

        holder = ContributedTool(
            id=f"draft:{tool['name']}",
            name=tool["name"],
            description=str(tool.get("description") or ""),
            tool_type=tool["tool_type"],
            config=dict(tool.get("config") or {}),
            parameters_schema=dict(tool.get("parameters_schema") or {}),
            plugin_slug=str(manifest.get("id") or "draft"),
            declared_name=tool["name"],
        )
        try:
            result = await CustomToolService().execute_tool(
                tool=holder,
                inputs=dict(source.get("params") or {}),
                user=user,
                db=db,
                bypass_approval_gate=True,
            )
        except Exception as exc:
            complaints.append(
                f"view {view_id!r}: its tool {tool['name']!r} failed to run: {exc}"
            )
            continue

        output = result.get("output")
        ok, stopped = _resolve(output, str(view.get("path") or ""))
        if not ok:
            complaints.append(
                f"view {view_id!r} reads path {view.get('path')!r}, but running "
                f"{tool['name']!r} returned {_shape(output)} and the path stops "
                f"at {stopped!r}. Use the path that actually reaches the data."
            )
    return complaints


async def draft_manifest(
    description: str,
    *,
    user: Any = None,
    db: Any = None,
    user_id: Any = None,
) -> Dict[str, Any]:
    """Draft a manifest, repair it against the validator, and check its paths.

    Returns ``{manifest, notes, attempts}``. ``manifest`` is None when no
    attempt produced something installable; ``notes`` then says what the
    validator kept refusing, which is more useful than a bare failure.
    """
    from app.services.llm_service import LLMService

    text = str(description or "").strip()
    if not text:
        return {
            "manifest": None,
            "notes": ["Describe what the plugin should do."],
            "attempts": 0,
        }

    llm = LLMService()
    system = _system_prompt()
    notes: List[str] = []
    message = f"Write a plugin manifest for this request:\n\n{text}"
    manifest: Optional[Dict[str, Any]] = None

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
        except Exception as exc:
            logger.warning(f"Plugin draft call failed on attempt {attempt}: {exc}")
            notes.append(f"The model could not be reached: {exc}")
            break

        payload = _payload(completion)
        if not payload:
            message = (
                f"{message}\n\nYour last reply was not a JSON object. Reply with "
                "one JSON object and nothing else."
            )
            notes.append(f"Attempt {attempt}: the reply was not JSON.")
            continue

        try:
            manifest = validate_manifest(payload)
        except ManifestError as exc:
            # The refusal names what is wrong and what would be right, which
            # is exactly what the next attempt needs.
            notes.append(f"Attempt {attempt}: {exc}")
            message = (
                f"{message}\n\nYour last manifest was rejected:\n{exc}\n\n"
                "Fix exactly that and return the whole manifest again."
            )
            manifest = None
            continue

        if user is None or db is None:
            break

        complaints = await _dry_run_paths(manifest, user, db)
        if not complaints:
            break

        notes.append(f"Attempt {attempt}: " + "; ".join(complaints))
        message = (
            f"{message}\n\nThe manifest validates, but running its tools showed:\n"
            + "\n".join(complaints)
            + "\n\nFix the paths and return the whole manifest again."
        )
        # Keep the manifest: a wrong path is a flaw a person can see and fix in
        # review, and handing back nothing would be worse than handing back
        # something imperfect with the flaw written down.

    return {"manifest": manifest, "notes": notes, "attempts": attempt}
