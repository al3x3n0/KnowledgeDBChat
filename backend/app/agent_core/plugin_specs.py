"""Turning a contributed tool into a tool the model can actually see.

A ``UserTool`` has always been callable by an agent, but only through
``run_custom_tool(tool_name, inputs)`` -- a single generic entry point. The
model therefore never saw a contributed tool in its own menu, with its own
name and its own parameter schema; to use one it first had to think to call
``list_custom_tools``, read the result, and then call the generic caller with
the name it found. Measured against the built-ins, which are offered by name
with their arguments described, that is a tool a model has to *discover* rather
than one it can *choose*.

Promoting a contributed tool to a real ``ToolSpec`` closes that gap. Everything
downstream -- the schema a model reads, the governance classification, the
job-type policy, the audit record -- already derives from ``ToolSpec``, so a
promoted tool is governed by exactly the machinery the built-ins are governed
by, and none of it had to learn what a plugin is.

Two things are deliberately *not* taken from the contributor.

**Governance is derived from the executor, not declared by the author.** A
manifest saying ``effects: read`` over a webhook that POSTs is a claim, and
believing it would let any contributor opt out of approval gates by typing a
word. The executor type is the thing that decides, because it is the thing that
actually runs.

**The name is validated, not trusted.** Tool names reach the provider as
function names, where they must match ``^[a-zA-Z0-9_-]{1,64}$``. ``UserTool.name``
has never been constrained beyond a length -- harmless while every call went
through ``run_custom_tool``, and a live failure the moment the name is offered
to a model. Existing tools with unusable names are skipped with a warning
rather than promoted, because breaking a user's agent runs to add a feature is
the wrong trade.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.agent_core.tool_specs.spec import ToolSpec

#: What a provider will accept as a function name.
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

#: Contributed tools are prefixed so they occupy a namespace of their own and
#: can never be mistaken for -- or shadow -- a built-in.
NAME_PREFIX = "p_"

#: How each executor type is classified, regardless of what its author said.
#:
#: ``effects`` follows what the executor can do to the world, and ``network``
#: whether it can reach off the host. A ``transform`` is the only genuinely
#: inert one: it renders a template over its inputs and returns the result.
EXECUTOR_GOVERNANCE: Dict[str, Dict[str, str]] = {
    # Speaks to an arbitrary URL with an arbitrary method.
    "webhook": {"effects": "write", "network": "external", "cost_tier": "low"},
    # Same, through a registered agent rather than a raw URL.
    "external_agent": {
        "effects": "write",
        "network": "external",
        "cost_tier": "medium",
    },
    # Jinja2/JSONPath over the inputs. Touches nothing.
    "transform": {"effects": "read", "network": "none", "cost_tier": "low"},
    # Runs code. The sandbox limits it; it is not read-only.
    "python": {"effects": "write", "network": "none", "cost_tier": "medium"},
    # Costs a model call, and the prompt can carry whatever it was given.
    "llm_prompt": {"effects": "read", "network": "external", "cost_tier": "medium"},
    # A container on the host daemon.
    "docker_container": {
        "effects": "write",
        "network": "external",
        "cost_tier": "high",
    },
    # Runs a whole workflow, which may do anything the workflow may do.
    "workflow_runner": {"effects": "write", "network": "external", "cost_tier": "high"},
}

#: Applied when the executor type is unknown. An unrecognised executor is the
#: case we know least about, so it is classified as the most dangerous rather
#: than the least -- the opposite of the ToolSpec defaults, which are lenient
#: because a built-in that says nothing was written in this repository.
UNKNOWN_GOVERNANCE: Dict[str, str] = {
    "effects": "write",
    "network": "external",
    "cost_tier": "high",
}


def contributed_tool_name(slug: str, tool_name: str) -> str:
    """The namespaced name a contributed tool is offered under."""
    return f"{NAME_PREFIX}{slug}_{tool_name}".strip()


def reject_contributed_name(
    name: str, *, reserved: Optional[Iterable[str]] = None
) -> Optional[str]:
    """Why this name cannot be offered to a model, or None if it can.

    Naming the rule that was broken matters: a caller told only that its choice
    was invalid will guess again, and the guess is usually the same shape.
    """
    candidate = str(name or "").strip()
    if not candidate:
        return "tool name is empty"
    if len(candidate) > 64:
        return (
            f"tool name is {len(candidate)} characters; a provider accepts at "
            "most 64"
        )
    if not NAME_PATTERN.match(candidate):
        bad = sorted({c for c in candidate if not re.match(r"[a-zA-Z0-9_-]", c)})
        return (
            f"tool name {candidate!r} contains {''.join(bad)!r}; a provider "
            "accepts only letters, digits, underscore and hyphen"
        )
    if reserved and candidate in set(reserved):
        return (
            f"tool name {candidate!r} is already a built-in tool and may not "
            "be overridden"
        )
    return None


def _parameters_of(tool: Any) -> Dict[str, Any]:
    """A JSON Schema object, whatever the stored value turned out to be.

    ``parameters_schema`` defaults to ``{}`` and is user-supplied, so it can be
    absent, null, or a bare property map with no envelope. A model handed
    ``{}`` as a parameter schema will call the tool with no arguments, which is
    a worse failure than being told the tool takes none.
    """
    raw = getattr(tool, "parameters_schema", None)
    if not isinstance(raw, dict) or not raw:
        return {"type": "object", "properties": {}}
    if raw.get("type") == "object" and isinstance(raw.get("properties"), dict):
        return raw
    # A bare property map: wrap it rather than reject it, since this is what
    # the existing editor produces for a simple tool.
    if "type" not in raw and all(isinstance(v, dict) for v in raw.values()):
        return {"type": "object", "properties": raw}
    return raw


def governance_for(tool_type: str) -> Dict[str, str]:
    """How a tool of this executor type is classified."""
    return dict(
        EXECUTOR_GOVERNANCE.get(str(tool_type or "").strip(), UNKNOWN_GOVERNANCE)
    )


def spec_from_user_tool(
    tool: Any,
    *,
    slug: str = "user",
    declared_name: Optional[str] = None,
    job_types: Optional[Tuple[str, ...]] = None,
    reserved: Optional[Iterable[str]] = None,
) -> ToolSpec:
    """The ``ToolSpec`` a contributed tool is offered as.

    ``declared_name`` is the name the author gave the tool, before namespacing.
    It is a parameter rather than always ``tool.name`` because a caller may
    already hold an object whose ``name`` is the *namespaced* one, and
    namespacing it twice produces ``p_bench_p_bench_note`` -- a name that is
    wrong in a way nothing downstream can detect, since it is still a valid
    function name for a tool that does not exist.

    Raises ``ValueError`` when the tool cannot be offered at all -- the caller
    decides whether that is a refusal (at install) or a skip (at resolution),
    and those are different: refusing at install protects the user from
    creating something unusable, while refusing at resolution would take down
    a job over a tool it was not going to call.
    """
    own = declared_name if declared_name is not None else getattr(tool, "name", "")
    name = contributed_tool_name(slug, str(own or "").strip())
    problem = reject_contributed_name(name, reserved=reserved)
    if problem:
        raise ValueError(problem)

    tool_type = str(getattr(tool, "tool_type", "") or "").strip()
    gov = governance_for(tool_type)

    described = str(getattr(tool, "description", "") or "").strip()
    # A model choosing between siblings reads the description; saying which
    # kind of thing this is costs one clause and prevents a webhook being
    # mistaken for a local computation.
    description = described or f"User-contributed {tool_type or 'custom'} tool."

    return ToolSpec(
        name=name,
        description=description,
        parameters=_parameters_of(tool),
        effects=gov["effects"],
        network=gov["network"],
        cost_tier=gov["cost_tier"],
        pii_risk="medium" if gov["network"] == "external" else "low",
        # ``None`` would mean every job type, which is not something a
        # contributor should be able to claim by omission. A contributed tool
        # is offered to the job types its plugin declares, and to none by
        # default.
        job_types=tuple(job_types or ()),
        produces=(),
        requires=(),
    )


def specs_from_user_tools(
    tools: Iterable[Any],
    *,
    slug: str = "user",
    job_types: Optional[Tuple[str, ...]] = None,
    reserved: Optional[Iterable[str]] = None,
) -> Tuple[List[ToolSpec], List[Tuple[str, str]]]:
    """Promote what can be promoted; report what could not, and why.

    Returns ``(specs, skipped)`` where ``skipped`` pairs the tool's own name
    with the reason. Resolution never raises: a single unusable tool must not
    cost a job every other tool it was given.
    """
    specs: List[ToolSpec] = []
    skipped: List[Tuple[str, str]] = []
    for tool in tools:
        own_name = str(getattr(tool, "name", "") or "").strip()
        try:
            specs.append(
                spec_from_user_tool(
                    tool, slug=slug, job_types=job_types, reserved=reserved
                )
            )
        except ValueError as exc:
            skipped.append((own_name, str(exc)))
    return specs, skipped
