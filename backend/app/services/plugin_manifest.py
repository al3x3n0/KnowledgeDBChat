"""What a plugin may declare, and why a given manifest is refused.

Validation happens at install, once, and says what is wrong in terms the
author can act on. That is deliberate and it is the same reason
``custom_tool_types.reject_custom_tool_type`` names the allowed set rather than
saying "invalid": a contributor told only that something failed will guess
again, and the guess is usually the same shape as the first attempt.

The envelope is validated even for parts this release does not yet render --
``nav``, ``views``, ``panels`` -- so that a manifest written today against the
documented shape keeps working when those arrive, and a manifest that got the
shape wrong is told now rather than after the feature ships.

Nothing here is a security boundary on its own. A manifest is a claim; the
classification a contributed tool actually runs under is derived from its
executor in ``agent_core.plugin_specs``, not from anything the author wrote.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from app.agent_core import tool_specs
from app.agent_core.plugin_specs import contributed_tool_name, reject_contributed_name
from app.services.custom_tool_types import allowed_custom_tool_types
from app.services.plugin_flows import FlowContributionError, validate_workflows
from app.services.plugin_ui import UiContributionError, validate_ui

#: A slug is lowercase so a tool's full name is predictable from the manifest,
#: and short because it is spent out of a 64-character function-name budget.
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+([-+][0-9A-Za-z.\-]+)?$")


#: Job types a contributed tool may be offered to. Taken from the built-in
#: declarations rather than restated, so a job type this application gains is
#: one a plugin may immediately target.
def known_job_types() -> frozenset[str]:
    types: set[str] = set()
    for spec in tool_specs.all_specs():
        if spec.job_types:
            types.update(spec.job_types)
    return frozenset(types)


#: Contribution keys the manifest may carry. Unknown keys are refused rather
#: than ignored: a typo in `contributes` is otherwise a plugin that installs
#: cleanly and does nothing, which is the most expensive kind of failure to
#: diagnose.
CONTRIBUTION_KEYS = frozenset({"tools", "nav", "views", "panels", "workflows"})


class ManifestError(ValueError):
    """A manifest that cannot be installed, with the reason."""


def _require_mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object, got {type(value).__name__}")
    return value


def _validate_tool(
    entry: Any, *, slug: str, index: int, reserved: Iterable[str], seen: set
) -> Dict[str, Any]:
    tool = _require_mapping(entry, f"contributes.tools[{index}]")

    name = str(tool.get("name") or "").strip()
    if not name:
        raise ManifestError(f"contributes.tools[{index}] has no name")

    full = contributed_tool_name(slug, name)
    problem = reject_contributed_name(full, reserved=reserved)
    if problem:
        raise ManifestError(
            f"contributes.tools[{index}] ({name!r}) cannot be offered: {problem}"
        )
    if full in seen:
        raise ManifestError(
            f"contributes.tools[{index}] ({name!r}) is declared twice in this manifest"
        )
    seen.add(full)

    tool_type = str(tool.get("tool_type") or "").strip().lower()
    allowed = allowed_custom_tool_types()
    if tool_type not in allowed:
        # Naming the allowed set, and saying when a type exists but is gated,
        # are different pieces of information and the author needs both.
        gated = ""
        if tool_type in {"docker_container"}:
            gated = (
                " That type exists but is disabled in this deployment "
                "(CUSTOM_TOOL_DOCKER_ENABLED)."
            )
        raise ManifestError(
            f"contributes.tools[{index}] ({name!r}) has tool_type {tool_type!r}; "
            f"must be one of: {', '.join(sorted(allowed))}.{gated}"
        )

    schema = tool.get("parameters_schema")
    if schema is not None and not isinstance(schema, dict):
        raise ManifestError(
            f"contributes.tools[{index}] ({name!r}) parameters_schema must be "
            "an object"
        )

    job_types = tool.get("job_types")
    if job_types is not None:
        if not isinstance(job_types, list) or not all(
            isinstance(j, str) for j in job_types
        ):
            raise ManifestError(
                f"contributes.tools[{index}] ({name!r}) job_types must be a list "
                "of strings"
            )
        unknown = sorted(set(job_types) - known_job_types())
        if unknown:
            raise ManifestError(
                f"contributes.tools[{index}] ({name!r}) names unknown job "
                f"type(s) {', '.join(unknown)}; known types are: "
                f"{', '.join(sorted(known_job_types()))}"
            )

    config = tool.get("config")
    if config is not None and not isinstance(config, dict):
        raise ManifestError(
            f"contributes.tools[{index}] ({name!r}) config must be an object"
        )

    return {
        "name": name,
        "description": str(tool.get("description") or "").strip(),
        "tool_type": tool_type,
        "parameters_schema": schema or {},
        "config": config or {},
        "job_types": list(job_types or []),
    }


def validate_manifest(
    raw: Any, *, reserved: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """Return the normalized manifest, or raise ``ManifestError`` saying why not.

    ``reserved`` defaults to the built-in tool names, which is what a
    contributed tool may never shadow.
    """
    manifest = _require_mapping(raw, "manifest")
    reserved = tool_specs.spec_names() if reserved is None else reserved

    slug = str(manifest.get("id") or manifest.get("slug") or "").strip().lower()
    if not SLUG_PATTERN.match(slug):
        raise ManifestError(
            f"id {slug!r} must be 1-32 characters, start with a letter, and use "
            "only lowercase letters, digits and underscore -- it becomes part "
            "of every tool name this plugin contributes"
        )

    name = str(manifest.get("name") or "").strip()
    if not name:
        raise ManifestError("name is required")

    version = str(manifest.get("version") or "0.1.0").strip()
    if not VERSION_PATTERN.match(version):
        raise ManifestError(f"version {version!r} must look like 1.2.3")

    contributes = _require_mapping(manifest.get("contributes") or {}, "contributes")
    unknown = sorted(set(contributes) - CONTRIBUTION_KEYS)
    if unknown:
        raise ManifestError(
            f"contributes has unknown key(s) {', '.join(unknown)}; supported "
            f"keys are: {', '.join(sorted(CONTRIBUTION_KEYS))}"
        )

    raw_tools = contributes.get("tools") or []
    if not isinstance(raw_tools, list):
        raise ManifestError("contributes.tools must be a list")

    seen: set = set()
    tools: List[Dict[str, Any]] = [
        _validate_tool(entry, slug=slug, index=i, reserved=reserved, seen=seen)
        for i, entry in enumerate(raw_tools)
    ]

    # A plugin that contributes nothing installs cleanly and does nothing,
    # which is worth saying out loud rather than discovering later.
    if not tools and not any(
        contributes.get(key) for key in ("nav", "views", "panels", "workflows")
    ):
        raise ManifestError(
            "this manifest contributes nothing: declare at least one entry "
            f"under one of {', '.join(sorted(CONTRIBUTION_KEYS))}"
        )

    try:
        ui = validate_ui(contributes, tools=tools)
        flows = validate_workflows(contributes, slug=slug, tools=tools)
    except (UiContributionError, FlowContributionError) as exc:
        # Same class of refusal, so a caller catching ManifestError sees every
        # reason a manifest was rejected rather than two kinds of failure.
        raise ManifestError(str(exc))

    return {
        "id": slug,
        "name": name,
        "version": version,
        "description": str(manifest.get("description") or "").strip(),
        "contributes": {
            "tools": tools,
            "nav": ui["nav"],
            "views": ui["views"],
            "panels": ui["panels"],
            "workflows": flows,
        },
        "settings": manifest.get("settings") or {},
    }
