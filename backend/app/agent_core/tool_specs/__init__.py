"""One declaration per tool, and every registry reading it.

Defining a tool used to mean editing four files that knew nothing about each
other: the schema a model reads (``agent_tools.AGENT_TOOLS``), the governance
metadata (``agent_core.tool_catalog``), the per-job-type allowlist
(``agent_job_tool_policy``), and — for measurement tools — the evidence map a
plan is derived from (``agent_evidence_map``). Those four were the four
most-changed files in the repository; sixteen commits in a year touched all
four of them.

Nothing failed when one was missed, which is why it kept happening. A tool
absent from a registry is not a broken tool, it is a quieter one: unadvertised,
ungoverned, or believed to produce no evidence. Three defects found in a single
day came from exactly that.

The declarations live in the domain modules beside this one, and every registry
derives from them. The handler stays in ``agent_tool_dispatch`` — that is code
rather than data, and ``tests/test_tool_specs.py`` asserts every spec has one.

Adding a tool is now: write the handler, write the spec.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from app.agent_core.tool_specs import (
    agent_ops,
    authoring,
    data_analysis,
    documents,
    execution,
    knowledge_graph,
    measurement,
    memory,
    orchestration,
    research,
)
from app.agent_core.tool_specs.spec import ToolSpec

#: Domain modules, in the order their tools are offered to a model. Grouping
#: related tools together is deliberate: the listing is read as a menu, and a
#: model choosing between siblings does better when they sit together.
_MODULES = (
    documents,
    knowledge_graph,
    research,
    authoring,
    execution,
    memory,
    orchestration,
    agent_ops,
    data_analysis,
    measurement,
)

TOOL_SPECS: Tuple[ToolSpec, ...] = tuple(
    spec for module in _MODULES for spec in module.SPECS
)

_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}

#: Which domain module declared each tool -- another view of the same
#: declarations, not a second list to keep in step.
#:
#: There was already a `tool_family` in agent_tool_scoring that guesses from
#: the name: prefixes like `search_`, tokens like `chart`. It answers "other"
#: for every measurement tool, because none of them is named after what it
#: does to a document. The module a tool is declared in knows without guessing.
TOOL_DOMAINS: Dict[str, str] = {
    spec.name: module.__name__.rsplit(".", 1)[-1]
    for module in _MODULES
    for spec in module.SPECS
}

if len(_BY_NAME) != len(TOOL_SPECS):
    seen, duplicated = set(), set()
    for spec in TOOL_SPECS:
        (duplicated if spec.name in seen else seen).add(spec.name)
    raise RuntimeError(f"tool declared in two modules: {sorted(duplicated)}")

__all__ = [
    "ToolSpec",
    "ToolCatalog",
    "TOOL_SPECS",
    "TOOL_DOMAINS",
    "STATIC_CATALOG",
    "tool_domain",
    "all_specs",
    "spec_for",
    "spec_names",
    "schemas",
    "tools_for_job_type",
]


class ToolCatalog:
    """The specs in force for one caller: the built-ins, plus what was added.

    Every registry in this application is a view of ``TOOL_SPECS``, which is
    computed once at import and frozen. That is right for tools declared in
    this repository and wrong for anything a *user* contributes: a plugin tool
    belongs to one user, and a module-level tuple has no idea who is asking.

    Rather than thread a user through the fifty-seven call sites that read the
    frozen views, the views stay exactly as they are -- they now delegate to
    ``STATIC_CATALOG``, which holds the built-ins and nothing else -- and a
    caller that *does* know whose tools it wants builds its own catalog with
    ``STATIC_CATALOG.extended_with(...)``.

    A dynamic spec may not shadow a built-in. Silently overriding
    ``run_repo_tests`` with a webhook would be a privilege-escalation route
    dressed as a feature, so the collision is refused here as well as at
    install time -- the second check costs nothing and this is the one that
    runs on every job.
    """

    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        self._specs: Tuple[ToolSpec, ...] = tuple(specs)
        self._by_name: Dict[str, ToolSpec] = {s.name: s for s in self._specs}

    def extended_with(self, specs: Iterable[ToolSpec]) -> "ToolCatalog":
        """This catalog plus ``specs``, refusing any that shadow a built-in."""
        added = []
        for spec in specs:
            if spec.name in self._by_name:
                raise ValueError(
                    f"tool {spec.name!r} already exists and may not be "
                    "overridden by a contributed tool"
                )
            added.append(spec)
        return ToolCatalog(self._specs + tuple(added))

    def all_specs(self) -> Tuple[ToolSpec, ...]:
        return self._specs

    def spec_for(self, tool_name: str) -> ToolSpec | None:
        return self._by_name.get(str(tool_name or "").strip())

    def spec_names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def schemas(self) -> List[Dict[str, Any]]:
        """Schema entries for every spec, in declaration order."""
        return [spec.schema() for spec in self._specs]

    def tools_for_job_type(self, job_type: str) -> List[str]:
        """Spec-declared tools this job type may call.

        ``job_types is None`` means every job type; an empty tuple means none,
        which is a real case rather than an omission — 58 tools are reachable
        from chat or MCP and from no autonomous job.
        """
        wanted = str(job_type or "").strip()
        return [
            spec.name
            for spec in self._specs
            if spec.job_types is None or wanted in spec.job_types
        ]


#: The built-in tools, and only those. Anything a user contributes extends a
#: copy of this rather than mutating it, so one user's plugin can never become
#: another user's tool.
STATIC_CATALOG = ToolCatalog(TOOL_SPECS)


def tool_domain(tool_name: str) -> str:
    """The domain module a tool was declared in, or "" if it has no spec."""
    return TOOL_DOMAINS.get(str(tool_name or "").strip(), "")


def all_specs() -> Tuple[ToolSpec, ...]:
    return STATIC_CATALOG.all_specs()


def spec_for(tool_name: str) -> ToolSpec | None:
    return STATIC_CATALOG.spec_for(tool_name)


def spec_names() -> frozenset[str]:
    return STATIC_CATALOG.spec_names()


def schemas() -> List[Dict[str, Any]]:
    """Schema entries for every built-in spec, in declaration order."""
    return STATIC_CATALOG.schemas()


def tools_for_job_type(job_type: str) -> List[str]:
    """Built-in tools this job type may call."""
    return STATIC_CATALOG.tools_for_job_type(job_type)
