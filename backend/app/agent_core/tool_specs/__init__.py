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

from typing import Any, Dict, List, Tuple

from app.agent_core.tool_specs import (
    agent_ops,
    authoring,
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
    measurement,
)

TOOL_SPECS: Tuple[ToolSpec, ...] = tuple(
    spec for module in _MODULES for spec in module.SPECS
)

_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}

if len(_BY_NAME) != len(TOOL_SPECS):
    seen, duplicated = set(), set()
    for spec in TOOL_SPECS:
        (duplicated if spec.name in seen else seen).add(spec.name)
    raise RuntimeError(f"tool declared in two modules: {sorted(duplicated)}")

__all__ = [
    "ToolSpec",
    "TOOL_SPECS",
    "all_specs",
    "spec_for",
    "spec_names",
    "schemas",
    "tools_for_job_type",
]


def all_specs() -> Tuple[ToolSpec, ...]:
    return TOOL_SPECS


def spec_for(tool_name: str) -> ToolSpec | None:
    return _BY_NAME.get(str(tool_name or "").strip())


def spec_names() -> frozenset[str]:
    return frozenset(_BY_NAME)


def schemas() -> List[Dict[str, Any]]:
    """Schema entries for every spec, in declaration order."""
    return [spec.schema() for spec in TOOL_SPECS]


def tools_for_job_type(job_type: str) -> List[str]:
    """Spec-declared tools this job type may call.

    ``job_types is None`` means every job type; an empty tuple means none,
    which is a real case rather than an omission — 58 tools are reachable from
    chat or MCP and from no autonomous job.
    """
    wanted = str(job_type or "").strip()
    return [
        spec.name
        for spec in TOOL_SPECS
        if spec.job_types is None or wanted in spec.job_types
    ]
