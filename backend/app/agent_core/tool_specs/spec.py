"""The shape of a tool declaration.

Kept apart from the declarations themselves so every domain module can import
it without importing its siblings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ToolSpec:
    """Everything the registries need to know about one tool.

    ``job_types`` empty means every job type may use it, which is what all of
    the tools below are: a measurement tool is not owned by a job type.
    """

    name: str
    description: str
    parameters: Dict[str, Any]

    # Governance. Defaults are the safe-sounding ones, so a spec that says
    # nothing is classified read-only and cheap -- state the opposite loudly.
    effects: str = "read"
    network: str = "none"
    cost_tier: str = "low"
    pii_risk: str = "low"

    #: Which job types may call it. ``None`` means every job type -- the
    #: common case for a measurement tool, which is not owned by one kind of
    #: job. An empty tuple means no job type offers it, which is different and
    #: real: 58 tools are reachable only from chat or MCP, and treating that
    #: as "all" would quietly hand every autonomous job a tool no policy ever
    #: meant it to have.
    job_types: Optional[Tuple[str, ...]] = None

    # Evidence: what a run gets from calling this, and what it needs first.
    # Only measurement tools carry these; an empty ``produces`` keeps the tool
    # out of the evidence map entirely, which is the honest default.
    produces: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    typical_seconds: int = 0
    consumes: str = ""

    def schema(self) -> Dict[str, Any]:
        """The shape ``AGENT_TOOLS`` holds, so a model sees no difference."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
