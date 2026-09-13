"""What a pipeline author is allowed to ask for.

A contract names finding types, and there is exactly one set of them that a
tool can actually produce. Everything that authors a pipeline needs that set:
the studio's stage editor, so a person picks from what exists rather than
typing a plausible string, and the drafter, so a model is constrained by it
rather than inventing one.

Both used to guess. Typing an evidence type nobody produces is the single most
common way a pipeline fails its own check -- the contract reads perfectly and
no tool in the system can satisfy it -- and a language model asked for a
pipeline invents them at a rate no prompt talks it out of.

Derived from the tool specs, never restated. A second list of finding types
would drift from the tools the way every parallel list in this project has:
`agent_evidence_map` exists because exactly that happened, four counter tools
went missing from a hand-maintained map, and contracts asking for their
findings were told nothing produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.agent_core import tool_specs
from app.services import agent_evidence_map


@dataclass(frozen=True)
class EvidenceType:
    """One finding type an author may require, and what asking for it costs."""

    #: The name a contract writes in `required_finding_types`.
    name: str
    #: Every tool that can produce it. More than one is normal -- a paper can
    #: be ingested by arXiv search or by id -- and naming only the first would
    #: hide the route that actually suits a given stage.
    producers: Tuple[str, ...] = ()
    #: What the whole chain to obtain it costs, not just the last tool: asking
    #: for a benchmark means building the thing first, and an author choosing
    #: between two kinds of evidence is choosing between those chains.
    typical_seconds: int = 0
    #: Job types every producing tool permits. Empty means unrestricted.
    #: A stage requiring evidence whose tools are restricted to `coding` and
    #: left at the default `research` plans tools it then cannot call -- the
    #: checker refuses it, and this is what lets an editor prevent it instead.
    job_types: Tuple[str, ...] = ()
    #: Evidence a later change invalidates, so it is never inherited from an
    #: upstream stage. An author needs to know: a looping stage cannot keep a
    #: perishable verdict it earned before its own edit.
    perishable: bool = False
    #: What the producing tool consumes, in the tool's own words.
    consumes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "producers": list(self.producers),
            "typical_seconds": self.typical_seconds,
            "job_types": list(self.job_types),
            "perishable": self.perishable,
            "consumes": self.consumes,
        }


def _job_types_for(producers: Tuple[str, ...]) -> Tuple[str, ...]:
    """Job types under which EVERY producer of this evidence can run.

    The intersection rather than the union, because a stage runs under one job
    type and must be able to reach whichever producer the planner picks. A
    producer with no restriction constrains nothing and is skipped; if none of
    them restrict anything the answer is "unrestricted", which is empty.
    """
    by_name = {spec.name: spec for spec in tool_specs.all_specs()}
    allowed: List[frozenset] = []
    for tool in producers:
        spec = by_name.get(tool)
        restriction = getattr(spec, "job_types", None) if spec else None
        if restriction:
            allowed.append(frozenset(str(j) for j in restriction))
    if not allowed:
        return ()
    common = frozenset.intersection(*allowed)
    return tuple(sorted(common))


def evidence_types() -> List[EvidenceType]:
    """Every finding type a contract may require, alphabetically."""
    names = sorted(
        {produced for spec in tool_specs.all_specs() for produced in spec.produces}
    )
    out: List[EvidenceType] = []
    for name in names:
        producers = tuple(agent_evidence_map.producers_of(name))
        out.append(
            EvidenceType(
                name=name,
                producers=producers,
                # The chain, not the tool. See the field's docstring.
                typical_seconds=agent_evidence_map.estimate_chain_seconds([name]),
                job_types=_job_types_for(producers),
                perishable=agent_evidence_map.is_perishable(name),
                consumes=(
                    (agent_evidence_map.entry_for(producers[0]).consumes or "")
                    if producers
                    else ""
                ),
            )
        )
    return out


def job_types() -> List[str]:
    """Job types a stage may declare, in the order a author should consider.

    Read off the specs rather than off `AgentJobType`: the question here is
    which job types the tools distinguish between, and a job type no tool
    mentions is not a choice an author is making.
    """
    declared = {
        str(j)
        for spec in tool_specs.all_specs()
        for j in (getattr(spec, "job_types", None) or ())
    }
    # `research` is the default a stage gets when it says nothing, so it must
    # be offered even if every tool that restricts itself excludes it.
    declared.add("research")
    return sorted(declared)


def as_dict() -> Dict[str, Any]:
    """The whole vocabulary, for an editor or a prompt."""
    return {
        "evidence_types": [e.as_dict() for e in evidence_types()],
        "job_types": job_types(),
    }


def unknown_types(required: List[str]) -> List[str]:
    """Which of these finding types nothing can produce.

    The same question `agent_evidence_map.unobtainable` answers for a chain,
    asked about a bare list so an editor can mark a field red before anything
    is checked.
    """
    known = {e.name for e in evidence_types()}
    return [str(t) for t in required if str(t) not in known]
