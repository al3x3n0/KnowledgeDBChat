"""What each tool needs before it can run, and what evidence it leaves behind.

A run given a goal and no method picks its tools well: asked whether any
sequence in a kernel was worth fusing, with no steps at all, one reached for
compile, profile, mine and cost unaided. What it could not do was order them or
say what passes between them. It called the miner before profiling anything,
learned the dependency from the refusal, and then spent three attempts handing
the costing tool raw assembly where a mined pattern belongs -- at one point
calling `list_custom_tools` to look for documentation.

None of that is a planning failure in the interesting sense. The chain was
right; the run simply had no way to know that a tool needs a profile before it
can mine, or that the thing to pass on is a pattern rather than the
instructions the pattern was found in. That is knowledge about the tools, and
the tools are where it should live.

So this maps evidence to the tool that produces it, and each tool to what it
needs first. From that a chain can be derived backwards from what a goal
contract demands -- which is planning, done deterministically, rather than
guessed and corrected one refusal at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ToolEvidence:
    """One tool: what it yields, what must come first, and how it is fed."""

    tool: str
    produces: Tuple[str, ...] = ()
    requires: Tuple[str, ...] = ()
    # Said in the tool's own terms, because "pattern" and "the instructions the
    # pattern came from" are easy to confuse and the difference is three
    # wasted attempts.
    consumes: str = ""


EVIDENCE_TOOLS: Tuple[ToolEvidence, ...] = (
    ToolEvidence(
        tool="compile_c_snippet",
        produces=("codegen_measurement",),
        consumes="C source; returns the assembly the compiler really emitted.",
    ),
    ToolEvidence(
        tool="profile_c_workload",
        produces=("dynamic_profile",),
        consumes="a self-contained program; runs it and counts what executed.",
    ),
    ToolEvidence(
        tool="find_fusion_candidates",
        produces=("fusion_candidate",),
        requires=("profile_c_workload",),
        consumes=(
            "the hot blocks of a profile taken in this run -- leave `blocks` "
            "out and the most recent profile is used."
        ),
    ),
    ToolEvidence(
        tool="cost_fusion_candidate",
        produces=("fusion_cost_bound",),
        requires=("find_fusion_candidates",),
        consumes=(
            "the `pattern` string of a candidate, e.g. 'fsqrt fdiv | 0>1' -- "
            "the shape, not the instructions it was found in."
        ),
    ),
    ToolEvidence(
        tool="analyze_snippet_cycles",
        produces=("cycle_model_measurement",),
        consumes="assembly fenced with # LLVM-MCA-BEGIN / # LLVM-MCA-END.",
    ),
    ToolEvidence(
        tool="benchmark_c_snippet",
        produces=("benchmark_measurement",),
        consumes="a program that times itself; runs it on the real host.",
    ),
    ToolEvidence(
        tool="simulate_c_workload",
        produces=("simulated_measurement",),
        consumes="a self-contained program; runs it in a modelled core.",
    ),
    ToolEvidence(
        tool="describe_model_parameters",
        produces=("model_parameters",),
        consumes="a core model name; returns its tunable parameters and paths.",
    ),
    ToolEvidence(
        tool="record_prediction",
        produces=("prediction_recorded",),
        consumes=(
            "a number, and `derived_from` naming finding types this run has "
            "already produced."
        ),
    ),
    ToolEvidence(
        tool="record_measurement",
        produces=("prediction_settled",),
        requires=("record_prediction",),
        consumes="the prediction_id returned by record_prediction, and the measured value.",
    ),
    ToolEvidence(
        tool="record_method",
        produces=("method_recorded",),
        consumes="the procedure, what it prevents, and the findings establishing it.",
    ),
    ToolEvidence(
        tool="axis_check",
        produces=("axis_description",),
        consumes="an .axisl description of an instruction.",
    ),
    ToolEvidence(
        tool="axis_prove",
        produces=("equivalence_proof",),
        requires=("axis_check",),
        consumes="an .axisl description and the sequence it should be equivalent to.",
    ),
)

_BY_TOOL: Dict[str, ToolEvidence] = {entry.tool: entry for entry in EVIDENCE_TOOLS}
_BY_EVIDENCE: Dict[str, ToolEvidence] = {
    produced: entry for entry in EVIDENCE_TOOLS for produced in entry.produces
}


def producer_of(finding_type: str) -> str:
    """The tool that yields this kind of evidence, or empty if none does."""
    entry = _BY_EVIDENCE.get(str(finding_type).strip())
    return entry.tool if entry else ""


def chain_for(required: Iterable[str]) -> List[str]:
    """An order of tools that produces every finding type asked for.

    Derived from the requirements rather than recited: a contract that wants a
    settled prediction gets record_prediction before record_measurement because
    one needs the other, not because a list said so.
    """
    ordered: List[str] = []

    def add(tool: str) -> None:
        entry = _BY_TOOL.get(tool)
        if entry is None or tool in ordered:
            return
        for prerequisite in entry.requires:
            add(prerequisite)
        if tool not in ordered:
            ordered.append(tool)

    for finding_type in required:
        producer = producer_of(finding_type)
        if producer:
            add(producer)
    return ordered


def describe_chain(required: Sequence[str]) -> List[str]:
    """Lines telling a run how to obtain the evidence its contract demands."""
    chain = chain_for(required)
    if not chain:
        return []
    lines: List[str] = []
    for tool in chain:
        entry = _BY_TOOL[tool]
        produced = ", ".join(entry.produces)
        after = f" after {', '.join(entry.requires)}" if entry.requires else ""
        detail = f" Takes {entry.consumes}" if entry.consumes else ""
        lines.append(f"{tool} yields {produced}{after}.{detail}")
    return lines


def unobtainable(required: Iterable[str]) -> List[str]:
    """Required evidence no tool here knows how to produce.

    Worth saying out loud: a contract asking for something unobtainable cannot
    be satisfied however well the run behaves, and that is a fault in the
    contract rather than in the run.
    """
    return [
        str(finding_type) for finding_type in required if not producer_of(finding_type)
    ]
