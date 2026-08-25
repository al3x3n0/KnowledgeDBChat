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
    #: Seconds for one call that produced USABLE evidence, from runs recorded
    #: in this project. Not a floor: the first version used the smallest
    #: plausible figure, pronounced a 60-minute budget sufficient for a chain
    #: that had just failed to finish in 90, and was therefore useless for the
    #: case it was written for. Workload size sets the real cost and nothing
    #: here knows it, so these are order-of-magnitude and the check says so.
    typical_seconds: int = 0
    # Said in the tool's own terms, because "pattern" and "the instructions the
    # pattern came from" are easy to confuse and the difference is three
    # wasted attempts.
    consumes: str = ""


EVIDENCE_TOOLS: Tuple[ToolEvidence, ...] = (
    ToolEvidence(
        tool="compile_c_snippet",
        typical_seconds=10,  # a compile in a container that must start first
        produces=("codegen_measurement",),
        consumes="C source; returns the assembly the compiler really emitted.",
    ),
    ToolEvidence(
        tool="profile_c_workload",
        typical_seconds=60,  # instrumented execution, minutes on real code
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
        typical_seconds=10,  # llvm-mca is fast; the container is not
        produces=("cycle_model_measurement",),
        consumes="assembly fenced with # LLVM-MCA-BEGIN / # LLVM-MCA-END.",
    ),
    ToolEvidence(
        tool="benchmark_c_snippet",
        typical_seconds=30,  # repeated trials on the host
        produces=("benchmark_measurement",),
        consumes="a program that times itself; runs it on the real host.",
    ),
    ToolEvidence(
        tool="simulate_c_workload",
        typical_seconds=60,  # observed 60-190s across recorded bundles
        produces=("simulated_measurement",),
        consumes="a self-contained program; runs it in a modelled core.",
    ),
    ToolEvidence(
        tool="describe_model_parameters",
        produces=("model_parameters",),
        consumes="a core model name; returns its tunable parameters and paths.",
    ),
    ToolEvidence(
        tool="sample_hardware_counters",
        # 38 and 105 minutes, the only two traces this project has taken that
        # were long enough to estimate on. A trace cheap enough to sample in
        # ten minutes is too short for the estimator downstream, so the cheap
        # case is not the relevant one.
        typical_seconds=2400,
        produces=("counter_trace",),
        consumes=(
            "a self-contained program that calls M5_SAMPLE() wherever a sample "
            "should be taken. Without those calls it returns one total, which "
            "is not a trace: predictability is a property of counters over "
            "time and cannot be read from a run's totals."
        ),
    ),
    ToolEvidence(
        tool="measure_predictability",
        produces=("predictability_ceiling",),
        requires=("sample_hardware_counters",),
        consumes=(
            "the name of the counter to predict. The trace this run already "
            "sampled is read from the run -- do not paste it back."
        ),
    ),
    ToolEvidence(
        tool="select_counter_taps",
        produces=("counter_tap_selection",),
        # Not a hard refusal, a method: which counters to tap TOGETHER is worth
        # asking only once one has shown signal on its own, because the joint
        # estimate costs samples exponentially and a trace is finite.
        requires=("measure_predictability",),
        consumes="the counter to predict; returns which taps survive their own null.",
    ),
    ToolEvidence(
        tool="evaluate_predictor_design",
        produces=("predictor_design_result",),
        requires=("select_counter_taps",),
        consumes=(
            "the counter to predict and the tap to read alongside its own last "
            "value -- normally the one select_counter_taps recommended. An "
            "information ceiling says what is available; this says what a "
            "table of counters reaches."
        ),
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
        typical_seconds=30,  # an SMT solver, unbounded above
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


#: Where the methodology for a tool lives. The chain above says which tool
#: yields what and in what order; these say what makes the number wrong even
#: when the order is right, which is a different kind of knowledge and belongs
#: with the module that implements the check.
_METHOD_SOURCES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (
        ("sample_hardware_counters", "measure_predictability", "select_counter_taps"),
        "app.services.agent_predictability",
    ),
    (("evaluate_predictor_design",), "app.services.agent_predictor_design"),
)


def method_notes(required: Iterable[str]) -> List[str]:
    """Methodology for the work this run's contract actually implies.

    Keyed to the derived CHAIN rather than to the evidence types named, because
    a contract asking only for a design result still makes the run sample a
    trace and measure a ceiling on the way there -- and the traps it needs
    warning about are the traps of the tools it will run.

    Every module here already carried a `describe()` saying what makes its
    numbers worth having: that predictability is measured beyond persistence,
    that a time series is split contiguously and never at random. None of it
    reached a model. The only caller of any `describe()` was the validity
    block, so the notes belonging to modules that implement no validity
    predicate were written and then never read. This is the path that reads
    them.
    """
    import importlib

    tools = set(chain_for(required))
    lines: List[str] = []
    for owned, module_path in _METHOD_SOURCES:
        if not tools.intersection(owned):
            continue
        module = importlib.import_module(module_path)
        for line in module.describe():
            if line not in lines:
                lines.append(line)
    return lines


#: What one iteration costs before any tool runs: the model reads a large
#: prompt and reasons before answering. Measured at roughly half a minute per
#: decision on the provider this was written against.
ITERATION_OVERHEAD_SECONDS = 30

#: How many times a run actually calls an expensive tool. Not once: the live
#: run this was written from called the counter sampler twice, at 38 and 105
#: minutes, and neither call was wasted -- an agent refines a workload it has
#: seen the output of. Assuming one call per tool made the check agree that a
#: budget which had just expired was ample.
EXPENSIVE_TOOL_ATTEMPTS = 2

#: Above this a tool is expensive enough that a second attempt matters.
EXPENSIVE_SECONDS = 300


def estimate_chain_seconds(required: Iterable[str], iterations: int = 0) -> int:
    """A floor on what obtaining this evidence costs, in seconds.

    One call per cheap tool in the derived chain, two per expensive one, plus
    the model's own time per iteration. Order-of-magnitude by construction: the
    workload a run hands a simulator sets the real cost and nothing here knows
    it. Useful for "this budget is not in the right range", not for planning.
    """
    total = 0
    for tool in chain_for(required):
        entry = _BY_TOOL.get(tool)
        if not entry:
            continue
        cost = int(entry.typical_seconds or 0)
        if cost >= EXPENSIVE_SECONDS:
            cost *= EXPENSIVE_TOOL_ATTEMPTS
        total += cost
    return total + int(iterations or 0) * ITERATION_OVERHEAD_SECONDS


def check_runtime_budget(
    required: Iterable[str],
    max_runtime_minutes: int,
    max_iterations: int = 0,
) -> Dict[str, object]:
    """Whether this job can finish the chain its contract demands.

    The default budget is 60 minutes and a single counter-sampling call has
    been observed at 105. A job like that does not fail: it runs, produces some
    of its evidence, and stops with its contract unmet, which reads as an agent
    that gave up rather than a budget that expired. Saying so at launch costs
    nothing; discovering it costs the whole run.
    """
    required = [str(x).strip() for x in required if str(x).strip()]
    floor = estimate_chain_seconds(required, max_iterations)
    budget = int(max_runtime_minutes or 0) * 60
    feasible = budget <= 0 or floor <= budget

    breakdown = [
        {"tool": tool, "floor_seconds": int(_BY_TOOL[tool].typical_seconds or 0)}
        for tool in chain_for(required)
        if tool in _BY_TOOL and _BY_TOOL[tool].typical_seconds
    ]
    message = ""
    if not feasible:
        slowest = ", ".join(
            f"{row['tool']} (>={row['floor_seconds'] // 60} min)"
            for row in sorted(breakdown, key=lambda r: -r["floor_seconds"])[:3]
        )
        message = (
            f"This contract needs {', '.join(required[:6])}. Obtaining it has "
            f"taken about {floor // 60} minutes in runs recorded here -- "
            f"{slowest}, counted twice each because a run refines an expensive "
            f"call after seeing its output -- and the job is allowed "
            f"{max_runtime_minutes}. The figures are order-of-magnitude and "
            "the workload usually makes them larger, so treat this as the "
            "wrong range rather than a precise shortfall. Raise "
            "max_runtime_minutes, or require less evidence. A job that runs "
            "out does not fail: it stops with its contract unmet, which reads "
            "as an agent that gave up."
        )
    return {
        "feasible": feasible,
        "floor_seconds": floor,
        "budget_seconds": budget,
        "breakdown": breakdown,
        "message": message,
    }


def unobtainable(required: Iterable[str]) -> List[str]:
    """Required evidence no tool here knows how to produce.

    Worth saying out loud: a contract asking for something unobtainable cannot
    be satisfied however well the run behaves, and that is a fault in the
    contract rather than in the run.
    """
    return [
        str(finding_type) for finding_type in required if not producer_of(finding_type)
    ]
