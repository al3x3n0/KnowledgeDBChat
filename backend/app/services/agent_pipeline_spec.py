"""A research pipeline stated as postconditions, checked before it is run.

A pipeline here is a DAG of stages. A stage is not a list of steps: it is a
goal contract, which says what must be true when the stage is done. The tools
that get there are derived from the contract by ``agent_evidence_map``, the
same way a single job's chain already is -- planning as deduction rather than
recitation, so a stage cannot quietly be authored against a tool that does not
exist.

Everything in this module happens *before* anything expensive runs. That is
the point of it. The failures this project has actually paid for are not
crashes; they are runs that went the distance and produced something that
could not be used:

  - a contract asking for evidence no tool produces, which no amount of good
    behaviour can satisfy
  - a budget that sounded generous and was two orders of magnitude short
  - a stage built on a measurement the stage before it never took
  - a loop with no bound, which is the one failure an autonomous pipeline
    cannot recover from on its own

Each of those is decidable from the spec alone, so each is refused here rather
than discovered in hour three. What this module deliberately does NOT do is
execute anything: chaining, checkpoints and the agent loop already exist, and a
spec that cannot even be compiled has no business reaching them.

Pricing is pessimistic on purpose. ``estimate_chain_seconds`` records why: its
first version used the smallest plausible figure, pronounced a 60-minute budget
sufficient for a chain that had just failed to finish in 90, and was therefore
useless for the one case it was written for. A loop is priced at its maximum
iterations, not its hoped-for ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.services import agent_evidence_map as evidence

#: How a loop is allowed to decide it is finished.
#:
#: ``contract_satisfied`` repeats the stage until its own contract passes.
#: ``no_new_findings`` repeats until a number of consecutive rounds produce
#: nothing new -- for discovery, where the finish line is not a count but the
#: absence of anything further, and a simple ``while count < n`` stops in the
#: middle of the tail.
LOOP_CONDITIONS = ("contract_satisfied", "no_new_findings")

#: A loop must say where it stops. This is the ceiling on what it may claim,
#: not a recommendation: a pipeline that can run unattended and cannot
#: terminate is the failure nobody is present to interrupt.
MAX_LOOP_ITERATIONS = 50


@dataclass(frozen=True)
class LoopPolicy:
    """Why a stage repeats, and what stops it."""

    max_iterations: int
    until: str = "contract_satisfied"
    #: Consecutive empty rounds before ``no_new_findings`` is believed. One is
    #: usually too few: a round can come up empty and the next one not.
    dry_rounds: int = 2


@dataclass(frozen=True)
class PipelineStage:
    """One stage: what must be true when it is done, and what it needs first."""

    id: str
    goal: str
    #: A goal contract in the shape the executor already normalises, including
    #: its ``validity`` block. This is the whole specification of the stage.
    contract: Dict[str, Any] = field(default_factory=dict)
    depends_on: Tuple[str, ...] = ()
    #: Finding types this stage expects to already exist, produced upstream.
    #: The typed connector between stages: a stage that reads a counter trace
    #: should say so, and be refused if nothing before it takes one.
    assumes: Tuple[str, ...] = ()
    job_type: str = "research"
    #: A deterministic runner, when the stage must not vary at all.
    runner: str = ""
    loop: Optional[LoopPolicy] = None
    #: Require a human decision before anything downstream starts. This is what
    #: makes a pipeline semi-autonomous rather than unattended.
    checkpoint: bool = False

    def required_finding_types(self) -> List[str]:
        """The finding types this stage's contract demands."""
        return _required_types(self.contract)

    def iterations(self) -> int:
        return self.loop.max_iterations if self.loop else 1


@dataclass(frozen=True)
class Pipeline:
    name: str
    stages: Tuple[PipelineStage, ...] = ()

    def by_id(self) -> Dict[str, PipelineStage]:
        return {stage.id: stage for stage in self.stages}


def _required_types(contract: Mapping[str, Any]) -> List[str]:
    """Read required finding types from a contract in any shape it is written.

    Mirrors agent_goal_contract_service, which accepts a list, a mapping of
    counts, or the normalised ``*_counts`` key. Reading only one of those here
    would make a stage look like it demands nothing.
    """
    if not isinstance(contract, Mapping):
        return []
    counts = contract.get("required_finding_type_counts")
    if isinstance(counts, Mapping):
        return [str(k).strip() for k in counts if str(k).strip()]
    names = contract.get("required_finding_types")
    if isinstance(names, Mapping):
        return [str(k).strip() for k in names if str(k).strip()]
    if isinstance(names, list):
        return [str(x).strip() for x in names if str(x).strip()]
    return []


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split(",") if x.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(x).strip() for x in value if str(x).strip())
    return ()


def normalize(spec: Mapping[str, Any]) -> Pipeline:
    """Read a pipeline from plain config, the way job configs are read."""
    raw_stages = spec.get("stages")
    stages: List[PipelineStage] = []
    for index, raw in enumerate(raw_stages if isinstance(raw_stages, list) else []):
        if not isinstance(raw, Mapping):
            continue
        loop_raw = raw.get("loop")
        loop = None
        if isinstance(loop_raw, Mapping):
            loop = LoopPolicy(
                max_iterations=_as_int(loop_raw.get("max_iterations"), 0),
                until=str(loop_raw.get("until") or "contract_satisfied").strip(),
                dry_rounds=_as_int(loop_raw.get("dry_rounds"), 2),
            )
        contract = raw.get("contract")
        stages.append(
            PipelineStage(
                id=str(raw.get("id") or f"stage_{index + 1}").strip(),
                goal=str(raw.get("goal") or "").strip(),
                contract=dict(contract) if isinstance(contract, Mapping) else {},
                depends_on=_as_tuple(raw.get("depends_on")),
                assumes=_as_tuple(raw.get("assumes")),
                job_type=str(raw.get("job_type") or "research").strip(),
                runner=str(raw.get("runner") or "").strip(),
                loop=loop,
                checkpoint=bool(raw.get("checkpoint")),
            )
        )
    return Pipeline(name=str(spec.get("name") or "").strip(), stages=tuple(stages))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate(pipeline: Pipeline) -> List[str]:
    """Everything wrong with this pipeline that is decidable without running it.

    Returns problems in the order a reader can act on them: the shape of the
    graph first, then what each stage asks for. An empty list means the spec is
    coherent -- not that the research is a good idea.
    """
    problems: List[str] = []
    if not pipeline.stages:
        return ["pipeline has no stages"]

    seen: set = set()
    for stage in pipeline.stages:
        if not stage.id:
            problems.append("a stage has no id")
        elif stage.id in seen:
            problems.append(f"duplicate stage id: {stage.id!r}")
        seen.add(stage.id)

    known = pipeline.by_id()
    for stage in pipeline.stages:
        for parent in stage.depends_on:
            if parent not in known:
                problems.append(f"{stage.id}: depends on unknown stage {parent!r}")
            elif parent == stage.id:
                problems.append(f"{stage.id}: depends on itself")

    cycle = _find_cycle(pipeline)
    if cycle:
        problems.append("stages form a cycle: " + " -> ".join(cycle))

    for stage in pipeline.stages:
        problems.extend(_stage_problems(stage, pipeline))
    return problems


def _stage_problems(stage: PipelineStage, pipeline: Pipeline) -> List[str]:
    problems: List[str] = []
    required = stage.required_finding_types()
    if not required and not stage.runner and not stage.checkpoint:
        # A stage with nothing to satisfy always passes, which makes the
        # pipeline longer without making it stronger.
        problems.append(
            f"{stage.id}: contract requires no finding types, so nothing can fail it"
        )

    unobtainable = evidence.unobtainable(required)
    if unobtainable:
        problems.append(
            f"{stage.id}: no tool produces {', '.join(sorted(unobtainable))} -- "
            "the contract cannot be satisfied however well the run behaves"
        )

    upstream = _upstream_finding_types(stage, pipeline)
    for assumed in stage.assumes:
        if assumed not in upstream:
            problems.append(
                f"{stage.id}: assumes {assumed!r}, which no stage before it produces"
            )

    if stage.loop is not None:
        loop = stage.loop
        if loop.until not in LOOP_CONDITIONS:
            problems.append(
                f"{stage.id}: loop stops on {loop.until!r}; "
                f"known conditions are {', '.join(LOOP_CONDITIONS)}"
            )
        if loop.max_iterations < 1:
            problems.append(
                f"{stage.id}: loop has no iteration bound. An unattended "
                "pipeline that cannot terminate is the one failure nobody is "
                "present to interrupt"
            )
        elif loop.max_iterations > MAX_LOOP_ITERATIONS:
            problems.append(
                f"{stage.id}: loop may run {loop.max_iterations} times, over the "
                f"{MAX_LOOP_ITERATIONS} allowed"
            )
        if loop.until == "no_new_findings" and loop.dry_rounds < 1:
            problems.append(f"{stage.id}: no_new_findings needs at least one dry round")
    return problems


def _upstream_finding_types(stage: PipelineStage, pipeline: Pipeline) -> set:
    """Every finding type produced by anything this stage transitively follows."""
    known = pipeline.by_id()
    produced: set = set()
    seen: set = set()
    frontier = list(stage.depends_on)
    while frontier:
        current = frontier.pop()
        if current in seen or current not in known:
            continue
        seen.add(current)
        parent = known[current]
        produced.update(parent.required_finding_types())
        frontier.extend(parent.depends_on)
    return produced


def _find_cycle(pipeline: Pipeline) -> List[str]:
    """One cycle, named, or an empty list.

    Named rather than merely detected: "these stages form a cycle" is
    actionable and "this graph is not a DAG" is not.
    """
    known = pipeline.by_id()
    state: Dict[str, int] = {}
    path: List[str] = []

    def walk(node: str) -> List[str]:
        state[node] = 1
        path.append(node)
        for parent in known[node].depends_on if node in known else ():
            if parent not in known:
                continue
            if state.get(parent) == 1:
                return path[path.index(parent) :] + [parent]
            if state.get(parent, 0) == 0:
                found = walk(parent)
                if found:
                    return found
        state[node] = 2
        path.pop()
        return []

    for stage in pipeline.stages:
        if state.get(stage.id, 0) == 0:
            found = walk(stage.id)
            if found:
                return found
    return []


def topological_order(pipeline: Pipeline) -> List[str]:
    """Stage ids in an order where every dependency precedes its dependant.

    Ties are broken by declaration order, so the same spec always compiles to
    the same plan and two runs of it can be compared.
    """
    known = pipeline.by_id()
    remaining = {
        s.id: set(p for p in s.depends_on if p in known) for s in pipeline.stages
    }
    order: List[str] = []
    while remaining:
        ready = [
            stage.id
            for stage in pipeline.stages
            if stage.id in remaining and not remaining[stage.id]
        ]
        if not ready:
            break  # a cycle; validate() reports it
        for node in ready:
            order.append(node)
            del remaining[node]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def _incremental_chain(
    required: Sequence[str], inherited: Iterable[str]
) -> Tuple[str, ...]:
    """The tools this stage still has to run, given what precedes it.

    ``chain_for`` derives a chain from nothing, which is right for a single job
    and wrong for a stage: it re-derives every prerequisite, so a stage that
    reads a trace an earlier stage already sampled is charged for sampling it
    again. Priced that way, cost grows with the depth of the graph and a
    perfectly affordable pipeline gets refused -- and refusing good work is not
    the safe direction, it is the other failure.

    This assumes evidence carries forward, which is what job chaining does: a
    child is handed what its parent produced. A stage that cannot see upstream
    evidence should not declare it in ``assumes``.
    """
    have = {str(x) for x in inherited}
    kept: List[str] = []
    for tool in evidence.chain_for(required):
        entry = evidence.entry_for(tool)
        produces = set(entry.produces) if entry else set()
        # Keep a tool unless everything it makes is already in hand. A tool
        # that produces something new stays, even if it also repeats something.
        #
        # Perishable evidence is never in hand: a test result from before the
        # patch describes a tree that no longer exists, so a verify stage that
        # inherited it would derive no tools and gate on nothing.
        if entry is not None and entry.perishable:
            kept.append(tool)
            continue
        if produces and produces <= have:
            continue
        kept.append(tool)
    return tuple(kept)


def _chain_seconds(tools: Sequence[str]) -> int:
    """Price exactly these tools, on the same terms estimate_chain_seconds uses."""
    total = 0
    for tool in tools:
        entry = evidence.entry_for(tool)
        if entry is None:
            continue
        cost = int(entry.typical_seconds or 0)
        if cost >= evidence.EXPENSIVE_SECONDS:
            cost *= evidence.EXPENSIVE_TOOL_ATTEMPTS
        total += cost
    return total


@dataclass(frozen=True)
class StagePlan:
    """What one stage will do, and what it is expected to cost."""

    stage_id: str
    tools: Tuple[str, ...]
    iterations: int
    seconds: int
    checkpoint: bool
    #: Tools in this stage the evidence map has no recorded cost for. Their
    #: time is counted as zero because nothing knows better, so a stage made
    #: only of these prices at nothing and is not therefore free.
    unpriced: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelinePlan:
    """The compiled pipeline: an order, a derived chain, and a price."""

    order: Tuple[str, ...]
    stages: Tuple[StagePlan, ...]
    total_seconds: int
    critical_path_seconds: int
    checkpoints: Tuple[str, ...]
    unpriced: Tuple[str, ...] = ()

    def describe(self) -> List[str]:
        lines = [
            f"{len(self.stages)} stages, "
            f"~{self.total_seconds // 60} min of work, "
            f"~{self.critical_path_seconds // 60} min on the longest path"
        ]
        if self.unpriced:
            lines.append(
                f"  estimate is a floor: no recorded cost for "
                f"{', '.join(self.unpriced)}"
            )
        for plan in self.stages:
            repeat = f" x{plan.iterations}" if plan.iterations > 1 else ""
            tools = ", ".join(plan.tools) if plan.tools else "no derived tools"
            gate = " [waits for a human]" if plan.checkpoint else ""
            cost = (
                "cost unknown"
                if plan.unpriced and not plan.seconds
                else f"~{plan.seconds // 60} min"
            )
            lines.append(f"  {plan.stage_id}{repeat}: {tools} ({cost}){gate}")
        return lines


def plan(pipeline: Pipeline) -> PipelinePlan:
    """Compile the pipeline: derive each stage's chain and price the whole graph.

    Two figures, because they answer different questions. ``total_seconds`` is
    what the pipeline spends and is what a budget has to cover.
    ``critical_path_seconds`` is the longest chain of dependent stages, which
    is the wall-clock floor however much runs in parallel -- a pipeline can be
    affordable and still take a week.
    """
    order = topological_order(pipeline)
    known = pipeline.by_id()
    plans: List[StagePlan] = []
    cost: Dict[str, int] = {}
    finish: Dict[str, int] = {}

    for stage_id in order:
        stage = known[stage_id]
        required = stage.required_finding_types()
        iterations = stage.iterations()
        inherited = _upstream_finding_types(stage, pipeline)
        tools = _incremental_chain(required, inherited)
        seconds = _chain_seconds(tools) * iterations
        cost[stage_id] = seconds
        upstream_finish = max(
            (finish.get(p, 0) for p in stage.depends_on if p in known), default=0
        )
        finish[stage_id] = upstream_finish + seconds
        unpriced = tuple(
            tool
            for tool in tools
            if not (
                evidence.entry_for(tool) and evidence.entry_for(tool).typical_seconds
            )
        )
        plans.append(
            StagePlan(
                stage_id=stage_id,
                tools=tools,
                iterations=iterations,
                seconds=seconds,
                checkpoint=stage.checkpoint,
                unpriced=unpriced,
            )
        )

    return PipelinePlan(
        order=tuple(order),
        stages=tuple(plans),
        total_seconds=sum(cost.values()),
        critical_path_seconds=max(finish.values(), default=0),
        checkpoints=tuple(p.stage_id for p in plans if p.checkpoint),
        unpriced=tuple(sorted({t for p in plans for t in p.unpriced})),
    )


def check_budget(pipeline: Pipeline, budget_seconds: int) -> Dict[str, Any]:
    """Whether this pipeline can afford itself, decided before it starts.

    The same question ``check_runtime_budget`` asks of one job, asked of the
    graph. A pipeline refused here has cost nothing; the alternative is finding
    out at the point where most of the budget is already gone and the evidence
    is half-collected, which is the worst place to learn it.
    """
    compiled = plan(pipeline)
    budget = max(0, int(budget_seconds or 0))
    affordable = compiled.total_seconds <= budget
    result: Dict[str, Any] = {
        "affordable": affordable,
        "budget_seconds": budget,
        "estimated_seconds": compiled.total_seconds,
        "critical_path_seconds": compiled.critical_path_seconds,
        "unpriced_tools": list(compiled.unpriced),
    }
    if compiled.unpriced and affordable:
        # Saying yes on an estimate that skipped tools is how a budget comes to
        # sound generous and be short. The verdict stands; the caveat travels
        # with it.
        result["caveat"] = (
            "Fits the budget, but the evidence map has no recorded cost for "
            + ", ".join(compiled.unpriced)
            + ", so this estimate is a floor rather than a prediction."
        )
    if not affordable:
        short_by = compiled.total_seconds - budget
        result["refusal"] = (
            f"This pipeline needs about {compiled.total_seconds // 60} minutes and "
            f"has {budget // 60}. It is short by roughly {short_by // 60} minutes. "
            "The estimate is order-of-magnitude and the workload sets the real "
            "cost, so treat this as the wrong range rather than a precise gap: "
            "cut a stage, lower a loop bound, or raise the budget."
        )
    return result


def describe(pipeline: Pipeline) -> List[str]:
    """The pipeline as a reader needs it: problems first, then the plan."""
    problems = validate(pipeline)
    if problems:
        return [f"{pipeline.name or 'pipeline'} cannot run:"] + [
            f"  - {p}" for p in problems
        ]
    return [f"{pipeline.name or 'pipeline'}:"] + plan(pipeline).describe()
