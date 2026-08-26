"""Turn a checked pipeline into the job chain that runs it.

``agent_pipeline_spec`` decides whether a pipeline is coherent. This decides
whether the machinery can actually express it, which is a different question
and has a narrower answer.

Chaining is a tree with one special case. A job's ``chain_config`` names its
children, each child's config names its own, and a job that completes triggers
them. Fan-in exists but only in one shape: the swarm gate defers a child until
its parent's *siblings* are terminal, so several stages can converge only when
they are all children of one earlier stage. That makes the expressible set the
series-parallel graphs, not every DAG.

So this refuses what it cannot build rather than quietly approximating it. The
tempting approximations are both worse than a refusal: linearising a diamond
throws away the parallelism the author asked for, and dropping a cross edge
runs a stage before the evidence it depends on. A pipeline that will not fit
should be reshaped by the person who wrote it.

Nothing here touches the database. It emits the config a launcher would create
jobs from, so a binding can be read, diffed and tested without running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.services import agent_pipeline_spec as spec_module
from app.services.agent_pipeline_spec import Pipeline, PipelineStage

#: Marks the aggregator child the swarm fan-in gate dedupes on. The executor
#: reads this exact string; it is not a label of ours.
FAN_IN_ORIGIN = "swarm_fan_in_aggregator"


@dataclass
class DeferredEdge:
    """An edge the chain cannot start on its own, and why.

    Checkpoints used to land here: the chain had no approval-gated trigger, so
    a gated stage could not be an edge at all and was reported instead of
    approximated. ``on_approval`` exists now and carries them, so this is empty
    in ordinary use -- kept because the next shape the runner cannot express
    should be reported the same way rather than faked.
    """

    after: str
    launch: str
    reason: str


@dataclass
class BoundPipeline:
    roots: List[Dict[str, Any]] = field(default_factory=list)
    deferred: List[DeferredEdge] = field(default_factory=list)

    def stage_ids(self) -> List[str]:
        """Every stage that appears somewhere in the emitted configs."""
        found: List[str] = []

        def walk(node: Dict[str, Any]) -> None:
            stage_id = str((node.get("config") or {}).get("pipeline_stage") or "")
            if stage_id and stage_id not in found:
                found.append(stage_id)
            chain = node.get("chain_config") or {}
            for child in chain.get("child_jobs") or []:
                walk(child)

        for root in self.roots:
            walk(root)
        return found


def _children_of(pipeline: Pipeline) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {stage.id: [] for stage in pipeline.stages}
    for stage in pipeline.stages:
        for parent in stage.depends_on:
            if parent in children:
                children[parent].append(stage.id)
    return children


def _roots(pipeline: Pipeline) -> List[str]:
    known = pipeline.by_id()
    return [
        s.id for s in pipeline.stages if not [p for p in s.depends_on if p in known]
    ]


def expressible(pipeline: Pipeline) -> List[str]:
    """What the chain machinery cannot build from this pipeline.

    Empty means it fits. This is deliberately separate from ``validate``: a
    pipeline can be perfectly coherent research and still be a shape the runner
    cannot execute, and the two failures want different fixes.
    """
    problems = spec_module.validate(pipeline)
    if problems:
        return [f"pipeline is not valid yet: {problems[0]}"]

    known = pipeline.by_id()
    children = _children_of(pipeline)
    for stage in pipeline.stages:
        parents = [p for p in stage.depends_on if p in known]
        if len(parents) <= 1:
            continue
        grandparents = {
            tuple(sorted(p for p in known[parent].depends_on if p in known))
            for parent in parents
        }
        if len(grandparents) != 1 or len(next(iter(grandparents))) != 1:
            problems.append(
                f"{stage.id}: waits for {', '.join(sorted(parents))}, which do not "
                "all branch from one earlier stage. Fan-in is only expressible "
                "where the converging stages are siblings, so give them a common "
                "parent or merge them"
            )
            continue
        common = next(iter(grandparents))[0]
        siblings = sorted(children.get(common, []))
        if siblings != sorted(parents):
            missing = sorted(set(siblings) - set(parents))
            problems.append(
                f"{stage.id}: waits for {', '.join(sorted(parents))} but "
                f"{common} also branches to {', '.join(missing)}. The gate waits "
                "for every sibling, so it would wait for those too -- depend on "
                "them as well, or move them"
            )
    return problems


def _job_spec(stage: PipelineStage, pipeline: Pipeline) -> Dict[str, Any]:
    """One stage as the config a chained job is created from.

    The keys are the ones ``create_chained_job`` reads. ``pipeline_stage`` is
    ours, carried in the job config so a running job can say which stage of
    which pipeline it is.
    """
    config: Dict[str, Any] = {
        "pipeline": pipeline.name,
        "pipeline_stage": stage.id,
    }
    if stage.contract:
        config["goal_contract"] = dict(stage.contract)
    if stage.assumes:
        config["pipeline_assumes"] = list(stage.assumes)
    if stage.runner:
        config["deterministic_runner"] = stage.runner

    job: Dict[str, Any] = {
        "name": f"{pipeline.name}: {stage.id}" if pipeline.name else stage.id,
        "description": stage.goal,
        "goal": stage.goal,
        "job_type": stage.job_type,
        "config": config,
    }
    if stage.loop is not None:
        # A stage that repeats is a job allowed more iterations, which is the
        # loop the executor already runs. The bound is the author's, not a
        # default: an unbounded one is refused before reaching here.
        job["max_iterations"] = stage.loop.max_iterations
        config["loop_until"] = stage.loop.until
        if stage.loop.until == "no_new_findings":
            config["loop_dry_rounds"] = stage.loop.dry_rounds
    return job


def bind(pipeline: Pipeline) -> BoundPipeline:
    """Build the chain configs for a pipeline that fits.

    Raises ValueError if it does not; ``expressible`` says why in terms the
    author can act on.
    """
    problems = expressible(pipeline)
    if problems:
        raise ValueError("; ".join(problems))

    known = pipeline.by_id()
    children = _children_of(pipeline)
    deferred: List[DeferredEdge] = []

    def build(stage_id: str, seen: Tuple[str, ...] = ()) -> Dict[str, Any]:
        stage = known[stage_id]
        job = _job_spec(stage, pipeline)

        direct = [c for c in children[stage_id] if len(known[c].depends_on) == 1]
        converging = [c for c in children[stage_id] if len(known[c].depends_on) > 1]

        child_jobs: List[Dict[str, Any]] = []
        chain_data: Dict[str, Any] = {}

        for child in direct:
            if child in seen:
                continue
            child_jobs.append(build(child, seen + (stage_id,)))

        for child in converging:
            parents = sorted(p for p in known[child].depends_on if p in known)
            spec = build(child, seen + (stage_id,))
            spec.setdefault("config", {})["origin"] = FAN_IN_ORIGIN
            spec["config"]["swarm_fan_in_group_id"] = f"{pipeline.name}:{child}"
            child_jobs.append(spec)
            chain_data.update(
                {
                    "swarm_fan_in_wait_for_all_siblings": True,
                    "swarm_fan_in_group_id": f"{pipeline.name}:{child}",
                    "swarm_fan_in_expected_siblings": len(parents),
                }
            )

        if child_jobs:
            # A checkpoint stage's chain waits for a person. Completing is what
            # makes it ready to be approved; approving is what starts the next
            # stage. Every other condition fires on its own.
            chain: Dict[str, Any] = {
                "trigger_condition": "on_approval"
                if stage.checkpoint
                else "on_complete",
                "inherit_results": True,
                "child_jobs": child_jobs,
            }
            if chain_data:
                chain["chain_data"] = chain_data
            job["chain_config"] = chain
        return job

    roots = [build(root_id) for root_id in _roots(pipeline)]
    return BoundPipeline(roots=roots, deferred=deferred)


def describe(pipeline: Pipeline) -> List[str]:
    """The binding as a reader needs it, or why there isn't one."""
    problems = expressible(pipeline)
    if problems:
        return [f"{pipeline.name or 'pipeline'} cannot be chained:"] + [
            f"  - {p}" for p in problems
        ]
    bound = bind(pipeline)
    lines = [
        f"{pipeline.name or 'pipeline'}: "
        f"{len(bound.roots)} root job(s), {len(bound.stage_ids())} stages chained"
    ]
    for edge in bound.deferred:
        lines.append(f"  held: {edge.launch} waits on {edge.after} -- {edge.reason}")
    return lines
