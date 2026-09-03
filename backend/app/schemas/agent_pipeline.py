"""Request and response shapes for pipeline specs.

The spec itself stays an untyped mapping on the way in. `agent_pipeline_spec.
normalize` is the authority on its shape and reports what is wrong in terms the
author can act on; re-declaring the same structure in Pydantic would give two
definitions of a pipeline that have to agree, and the error messages from the
weaker one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineSpecRequest(BaseModel):
    """A pipeline to check. `budget_seconds` is optional: a pipeline can be
    valid and still be unaffordable, and those are different answers."""

    spec: Dict[str, Any] = Field(..., description="The pipeline spec: name and stages")
    budget_seconds: Optional[int] = Field(
        None, ge=1, description="If given, also check the plan fits this budget"
    )


class StagePlanResponse(BaseModel):
    stage_id: str
    tools: List[str] = Field(default_factory=list)
    iterations: int = 1
    seconds: int = 0
    checkpoint: bool = False
    #: Tools with no recorded cost. Their time counts as zero because nothing
    #: knows better, which is not the same as being free — a stage made only of
    #: these prices at nothing and should be read with that in mind.
    unpriced: List[str] = Field(default_factory=list)


class PipelinePlanResponse(BaseModel):
    order: List[str] = Field(default_factory=list)
    stages: List[StagePlanResponse] = Field(default_factory=list)
    total_seconds: int = 0
    critical_path_seconds: int = 0
    checkpoints: List[str] = Field(default_factory=list)


class PipelineCheckResponse(BaseModel):
    """What is wrong with a pipeline, before anything expensive runs.

    `valid` covers only the spec. A pipeline can be valid, be expressible as a
    chain, and still not fit its budget — so the three are reported separately
    rather than collapsed into one boolean the caller has to interpret.
    """

    valid: bool
    problems: List[str] = Field(default_factory=list)
    #: Whether the pipeline can be expressed as a job chain at all, and why not.
    expressible: bool = False
    binding_problems: List[str] = Field(default_factory=list)
    #: Human-readable account of the pipeline, stage by stage.
    description: List[str] = Field(default_factory=list)
    plan: Optional[PipelinePlanResponse] = None
    budget: Optional[Dict[str, Any]] = None


class PipelineBindResponse(BaseModel):
    """The chain a pipeline compiles to, without launching anything."""

    name: str
    chain_config: Dict[str, Any] = Field(default_factory=dict)
    #: Edges the chain shape cannot carry directly, which the runtime has to
    #: gate instead. Reported rather than hidden: a pipeline that quietly loses
    #: a dependency produces a run that starts before its input exists.
    deferred_edges: List[Dict[str, Any]] = Field(default_factory=list)
    checkpoints: List[str] = Field(default_factory=list)
    description: List[str] = Field(default_factory=list)


class PipelineLaunchRequest(PipelineSpecRequest):
    """Launch a pipeline. `budget_seconds` is a limit, not a note.

    On `/check` a budget is information. Here it is a refusal: a pipeline that
    does not fit is not started, because the whole point of pricing it before
    it runs is to not run the one that cannot afford itself.
    """

    #: Say the estimate out loud. If it does not match what the server computes
    #: the spec changed between checking and launching, and the caller is about
    #: to spend on something they have not seen.
    acknowledged_seconds: Optional[int] = Field(
        None,
        ge=0,
        description="The total the caller was shown when they checked",
    )


class PipelineLaunchResponse(BaseModel):
    """What was started."""

    job_id: str
    name: str
    stages: List[str] = Field(default_factory=list)
    estimated_seconds: int = 0
    checkpoints: List[str] = Field(default_factory=list)
