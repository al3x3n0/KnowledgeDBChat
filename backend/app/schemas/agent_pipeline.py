"""Request and response shapes for pipeline specs.

The spec itself stays an untyped mapping on the way in. `agent_pipeline_spec.
normalize` is the authority on its shape and reports what is wrong in terms the
author can act on; re-declaring the same structure in Pydantic would give two
definitions of a pipeline that have to agree, and the error messages from the
weaker one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

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
    #: The saved pipeline this run came from, if any. Recorded on both sides so
    #: a run can say which pipeline produced it and a pipeline can say what it
    #: has produced — otherwise a launched spec is anonymous the moment the
    #: editor is closed.
    pipeline_id: Optional[UUID] = None


class PipelineLaunchResponse(BaseModel):
    """What was started."""

    job_id: str
    pipeline_id: Optional[str] = None
    name: str
    stages: List[str] = Field(default_factory=list)
    estimated_seconds: int = 0
    checkpoints: List[str] = Field(default_factory=list)


class SavedPipelineCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    spec: Dict[str, Any]
    description: Optional[str] = Field(None, max_length=2000)


class SavedPipelineUpdate(BaseModel):
    """Every field optional, so renaming does not require restating the spec."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    spec: Optional[Dict[str, Any]] = None
    description: Optional[str] = Field(None, max_length=2000)


class SavedPipelineResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    spec: Dict[str, Any]
    #: What the checker said when it was last saved. A cache for the list, not
    #: an answer: tools and their costs change underneath a saved spec, so the
    #: studio re-checks whatever it opens.
    last_check_valid: Optional[str] = None
    last_estimated_seconds: Optional[int] = None
    launch_count: int = 0
    last_launched_at: Optional[str] = None
    last_job_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def of(cls, row: Any) -> "SavedPipelineResponse":
        return cls(
            id=str(row.id),
            name=row.name,
            description=row.description,
            spec=row.spec or {},
            last_check_valid=row.last_check_valid,
            last_estimated_seconds=row.last_estimated_seconds,
            launch_count=row.launch_count or 0,
            last_launched_at=row.last_launched_at.isoformat()
            if row.last_launched_at
            else None,
            last_job_id=str(row.last_job_id) if row.last_job_id else None,
            created_at=row.created_at.isoformat() if row.created_at else None,
            updated_at=row.updated_at.isoformat() if row.updated_at else None,
        )
