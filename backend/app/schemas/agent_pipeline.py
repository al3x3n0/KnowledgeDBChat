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


class PipelineEvidenceType(BaseModel):
    """One finding type an author may require."""

    name: str
    producers: List[str] = Field(default_factory=list)
    typical_seconds: int = 0
    #: Job types every producer permits. Empty means unrestricted. A stage
    #: requiring this evidence under any other job type plans tools it cannot
    #: then call.
    job_types: List[str] = Field(default_factory=list)
    perishable: bool = False
    consumes: str = ""


class PipelineVocabularyResponse(BaseModel):
    """What a stage editor may offer, and a drafter may use.

    Served rather than hardcoded in the frontend for the same reason the
    backend derives it from the tool specs: a second list of finding types
    drifts from the tools the first time one is added.
    """

    evidence_types: List[PipelineEvidenceType] = Field(default_factory=list)
    job_types: List[str] = Field(default_factory=list)


class PipelineDraftRequest(BaseModel):
    """Draft a pipeline from a description of what someone wants done."""

    description: str = Field(..., min_length=1, max_length=4000)
    budget_seconds: Optional[int] = Field(
        None, gt=0, description="Shape the draft to roughly fit this budget"
    )


class PipelineDraftResponse(BaseModel):
    spec: Dict[str, Any]
    #: What the checker still says about it. A draft is a starting point, not
    #: a finished pipeline, and one that does not check is still worth having
    #: in the editor -- so this is reported rather than raised.
    problems: List[str] = Field(default_factory=list)
    #: Whether the first attempt had to be repaired against the checker.
    repaired: bool = False


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


class PipelineRunStage(BaseModel):
    """One stage of one run, as a caller choosing a restart point needs it."""

    stage: str
    #: Empty for a stage the run has not reached. A planned stage is still part
    #: of the run and has to be shown as one -- a progress view built only from
    #: the stages that have jobs reports "2 stages" at stage two of six and
    #: looks finished.
    job_id: str = ""
    status: str
    iteration: int
    #: Not the same question as `status`. A stage can complete without meeting
    #: its contract -- out of iterations, or it gave up -- and that is exactly
    #: the stage whose output nothing downstream should be built on.
    contract_satisfied: bool
    #: Whether this stage can be run again. A stage with a predecessor re-fires
    #: the chain from it; the head is re-run from its own definition, since the
    #: stage a run most often needs redone is the one that produced what
    #: everything else derives from. False for a stage that has not started --
    #: there is nothing yet to run again -- and for one that is running, which
    #: is the collision the execution lease exists to prevent.
    restartable: bool

    #: What the stage was asked to establish. Carried so a run can be read
    #: without the spec that produced it beside you; a stage id alone is a
    #: label, not a description of the work.
    goal: str = ""
    #: A stage that holds the run until a person approves it. The next stage
    #: does not start on completion, so "completed and nothing happening" is
    #: the expected state rather than a stall.
    checkpoint: bool = False
    #: `current_phase` says the job is parked on a person rather than working.
    #: Distinguished from a plain status because a waiting run and a dead run
    #: look identical in `status` alone.
    waiting_on_person: bool = False
    #: How many jobs this stage has had. More than one means it was restarted;
    #: the fields above describe the most recent attempt.
    attempts: int = 1
    #: A person rejected some of this stage's evidence. Advisory -- no verdict
    #: changed -- but a stage resting on rejected evidence must not read as
    #: clean.
    disputed: bool = False
    progress: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PipelineRunStagesResponse(BaseModel):
    root_job_id: str
    stages: List[PipelineRunStage]

    #: The run as a whole, derived from its stages rather than stored: there is
    #: no row for a run, only a chain of jobs, so the chain is the only thing
    #: that knows.
    pipeline: str = ""
    saved_pipeline_id: Optional[str] = None
    status: str = "pending"
    total_stages: int = 0
    completed_stages: int = 0
    #: The stage the run is on, or the one it stopped at. None when every stage
    #: is done.
    current_stage: Optional[str] = None


class PipelineRestartRequest(BaseModel):
    """Restart one stage of a run."""

    stage: str = Field(..., min_length=1, description="The stage id to run again")
    #: Attached as an operator correction the restarted stage actually reads,
    #: through the same channel a resumed job uses. A restart with no note is a
    #: stage repeating what it already did, and the reason it stopped is
    #: usually something only a person knows.
    note: Optional[str] = Field(
        None, max_length=2000, description="A correction for the restarted stage"
    )


class PipelineRestartResponse(BaseModel):
    root_job_id: str
    stage: str
    job_id: str
    note_attached: bool


class PipelineInsertStageRequest(BaseModel):
    """Insert a stage between one that succeeded and whatever followed it.

    The move a stuck run usually needs: the gap is between two stages, and
    both restarting and relaunching redo work that was fine.
    """

    after: str = Field(..., min_length=1, description="Completed stage to insert after")
    #: The stage itself, in the same shape a pipeline spec uses. It needs a
    #: contract: a stage with nothing to satisfy cannot fail, so it cannot be
    #: the fix for a stage that did.
    stage: Dict[str, Any] = Field(..., description="id, goal, contract, job_type")
    note: Optional[str] = Field(
        None, max_length=2000, description="A correction the new stage reads"
    )


class PipelineInsertStageResponse(BaseModel):
    root_job_id: str
    stage: str
    after: str
    job_id: str
    #: Stages that were going to run next and will now run beneath the new one
    #: instead. Named because a caller should see what its insertion moved.
    displaced: List[str]
