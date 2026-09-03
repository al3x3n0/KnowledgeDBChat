"""Pipeline specs: check one before it runs, and compile it to a chain.

`agent_pipeline_spec` and `agent_pipeline_binding` have existed and been tested
for a while with no way to reach them: no endpoint, no client method, nothing
in the UI. The capability they carry is the one that saves the most — every
check here is decidable from the spec alone, before anything expensive starts:

  - a contract asking for evidence no tool produces
  - a budget that sounded generous and is two orders of magnitude short
  - a stage built on a measurement the stage before it never takes
  - a loop with no bound

Those are the failures that otherwise cost a full run to discover.

Nothing here launches anything. `/check` is read-only, and `/bind` returns the
chain a pipeline compiles to so it can be inspected before it is used —
launching stays with the existing chain endpoints, which already carry the
authorisation and budget checks a launch needs.
"""

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.agent_job import AgentJobCreate
from app.schemas.agent_pipeline import (
    PipelineBindResponse,
    PipelineCheckResponse,
    PipelineLaunchRequest,
    PipelineLaunchResponse,
    PipelinePlanResponse,
    PipelineSpecRequest,
    StagePlanResponse,
)
from app.services import agent_pipeline_binding, agent_pipeline_spec
from app.services.agent_job_creation_service import agent_job_creation_service
from app.services.auth_service import get_current_user

router = APIRouter()


def _normalized(spec):
    """Parse a spec, refusing the shapes `normalize` would quietly swallow.

    `normalize` is deliberately lenient: `stages: "oops"` becomes no stages at
    all, and `validate` then reports "pipeline has no stages". That is true and
    useless — the author did write stages, they wrote them wrongly, and being
    told the opposite sends them looking in the wrong place. The one structural
    check that leniency hides belongs here rather than as a second declaration
    of the whole spec shape.
    """
    if not isinstance(spec, dict):
        raise HTTPException(
            status_code=400, detail="Not a pipeline spec: expected an object"
        )
    stages = spec.get("stages")
    if stages is not None and not isinstance(stages, (list, tuple)):
        raise HTTPException(
            status_code=400,
            detail=(
                "Not a pipeline spec: 'stages' must be a list, got "
                f"{type(stages).__name__}"
            ),
        )
    try:
        return agent_pipeline_spec.normalize(spec)
    except (ValueError, TypeError, AttributeError) as error:
        # A spec too malformed to parse is still an answer about the spec, not
        # a server fault.
        raise HTTPException(status_code=400, detail=f"Not a pipeline spec: {error}")


@router.post("/check", response_model=PipelineCheckResponse)
async def check_pipeline(
    payload: PipelineSpecRequest,
    current_user: User = Depends(get_current_user),
):
    """Say everything that is wrong with a pipeline, without running it.

    Deliberately not fail-fast. An author wants the whole list — fixing one
    problem only to be told about the next one is the slow way to find out a
    spec is unusable.
    """
    pipeline = _normalized(payload.spec)
    problems = agent_pipeline_spec.validate(pipeline)
    valid = not problems

    # Only ask the later questions once the earlier ones are settled: binding
    # and planning a spec that is already invalid produces noise, not answers.
    binding_problems: list[str] = []
    plan_response = None
    budget = None
    description: list[str] = []

    if valid:
        binding_problems = agent_pipeline_binding.expressible(pipeline)
        description = agent_pipeline_spec.describe(pipeline)
        compiled = agent_pipeline_spec.plan(pipeline)
        plan_response = PipelinePlanResponse(
            order=list(compiled.order),
            stages=[
                StagePlanResponse(
                    stage_id=s.stage_id,
                    tools=list(s.tools),
                    iterations=s.iterations,
                    seconds=s.seconds,
                    checkpoint=s.checkpoint,
                    unpriced=list(s.unpriced),
                )
                for s in compiled.stages
            ],
            total_seconds=compiled.total_seconds,
            critical_path_seconds=compiled.critical_path_seconds,
            checkpoints=list(compiled.checkpoints),
        )
        if payload.budget_seconds:
            budget = agent_pipeline_spec.check_budget(pipeline, payload.budget_seconds)

    return PipelineCheckResponse(
        valid=valid,
        problems=problems,
        expressible=valid and not binding_problems,
        binding_problems=binding_problems,
        description=description,
        plan=plan_response,
        budget=budget,
    )


@router.post("/bind", response_model=PipelineBindResponse)
async def bind_pipeline(
    payload: PipelineSpecRequest,
    current_user: User = Depends(get_current_user),
):
    """Compile a pipeline to the job chain it would run as.

    Returns the chain rather than launching it, so the shape can be read before
    anything is committed to it.
    """
    pipeline = _normalized(payload.spec)
    problems = agent_pipeline_spec.validate(pipeline)
    if problems:
        # 422 rather than 400: the request was well-formed and the pipeline is
        # not, which is a different thing for a caller to handle.
        raise HTTPException(status_code=422, detail="; ".join(problems))

    try:
        bound = agent_pipeline_binding.bind(pipeline)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    compiled = agent_pipeline_spec.plan(pipeline)
    return PipelineBindResponse(
        name=pipeline.name,
        chain_config={"roots": bound.roots},
        deferred_edges=[
            {"after": edge.after, "launch": edge.launch, "reason": edge.reason}
            for edge in bound.deferred
        ],
        checkpoints=list(compiled.checkpoints),
        description=agent_pipeline_binding.describe(pipeline),
    )


@router.post("/launch", response_model=PipelineLaunchResponse, status_code=201)
async def launch_pipeline(
    payload: PipelineLaunchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a pipeline. The first thing here that actually spends anything.

    Every check `/check` performs runs again, and none of them are advisory at
    this point: a pipeline that cannot be satisfied, cannot be expressed as a
    chain, or cannot afford itself is refused rather than started. Re-running
    them is deliberate — the spec that arrives here is not necessarily the one
    the caller last checked.
    """
    pipeline = _normalized(payload.spec)

    problems = agent_pipeline_spec.validate(pipeline)
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))

    binding_problems = agent_pipeline_binding.expressible(pipeline)
    if binding_problems:
        raise HTTPException(status_code=422, detail="; ".join(binding_problems))

    compiled = agent_pipeline_spec.plan(pipeline)

    if payload.budget_seconds:
        budget = agent_pipeline_spec.check_budget(pipeline, payload.budget_seconds)
        if not budget.get("affordable"):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Needs {compiled.total_seconds}s and the budget is "
                    f"{payload.budget_seconds}s"
                ),
            )

    # The estimate the caller agreed to must be the estimate that is about to
    # be spent. A spec edited between checking and launching is the ordinary
    # way someone starts a run they have not actually seen priced.
    if (
        payload.acknowledged_seconds is not None
        and payload.acknowledged_seconds != compiled.total_seconds
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"This pipeline now costs {compiled.total_seconds}s, not "
                f"{payload.acknowledged_seconds}s. Check it again before launching."
            ),
        )

    bound = agent_pipeline_binding.bind(pipeline)
    if not bound.roots:
        raise HTTPException(status_code=422, detail="Pipeline compiled to no jobs")
    if len(bound.roots) > 1:
        # A chain has one head. A pipeline with several independent roots is a
        # real shape, and launching only the first of them silently would run
        # part of what was asked for.
        raise HTTPException(
            status_code=422,
            detail=(
                f"This pipeline has {len(bound.roots)} independent starting stages; "
                "a chain runs from one. Give them a common first stage."
            ),
        )

    root = bound.roots[0]
    chain_config = root.get("chain_config") or {}
    job = await agent_job_creation_service.create_from_request(
        request=AgentJobCreate(
            name=root.get("name") or pipeline.name,
            description=root.get("description"),
            goal=root.get("goal") or "",
            job_type=root.get("job_type") or "research",
            config=root.get("config") or {},
            chain_config=chain_config,
            max_iterations=root.get("max_iterations") or 100,
        ),
        user_id=current_user.id,
        db=db,
    )

    logger.info(
        f"Launched pipeline '{pipeline.name}' as job {job.id} "
        f"({len(compiled.order)} stages, ~{compiled.total_seconds}s)"
    )
    return PipelineLaunchResponse(
        job_id=str(job.id),
        name=pipeline.name,
        stages=list(compiled.order),
        estimated_seconds=compiled.total_seconds,
        checkpoints=list(compiled.checkpoints),
    )
