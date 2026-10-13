"""A run plans the work that should follow it, and a person decides.

A run could spawn a child job, and a person could draft a pipeline from a
sentence. Nothing let a run do what a person does at the end of exploratory
work: say "the next thing is these four stages, each done when this evidence
exists", and have that carried out. So the step from one job to a plan was
always a person's.

`propose` is that step, with three things kept out of the model's hands:

* **It is checked, not trusted.** The draft goes through the same validator,
  the same chain-expressibility check and the same pricing a person's
  pipeline does. A plan that cannot be satisfied or cannot be expressed is
  handed back with the reasons, and nothing is attached.
* **It is priced and bounded.** A plan over the stage or time ceiling is
  refused with the figure, since the run can ask for a smaller one.
* **It starts nothing.** The plan is attached to the run as its next step
  *waiting on approval* -- the trigger a pipeline's checkpoint stage uses --
  so when the run completes it is held like any checkpoint, appears in the
  queue and on the Review page with its stages and its estimate, and runs
  only when someone approves. Rejecting it ends there.

Only a run that is nobody's stage may propose one: a stage planning its own
pipeline would be a plan that grows a plan, with an approval at every level
and no bound on the depth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from app.models.agent_job import ChainTriggerCondition
from app.services import (
    agent_pipeline_binding,
    agent_pipeline_draft,
    agent_pipeline_spec,
)

#: Marks a chain_config written here, so a second proposal may replace the
#: first and nothing else's chain is ever overwritten.
SELF_PLANNED = "self_planned_pipeline"
#: A run may ask this many times; a plan refused twice is not improved by a
#: third sentence.
MAX_PROPOSALS_PER_RUN = 3
PROPOSAL_FINDING = "pipeline_proposal"


def enabled() -> bool:
    from app.core.config import settings

    return bool(getattr(settings, "AGENT_SELF_PLANNED_PIPELINES_ENABLED", True))


def _limits() -> Dict[str, int]:
    from app.core.config import settings

    return {
        "stages": int(getattr(settings, "AGENT_SELF_PLANNED_PIPELINE_MAX_STAGES", 6)),
        "seconds": int(
            getattr(settings, "AGENT_SELF_PLANNED_PIPELINE_MAX_SECONDS", 4 * 3600)
        ),
    }


def prompt_note(job: Any) -> str:
    """What a run that may propose a pipeline is told about doing so.

    Two live runs were asked to leave follow-up work for a person to agree to.
    One called create_handoff, which started the work at once; the other wrote
    the plan into a finding and stopped. Both had propose_pipeline in their
    tool list and neither called it: one line among 250 does not say that the
    other ways of "planning" start work unasked or reach nobody.

    It goes in the stable prompt, so it depends only on what cannot change
    during a run: the setting, and whether the run is someone's stage.
    """
    if not enabled():
        return ""
    config = job.config if isinstance(getattr(job, "config", None), dict) else {}
    if getattr(job, "parent_job_id", None) or config.get("pipeline_stage"):
        return ""
    return (
        "FOLLOW-UP WORK: if this run shows what should be done next, or its "
        "goal asks for later work to be planned, call propose_pipeline before "
        "you finish. A plan written into a finding or a report starts nothing "
        "and reaches nobody's queue; delegate_subtask and create_handoff "
        "start a child at once and ask nobody. propose_pipeline is the one "
        "that leaves a checked, priced plan for a person to approve. Not "
        "every run has follow-up work: do not propose a plan for its own "
        "sake."
    )


#: The limits a stage has when a person launches the same pipeline: the launch
#: route's iteration budget and the job model's own defaults for the rest.
_STAGE_LIMITS = ("max_tool_calls", "max_llm_calls", "max_runtime_minutes")
_LAUNCHED_STAGE_ITERATIONS = 100


def _with_a_stage_budget(root: Dict[str, Any]) -> Dict[str, Any]:
    """The plan's first stage, with a launched pipeline's limits written in.

    A chained job takes any limit its config does not name from its parent,
    and a proposed plan's parent is the run that proposed it. The first plan
    approved live came from a scoping run allowed ten iterations, so the
    stage that had to clone a repository, write a file and propose a patch
    was allowed ten as well: it spent them orienting, ended with its
    contract unmet, and the stages behind it never started. A stage's budget
    belongs to the stage. Stated on the first stage only, since the later
    ones inherit from it; a limit the plan names (a stage with a `loop`) is
    kept.
    """
    from app.models.agent_job import AgentJob

    stage = dict(root)
    stage.setdefault("max_iterations", _LAUNCHED_STAGE_ITERATIONS)
    for limit in _STAGE_LIMITS:
        default = getattr(AgentJob.__table__.c, limit).default
        if default is not None:
            stage.setdefault(limit, default.arg)
    return stage


def why_this_run_may_not_plan(job: Any) -> Optional[str]:
    """The reason this run cannot propose a pipeline, or None if it can."""
    if not enabled():
        return (
            "Self-planned pipelines are turned off on this deployment "
            "(AGENT_SELF_PLANNED_PIPELINES_ENABLED)."
        )
    config = job.config if isinstance(getattr(job, "config", None), dict) else {}
    if getattr(job, "parent_job_id", None) or config.get("pipeline_stage"):
        return (
            "This run is a stage of a larger run, and only a run that is "
            "nobody's stage may propose a pipeline. Record what you found; "
            "the stages after this one already exist."
        )
    chain = (
        job.chain_config if isinstance(getattr(job, "chain_config", None), dict) else {}
    )
    if chain.get("child_jobs") and not chain.get(SELF_PLANNED):
        return "This run already has work configured to follow it."
    if getattr(job, "chain_triggered", False):
        return "This run's follow-up has already started."
    return None


async def _room_in_campaign(job: Any, db: Any) -> Optional[int]:
    """Jobs left in the budget of the campaign this run belongs to, if any.

    A campaign's budget is what stops findings begetting work without end,
    and a plan's stages are work the campaign caused. Not knowing is not a
    refusal: the campaign counts the stages when the run ends either way.
    """
    config = job.config if isinstance(getattr(job, "config", None), dict) else {}
    campaign_id = config.get("campaign_id")
    if not campaign_id or db is None:
        return None
    try:
        from app.services import research_campaign_service

        return await research_campaign_service.remaining_jobs(db, campaign_id)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning(f"Self-planned pipeline: campaign budget not read: {exc}")
        return None


def _refused(message: str, **data: Any) -> Dict[str, Any]:
    return {"error": message, **({"data": data} if data else {})}


async def propose(
    job: Any,
    description: str,
    *,
    db: Any,
    llm_service: Any,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Draft, check, price and attach a follow-up pipeline for `job`.

    Returns a tool result. On success the job's `chain_config` holds the
    pipeline behind an approval trigger; on refusal nothing has changed.
    """
    state = state if isinstance(state, dict) else {}
    barrier = why_this_run_may_not_plan(job)
    if barrier:
        return _refused(barrier)

    wanted = str(description or "").strip()
    if len(wanted) < 20:
        return _refused(
            "Describe the follow-up work in a sentence or two: what is to be "
            "found out or built, and what would show it is done."
        )
    attempts = int(state.get("pipeline_proposals_made") or 0)
    if attempts >= MAX_PROPOSALS_PER_RUN:
        return _refused(
            f"This run has already proposed a pipeline {attempts} times. "
            "Finish with what you have; a person can draft one in Pipeline Studio."
        )
    state["pipeline_proposals_made"] = attempts + 1

    limits = _limits()
    user_id = getattr(job, "user_id", None)
    skill_names: List[str] = []
    extra_evidence: List[Any] = []
    try:
        from app.services import sandbox_skill_service

        extra_evidence = list(
            await sandbox_skill_service.evidence_types_for_user(db, user_id)
        )
        skill_names = [str(entry.name) for entry in extra_evidence]
    except Exception as exc:  # noqa: BLE001 - a plan without skills is still a plan
        logger.warning(f"Self-planned pipeline: skills not read: {exc}")

    try:
        spec, problems, _repaired = await agent_pipeline_draft.draft_pipeline(
            description=(
                f"{wanted}\n\nThis follows a run whose goal was: "
                f"{str(getattr(job, 'goal', '') or '').strip()[:600]}"
            ),
            llm_service=llm_service,
            user_id=user_id,
            db=db,
            budget_seconds=limits["seconds"],
            extra_evidence=extra_evidence,
        )
    except agent_pipeline_draft.PipelineDraftError as error:
        return _refused(f"No pipeline could be drafted: {error}")

    # Judged again here, by everything that would judge it in the studio:
    # the validator, this person's skills, and whether it fits a chain.
    problems = list(
        dict.fromkeys(
            agent_pipeline_draft.problems_with(spec, skill_evidence=skill_names)
        )
    )
    if problems:
        return _refused(
            "The drafted pipeline does not pass the checker, so nothing was "
            "proposed. Describe the work differently, taking these into "
            "account: " + "; ".join(problems[:6]),
            problems=problems[:12],
        )

    pipeline = agent_pipeline_spec.normalize(spec)
    compiled = agent_pipeline_spec.plan(pipeline)
    stages = list(compiled.order)
    if len(stages) > limits["stages"]:
        return _refused(
            f"The plan has {len(stages)} stages and a run may propose at most "
            f"{limits['stages']}. Ask for the part that matters most.",
            stages=stages,
        )
    if compiled.total_seconds > limits["seconds"]:
        return _refused(
            f"The plan is estimated at {compiled.total_seconds}s and a run may "
            f"propose at most {limits['seconds']}s. Ask for less: fewer "
            "stages, or fewer loop iterations.",
            estimated_seconds=compiled.total_seconds,
        )

    room = await _room_in_campaign(job, db)
    if room is not None and len(stages) > room:
        return _refused(
            f"This run belongs to a campaign with {room} job(s) left in its "
            f"budget, and the plan has {len(stages)} stages. Ask for the part "
            "that matters most.",
            stages=stages,
            campaign_jobs_left=room,
        )

    bound = agent_pipeline_binding.bind(pipeline)
    if len(bound.roots) != 1:
        return _refused(
            f"The plan has {len(bound.roots)} independent starting stages and "
            "a follow-up starts from one. Give them a common first stage."
        )

    proposal = {
        "name": pipeline.name,
        "description": wanted[:1000],
        "stages": stages,
        "stage_goals": {stage.id: stage.goal for stage in pipeline.stages},
        "estimated_seconds": int(compiled.total_seconds),
        "checkpoints": list(compiled.checkpoints),
        "spec": spec,
        "proposed_at": datetime.utcnow().isoformat(),
        "iteration": int(getattr(job, "iteration", 0) or 0),
    }
    # A new dict, assigned: the column is plain JSON.
    job.chain_config = {
        "trigger_condition": ChainTriggerCondition.ON_APPROVAL.value,
        "inherit_results": True,
        "child_jobs": [_with_a_stage_budget(root) for root in bound.roots],
        SELF_PLANNED: proposal,
    }
    state["pipeline_proposal"] = {k: v for k, v in proposal.items() if k != "spec"}

    summary = {k: v for k, v in proposal.items() if k != "spec"}
    return {
        "success": True,
        "data": {
            **summary,
            "note": (
                "Attached to this run, not started. When this run completes "
                "it will wait for a person to approve the plan. Finish your "
                "own work and say in your conclusion why this follow-up is "
                "worth running."
            ),
        },
        "findings": [
            {
                "type": PROPOSAL_FINDING,
                "title": f"Proposed follow-up: {pipeline.name}",
                "stages": stages,
                "estimated_seconds": int(compiled.total_seconds),
            }
        ],
    }


def proposal_of(job: Any) -> Optional[Dict[str, Any]]:
    """The pipeline this run proposed, if it proposed one."""
    chain = (
        job.chain_config if isinstance(getattr(job, "chain_config", None), dict) else {}
    )
    proposal = chain.get(SELF_PLANNED)
    return proposal if isinstance(proposal, dict) else None


def approval_message(proposal: Dict[str, Any]) -> str:
    """What the person deciding is told, in one sentence."""
    stages = [str(s) for s in proposal.get("stages") or []]
    minutes = max(1, round(int(proposal.get("estimated_seconds") or 0) / 60))
    return (
        f"This run proposes a follow-up pipeline of {len(stages)} "
        f"stage{'s' if len(stages) != 1 else ''} (about {minutes} min): "
        f"{' → '.join(stages)}. Approving starts it; rejecting ends here."
    )


__all__ = [
    "MAX_PROPOSALS_PER_RUN",
    "PROPOSAL_FINDING",
    "SELF_PLANNED",
    "approval_message",
    "enabled",
    "proposal_of",
    "propose",
    "why_this_run_may_not_plan",
]
