"""Turn a sentence about what someone wants into a pipeline they can edit.

The studio's blank state is a JSON document in a format nobody knows, and the
worked examples only help someone whose question resembles one of them. This
is the other way in: say what you want done, get a spec that checks, then edit
it. The draft is a STARTING POINT and is never launched from here -- it lands
in the editor, where the ordinary checks apply and a person decides.

Three things make the difference between this and asking a model for JSON:

* **The vocabulary is closed.** A contract may only require finding types that
  a tool actually produces, and inventing them is the single most common way
  an authored pipeline fails its own check. The list goes in the prompt, and
  anything invented anyway is caught below rather than shipped.
* **The checker is the judge, not the model.** Whatever comes back is put
  through the same `validate`/`expressible` the studio runs, and its
  complaints are fed back for one repair round. A model that is told what is
  wrong with its own draft fixes most of it; a model asked to be careful does
  not.
* **A draft that still does not check is returned anyway**, with the problems
  attached. The same reason the studio saves an invalid spec: work in progress
  is worth keeping, and a refusal here would leave the author with nothing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import (
    agent_pipeline_binding,
    agent_pipeline_spec,
    agent_pipeline_vocabulary,
)


class PipelineDraftError(Exception):
    """The model produced nothing usable at all."""


#: The shape a draft must have. Constrained rather than described in prose
#: because a provider that supports schema-constrained output will not return
#: anything else, and the ones that do not still read it as documentation.
DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "stages"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Short kebab-case name for the pipeline",
        },
        "stages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "goal", "contract"],
                "properties": {
                    "id": {"type": "string"},
                    "goal": {
                        "type": "string",
                        "description": (
                            "What must be TRUE when the stage is done, in one "
                            "sentence. Not a list of steps."
                        ),
                    },
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "assumes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Finding types this stage expects to already "
                            "exist. Each must be required by some stage it "
                            "depends on."
                        ),
                    },
                    "job_type": {"type": "string"},
                    "checkpoint": {"type": "boolean"},
                    "loop": {
                        "type": "object",
                        "properties": {
                            "max_iterations": {"type": "integer"},
                            "until": {"type": "string"},
                        },
                    },
                    "contract": {
                        "type": "object",
                        "properties": {
                            "required_finding_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            },
        },
    },
}


def _vocabulary_block() -> str:
    """The evidence types, as a table a model can only choose from."""
    lines: List[str] = []
    for entry in agent_pipeline_vocabulary.evidence_types():
        bits = [f"~{entry.typical_seconds}s"]
        if entry.job_types:
            # The trap this prevents: a stage requiring coding evidence left
            # at the default job type plans tools it cannot then call.
            bits.append("job_type must be one of: " + ", ".join(entry.job_types))
        if entry.perishable:
            bits.append("perishable (not inherited by later stages)")
        lines.append(f"  {entry.name} -- {'; '.join(bits)}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You draft research pipelines for an autonomous agent system.

A pipeline is a DAG of stages. A stage is NOT a list of steps: it is a GOAL
CONTRACT saying what must be true when the stage is done. The tools that get
there are derived from the contract automatically -- never name tools.

Rules that decide whether a draft is usable at all:

1. `contract.required_finding_types` may ONLY contain names from the evidence
   list below. A name that is not on the list can be produced by nothing, and
   the pipeline is rejected. If the user asks for something the list cannot
   express, get as close as the list allows and leave the rest to the goal
   text.
2. A stage's `assumes` must be required by one of the stages it `depends_on`.
   That is the typed connector between stages.
3. `depends_on` must form a DAG with a single starting stage.
4. Any stage requiring evidence whose entry names permitted job types must set
   `job_type` to one of them. The default is `research`.
5. Use `loop` only where one attempt genuinely will not do -- writing code
   from prose, for instance -- and always with `max_iterations`.
6. Use `checkpoint: true` only where a person must approve before the work
   continues.

Evidence types you may require:
{vocabulary}

Prefer few stages over many. Each stage should be a distinct kind of work with
a distinct kind of evidence, not a subdivision of the same work."""


def _extract(completion: Any) -> Dict[str, Any]:
    """The spec out of a completion, whichever way the provider returned it."""
    structured = getattr(completion, "structured", None)
    if isinstance(structured, dict) and structured:
        return structured
    text = str(getattr(completion, "text", "") or "").strip()
    if not text:
        raise PipelineDraftError("The model returned nothing.")
    # Providers without native schema output fence their JSON.
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PipelineDraftError("The model did not return a pipeline.")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise PipelineDraftError(f"The model returned invalid JSON: {error}")


def tidy(spec: Dict[str, Any], description: str) -> Dict[str, Any]:
    """Fix the two things models reliably get wrong, before spending a repair.

    Both are misplacements rather than mistakes of judgement, so a second LLM
    call to correct them is a waste of the round that should go to the things
    only a model can fix.

    * **The goal ends up inside the contract.** Everything else about a stage
      lives there, so `contract.goal` is the natural place to put it -- and
      nothing reads it there. Measured: a five-stage draft scored zero problems
      with every stage's goal misplaced and therefore empty.
    * **The pipeline has no name.** Harmless to the checker, which is exactly
      why nothing catches it; it then reaches the library as "None".
    """
    out = dict(spec)

    raw_stages = out.get("stages")
    stages = []
    for raw in raw_stages if isinstance(raw_stages, list) else []:
        if not isinstance(raw, Mapping):
            continue
        stage = dict(raw)
        raw_contract = stage.get("contract")
        if isinstance(raw_contract, Mapping):
            contract = dict(raw_contract)
            misplaced = str(contract.pop("goal", "") or "").strip()
            if not str(stage.get("goal") or "").strip() and misplaced:
                stage["goal"] = misplaced
            # Written back without the hoisted key: leaving it would keep a
            # second copy of the goal that nothing reads and the author would
            # have to maintain.
            stage["contract"] = contract
        # A contract that is not an object at all is left exactly as it came.
        # This is untrusted model output, and a draft is only a draft: the
        # checker names the problem and the repair round gets a chance at it.
        # Repairing it here by guessing would hide from the model the very
        # mistake it needs to see.
        #
        # Measured: a live draft returned `contract` as a LIST, and this
        # function -- written against its own well-formed fixtures -- turned a
        # recoverable draft into a 500.
        stages.append(stage)
    if stages:
        out["stages"] = stages

    if not str(out.get("name") or "").strip():
        words = [w for w in str(description or "").lower().split() if w.isalnum()]
        out["name"] = "-".join(words[:4]) or "drafted-pipeline"

    return out


def problems_with(spec: Dict[str, Any]) -> List[str]:
    """Everything the studio's own checks would say about this spec.

    The same two calls the check endpoint makes, in the same order, so a draft
    is judged by exactly what will judge it a second later on screen.
    """
    try:
        pipeline = agent_pipeline_spec.normalize(spec)
    except Exception as error:  # noqa: BLE001 - any shape error is a problem
        return [f"Not a pipeline: {error}"]
    problems = list(agent_pipeline_spec.validate(pipeline))
    if not problems:
        problems.extend(agent_pipeline_binding.expressible(pipeline))
    return problems


async def draft_pipeline(
    *,
    description: str,
    llm_service: Any,
    user_id: Optional[UUID] = None,
    db: Optional[AsyncSession] = None,
    budget_seconds: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[str], bool]:
    """Draft a pipeline for `description`.

    Returns the spec, the problems still outstanding, and whether a repair
    round was needed. A spec with outstanding problems is still returned: the
    editor is where a pipeline gets fixed, and handing back nothing would make
    a near-miss worse than useless.
    """
    wanted = str(description or "").strip()
    if not wanted:
        raise PipelineDraftError("Say what the pipeline should do.")

    system = SYSTEM_PROMPT.format(vocabulary=_vocabulary_block())
    user = f"Draft a pipeline for this request:\n\n{wanted}"
    if budget_seconds:
        # Said as a constraint on shape, because the model cannot price a
        # chain -- the checker does that afterwards.
        user += (
            f"\n\nIt must fit a budget of about {int(budget_seconds)} seconds, "
            "so prefer fewer stages and fewer loop iterations."
        )

    completion = await llm_service.generate_structured(
        system_prompt=system,
        user_message=user,
        response_schema=DRAFT_SCHEMA,
        task_type="reasoning",
        user_id=user_id,
        db=db,
        snapshot_context={"phase": "pipeline_draft"},
    )
    spec = tidy(_extract(completion), wanted)

    problems = problems_with(spec)
    if not problems:
        return spec, [], False

    # One repair round. A model told what is wrong with its own draft fixes
    # most of it; a model told to be careful in advance does not. Only one,
    # because a second rarely converges and the editor is right there.
    logger.info(f"Pipeline draft had {len(problems)} problems; repairing")
    repair = (
        "This draft was rejected by the checker. Fix exactly these problems "
        "and return the whole corrected pipeline:\n\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\n\nThe draft was:\n"
        + json.dumps(spec, indent=2)
    )
    try:
        repaired_completion = await llm_service.generate_structured(
            system_prompt=system,
            user_message=repair,
            response_schema=DRAFT_SCHEMA,
            task_type="reasoning",
            user_id=user_id,
            db=db,
            snapshot_context={"phase": "pipeline_draft_repair"},
        )
        repaired = tidy(_extract(repaired_completion), wanted)
    except (PipelineDraftError, Exception) as error:  # noqa: BLE001
        logger.warning(f"Pipeline draft repair failed: {error}")
        return spec, problems, False

    repaired_problems = problems_with(repaired)
    # Keep whichever is less broken. A repair that made it worse is a repair
    # that should not be shown -- measured once as a model "fixing" an unknown
    # finding type by inventing two more.
    if len(repaired_problems) <= len(problems):
        return repaired, repaired_problems, True
    return spec, problems, False
