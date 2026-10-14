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
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import (
    agent_evidence_map,
    agent_pipeline_binding,
    agent_pipeline_spec,
    agent_pipeline_vocabulary,
    llm_json,
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


def _what_makes(entry: Any, most: int = 140) -> str:
    """The tool that yields this evidence, in that tool's own first sentence.

    The list gave a name, a duration and the job types, and nothing about
    what the evidence is. Asked to implement fastmod in C, check it against
    reference cases and benchmark it, the drafter required
    `code_patch_proposal` and `patch_applied`, which mean changing a
    repository: the stage cloned lemire/fastmod, found the function already
    there, and had nothing to change. `implementation_verified` --
    "Run an implementation against reference cases with known outputs" --
    was on the list as a bare word. The sentence is the tool's own
    description, so it says what the tool says and cannot drift from it.
    """
    from app.agent_core import tool_specs

    producers = getattr(entry, "producers", ()) or ()
    if not producers:
        return ""
    spec = tool_specs.spec_for(producers[0])
    text = " ".join(str(getattr(spec, "description", "") or "").split())
    if not text:
        return ""
    end = re.search(r"[.!?](\s|$)", text)
    sentence = text[: end.start() + 1] if end else text
    if len(sentence) > most:
        sentence = sentence[: most - 3].rstrip() + "..."
    return f"made by {producers[0]}: {sentence}"


def _vocabulary_block(extra_evidence: Sequence[Any] = ()) -> str:
    """The evidence types, as a table a model can only choose from.

    ``extra_evidence`` is what the static vocabulary cannot know: the evidence
    this author's own sandbox skills yield.
    """
    lines: List[str] = []
    for entry in [*agent_pipeline_vocabulary.evidence_types(), *extra_evidence]:
        bits = [f"~{entry.typical_seconds}s"]
        if entry.job_types:
            # The trap this prevents: a stage requiring coding evidence left
            # at the default job type plans tools it cannot then call.
            bits.append("job_type must be one of: " + ", ".join(entry.job_types))
        if entry.perishable:
            bits.append("perishable (not inherited by later stages)")
        if agent_evidence_map.is_skill_evidence(entry.name) and entry.consumes:
            # A skill's name says nothing about what it does; its description
            # is the only way a model can tell which request it answers.
            bits.append(f"from {entry.consumes}")
        else:
            made_by = _what_makes(entry)
            if made_by:
                bits.append(made_by)
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
7. Code that belongs to no repository -- an algorithm written from a paper,
   a spec or a finding -- is written, checked and measured in sandboxes
   that take the source directly: writing it is part of the stage that
   requires `implementation_verified`, not a stage of its own. Require
   `code_patch_proposal`, `patch_applied` or `test_result` only when the
   work changes a repository it names.

Evidence types you may require:
{vocabulary}

Prefer few stages over many. Each stage should be a distinct kind of work with
a distinct kind of evidence, not a subdivision of the same work."""


def _extract(completion: Any) -> Dict[str, Any]:
    """The spec out of a completion, whichever way the provider returned it."""
    spec = llm_json.completion_object(completion)
    if spec:
        return spec
    if not str(getattr(completion, "text", "") or "").strip():
        raise PipelineDraftError("The model returned nothing.")
    raise PipelineDraftError("The model did not return a pipeline.")


#: Fields of a stage that a model writes inside the stage's contract.
_STAGE_FIELDS_FOUND_IN_CONTRACTS = ("job_type", "assumes")


def _without_implied_dependencies(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a dependency that another dependency of the same stage implies.

    Only where the shape is unambiguous: every id a string, every
    `depends_on` a list. Anything else is left for the checker to name. Two
    stages that wait on each other are left as well -- that is a cycle, and
    removing an edge would hide it.
    """
    waits: Dict[str, List[str]] = {}
    for stage in stages:
        stage_id, parents = stage.get("id"), stage.get("depends_on")
        if not isinstance(stage_id, str) or stage_id in waits:
            return stages
        if parents is None:
            parents = []
        if not isinstance(parents, list) or not all(
            isinstance(parent, str) for parent in parents
        ):
            return stages
        waits[stage_id] = list(parents)

    def before(stage_id: str) -> set:
        seen: set = set()
        pending = list(waits.get(stage_id, []))
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(waits.get(current, []))
        return seen

    earlier = {stage_id: before(stage_id) for stage_id in waits}
    out = []
    for stage in stages:
        parents = waits[stage["id"]]
        kept = [
            parent
            for parent in parents
            if not any(
                other != parent
                and parent in earlier.get(other, ())
                and other not in earlier.get(parent, ())
                for other in parents
            )
        ]
        if len(kept) != len(parents):
            stage = {**stage, "depends_on": kept}
        out.append(stage)
    return out


def _with_stage_fields_in_place(stage: Dict[str, Any]) -> Dict[str, Any]:
    """Move what a stage says to where it is read, when it is plainly there.

    Only when the right place is empty: a draft that says the same thing in
    both places is not ambiguous, and one that says different things is left
    for the checker to name.
    """
    name = str(stage.get("name") or "").strip()
    if not str(stage.get("id") or "").strip() and name:
        stage["id"] = name

    required = stage.get("required_finding_types")
    if required is not None:
        contract = stage.get("contract")
        if contract is None:
            stage["contract"] = {"required_finding_types": required}
            stage.pop("required_finding_types")
        elif isinstance(contract, Mapping) and not contract.get(
            "required_finding_types"
        ):
            stage["contract"] = {**contract, "required_finding_types": required}
            stage.pop("required_finding_types")

    loop = stage.get("loop")
    if isinstance(loop, bool):
        bound = stage.get("max_iterations")
        if not loop:
            stage.pop("loop")
        elif isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
            stage["loop"] = {"max_iterations": bound}
            stage.pop("max_iterations")
    return stage


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
    * **`job_type` and `assumes` end up there too.** Nothing reads them in a
      contract, so a coding stage was planned as the default `research` and
      refused for requiring evidence research cannot produce. Seen in a live
      draft, every stage.
    * **A stage is shaped like a stage with no contract.** Seen in live
      drafts of an ordinary repository fix (2026-10-14): the stage named by
      `name` with no `id`, so every `depends_on` named a stage that did not
      exist; `required_finding_types` beside the goal instead of in
      `contract`, so the contract required nothing; and `loop: true` with
      `max_iterations` beside it. The repair round did not fix either of the
      first two in two drafts of three. See `_with_stage_fields_in_place`.
    * **A stage waits for a stage and for that stage's own predecessor.** A
      model writes `depends_on: [extract, implement]` to say "this uses the
      paper too", when `implement` already waits for `extract`. The extra
      edge orders nothing and evidence carries forward along a chain anyway,
      but it reads to the binding as a fan-in the runtime cannot build.
      Measured live: two of four ordinary requests were refused for this
      alone, and the repair round did not remove it.
    """
    out = dict(spec)

    raw_stages = out.get("stages")
    stages = []
    for raw in raw_stages if isinstance(raw_stages, list) else []:
        if not isinstance(raw, Mapping):
            continue
        stage = _with_stage_fields_in_place(dict(raw))
        raw_contract = stage.get("contract")
        if isinstance(raw_contract, Mapping):
            contract = dict(raw_contract)
            misplaced = str(contract.pop("goal", "") or "").strip()
            if not str(stage.get("goal") or "").strip() and misplaced:
                stage["goal"] = misplaced
            for key in _STAGE_FIELDS_FOUND_IN_CONTRACTS:
                if key in contract:
                    value = contract.pop(key)
                    if not stage.get(key) and value:
                        stage[key] = value
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
        out["stages"] = _without_implied_dependencies(stages)

    if not str(out.get("name") or "").strip():
        words = [w for w in str(description or "").lower().split() if w.isalnum()]
        out["name"] = "-".join(words[:4]) or "drafted-pipeline"

    return out


def problems_with(
    spec: Dict[str, Any], *, skill_evidence: Optional[Sequence[str]] = None
) -> List[str]:
    """Everything the studio's own checks would say about this spec.

    The same two calls the check endpoint makes, in the same order, so a draft
    is judged by exactly what will judge it a second later on screen.

    ``skill_evidence`` is the skill evidence this author can actually obtain.
    When given, a stage requiring any other `skill_*` type is a problem: the
    static checker accepts the whole namespace, so without this a model could
    invent a skill and the draft would score clean.
    """
    try:
        pipeline = agent_pipeline_spec.normalize(spec)
    except Exception as error:  # noqa: BLE001 - any shape error is a problem
        return [f"Not a pipeline: {error}"]
    problems = list(agent_pipeline_spec.validate(pipeline))
    if skill_evidence is not None:
        problems.extend(skill_problems(pipeline, skill_evidence))
    if not problems:
        problems.extend(agent_pipeline_binding.expressible(pipeline))
    return problems


def skill_problems(pipeline: Any, available: Sequence[str]) -> List[str]:
    """Stages requiring skill evidence this author has no active skill for."""
    have = {str(name) for name in available}
    problems: List[str] = []
    for stage in pipeline.stages:
        for name in stage.required_finding_types():
            if agent_evidence_map.is_skill_evidence(name) and name not in have:
                skill = name[len(agent_evidence_map.SKILL_EVIDENCE_PREFIX) :]
                problems.append(
                    f"{stage.id}: requires {name}, but there is no active "
                    f"sandbox skill {skill!r}"
                    + (
                        f". Active skill evidence: {', '.join(sorted(have))}"
                        if have
                        else ", and no sandbox skill is active at all"
                    )
                    + ". Create and activate the skill, or require evidence "
                    "that exists."
                )
    return problems


async def draft_pipeline(
    *,
    description: str,
    llm_service: Any,
    user_id: Optional[UUID] = None,
    db: Optional[AsyncSession] = None,
    budget_seconds: Optional[int] = None,
    extra_evidence: Sequence[Any] = (),
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

    system = SYSTEM_PROMPT.format(vocabulary=_vocabulary_block(extra_evidence))
    skill_evidence = [str(entry.name) for entry in extra_evidence]
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

    problems = problems_with(spec, skill_evidence=skill_evidence)
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

    repaired_problems = problems_with(repaired, skill_evidence=skill_evidence)
    # Keep whichever is less broken. A repair that made it worse is a repair
    # that should not be shown -- measured once as a model "fixing" an unknown
    # finding type by inventing two more.
    if len(repaired_problems) <= len(problems):
        return repaired, repaired_problems, True
    return spec, problems, False
