"""Convert a saved job chain into a pipeline spec.

A chain and a pipeline produce the same thing: chained jobs, linked by
``chain_config``, created by ``create_chained_job``. They differ only in what
the author wrote down. A chain says *when* the next step fires; a pipeline says
*what must be true* when a stage is done and derives the rest. The pipeline is
the better of the two ways to say it, so chains are being retired as an
authoring concept -- the runtime they share is not going anywhere.

What does not convert
---------------------
A chain step's ``trigger_condition`` has six values. Three have a pipeline
equivalent:

    on_complete   -> depends_on the step before it
    on_approval   -> the same, plus checkpoint=True
    on_findings   -> the same, plus spawn_on={"findings": N} on the parent

``on_findings`` was the interesting one. It fires the next step *while the
parent is still running*, which is what makes a continuous monitor able to
raise alerts without ever finishing, and no ordinary stage dependency can say
that. Pipelines gained ``spawn_on`` for it -- the one place an edge does not
mean "after".

The remaining three still do not convert. ``on_fail`` and ``on_any_end`` branch
on an outcome a DAG edge cannot express, and ``on_progress`` fires partway
through work rather than on evidence.

So this refuses them rather than approximating them. A conversion that looked
successful and quietly turned a live monitor into a stage waiting for an end
that never comes is worse than no conversion at all.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Trigger conditions a pipeline stage can express, and how.
CONVERTIBLE = {
    "on_complete": "depends_on the preceding stage",
    "on_approval": "depends_on the preceding stage, held for review",
    "on_findings": "spawn_on the preceding stage's findings threshold",
}

#: Why each of the rest has no stage equivalent. Shown to whoever is
#: converting, because "cannot convert" without a reason is not actionable.
NOT_CONVERTIBLE = {
    "on_progress": (
        "fires partway through the parent; a stage dependency has no notion of "
        "partway"
    ),
    "on_fail": (
        "branches on the parent failing; a stage dependency only expresses "
        "'after it succeeded'"
    ),
    "on_any_end": (
        "fires on success or failure alike; a stage dependency cannot say "
        "'either way'"
    ),
}


#: ``{topic}`` in a chain's goal_template. Chains fill these at launch from the
#: variables the launcher is given; a pipeline's goal is literal, so an
#: unfilled placeholder would reach an agent as the characters "{topic}".
_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def placeholders(chain_steps: Any) -> List[str]:
    """The variable names a chain's goals expect, in the order first seen.

    A chain is often a template: "Research {topic} comprehensively", filled at
    launch. A pipeline has no such step -- its goal is the goal. So converting
    one without saying what the variables were would produce a pipeline that
    runs against the literal text, which reads as working and is not.
    """
    found: List[str] = []
    for step in chain_steps if isinstance(chain_steps, list) else []:
        if not isinstance(step, Mapping):
            continue
        for field in ("goal_template", "step_name"):
            for name in _PLACEHOLDER.findall(str(step.get(field) or "")):
                if name not in found:
                    found.append(name)
    return found


class ChainNotConvertible(Exception):
    """A chain that would lose meaning as a pipeline.

    Carries the offending steps so the caller can say which ones and why,
    rather than only that something failed.
    """

    def __init__(self, chain_name: str, reasons: List[Tuple[str, str, str]]):
        self.chain_name = chain_name
        #: (step name, trigger condition, why it cannot convert)
        self.reasons = reasons
        detail = "; ".join(
            f"{step} ({trigger}): {why}" for step, trigger, why in reasons
        )
        super().__init__(f"{chain_name} cannot become a pipeline -- {detail}")


def _stage_id(step: Mapping[str, Any], index: int) -> str:
    """A stage id from the step's name, or its position if it has none."""
    raw = str(step.get("step_name") or "").strip()
    if not raw:
        return f"stage_{index + 1}"
    slug = "".join(c if c.isalnum() else "_" for c in raw.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or f"stage_{index + 1}"


def describe(chain_steps: Any) -> List[Tuple[str, str, str]]:
    """The steps that cannot convert, as (step, trigger, why).

    Empty means the chain converts cleanly. Callers that want to report before
    converting use this; ``convert`` raises on the same condition.
    """
    problems: List[Tuple[str, str, str]] = []
    for index, step in enumerate(chain_steps if isinstance(chain_steps, list) else []):
        if not isinstance(step, Mapping):
            continue
        trigger = str(step.get("trigger_condition") or "on_complete").strip()
        if trigger in NOT_CONVERTIBLE:
            problems.append(
                (
                    str(step.get("step_name") or f"step {index + 1}"),
                    trigger,
                    NOT_CONVERTIBLE[trigger],
                )
            )
    return problems


def _fill(text: str, variables: Optional[Mapping[str, Any]]) -> str:
    """Substitute the way the chain launcher does, so the result is the same."""
    out = text
    for key, value in (variables or {}).items():
        out = out.replace(f"{{{key}}}", str(value))
    return out


def convert(
    *,
    name: str,
    chain_steps: Any,
    default_config: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The pipeline spec equivalent of a chain, or raise saying why not.

    Stages come out in the chain's order, each depending on the one before it --
    a chain is a path, which is a DAG with one branch. Contracts are left empty:
    the chain never said what a step had to achieve, and inventing a contract
    here would assert something its author never did. A converted pipeline is
    therefore a faithful chain, not a better one; the contracts are what its
    author adds next, which is the whole reason for moving.
    """
    problems = describe(chain_steps)
    if problems:
        raise ChainNotConvertible(name, problems)

    stages: List[Dict[str, Any]] = []
    previous_id: Optional[str] = None
    for index, step in enumerate(chain_steps if isinstance(chain_steps, list) else []):
        if not isinstance(step, Mapping):
            continue
        stage_id = _stage_id(step, index)
        # Two steps can share a name; a stage id may not.
        if any(existing["id"] == stage_id for existing in stages):
            stage_id = f"{stage_id}_{index + 1}"

        stage: Dict[str, Any] = {
            "id": stage_id,
            # Filled here or not at all: a pipeline's goal is literal, so an
            # unfilled {topic} would reach the agent as those characters.
            "goal": _fill(
                str(step.get("goal_template") or step.get("step_name") or "").strip(),
                variables,
            ),
            "job_type": str(step.get("job_type") or "research").strip(),
            "contract": {},
        }
        if previous_id:
            stage["depends_on"] = [previous_id]

        trigger = str(step.get("trigger_condition") or "on_complete").strip()
        if trigger == "on_approval":
            # The chain's way of saying "stop for a person"; the stage has a
            # word for it.
            stage["checkpoint"] = True
        elif trigger == "on_findings":
            # The successors start while this stage keeps running. The chain
            # put the threshold in trigger_thresholds; the stage says it plainly.
            thresholds = step.get("trigger_thresholds")
            threshold = 1
            if isinstance(thresholds, Mapping):
                try:
                    threshold = max(1, int(thresholds.get("findings_threshold") or 1))
                except (TypeError, ValueError):
                    threshold = 1
            stage["spawn_on"] = {"findings": threshold}

        config = step.get("config")
        merged: Dict[str, Any] = {}
        if isinstance(default_config, Mapping):
            merged.update(dict(default_config))
        if isinstance(config, Mapping):
            merged.update(dict(config))
        if merged:
            stage["config"] = merged

        stages.append(stage)
        previous_id = stage_id

    return {"name": name, "stages": stages}


__all__ = [
    "CONVERTIBLE",
    "placeholders",
    "NOT_CONVERTIBLE",
    "ChainNotConvertible",
    "convert",
    "describe",
]
