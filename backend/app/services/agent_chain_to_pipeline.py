"""Convert a saved job chain into a pipeline spec.

A chain and a pipeline produce the same thing: chained jobs, linked by
``chain_config``, created by ``create_chained_job``. They differ only in what
the author wrote down. A chain says *when* the next step fires; a pipeline says
*what must be true* when a stage is done and derives the rest. The pipeline is
the better of the two ways to say it, so chains are being retired as an
authoring concept -- the runtime they share is not going anywhere.

What does not convert
---------------------
A chain step's ``trigger_condition`` has six values and a pipeline stage's
``depends_on`` expresses one of them: "after the previous stage finished". Two
map cleanly:

    on_complete   -> depends_on the step before it
    on_approval   -> the same, plus checkpoint=True

The other four do not, and the reason is not cosmetic. ``on_findings`` fires the
next step *while the parent is still running* -- that is what makes a continuous
monitor able to raise alerts without ever finishing. ``depends_on`` waits for
the parent to finish, which such a monitor never does. ``on_fail`` and
``on_any_end`` branch on an outcome a DAG edge cannot express, and
``on_progress`` fires partway through.

So this refuses them rather than approximating them. A conversion that looked
successful and quietly turned a live monitor into a stage waiting for an end
that never comes is worse than no conversion at all.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Trigger conditions a pipeline stage can express, and how.
CONVERTIBLE = {
    "on_complete": "depends_on the preceding stage",
    "on_approval": "depends_on the preceding stage, held for review",
}

#: Why each of the rest has no stage equivalent. Shown to whoever is
#: converting, because "cannot convert" without a reason is not actionable.
NOT_CONVERTIBLE = {
    "on_findings": (
        "fires while the parent is still running, so the two run together; a "
        "stage dependency waits for the parent to finish"
    ),
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


def convert(
    *,
    name: str,
    chain_steps: Any,
    default_config: Optional[Mapping[str, Any]] = None,
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
            "goal": str(
                step.get("goal_template") or step.get("step_name") or ""
            ).strip(),
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
    "NOT_CONVERTIBLE",
    "ChainNotConvertible",
    "convert",
    "describe",
]
