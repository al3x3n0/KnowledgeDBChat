"""Draft something with a model, and repair it against the thing that judges it.

Three drafters -- plugins, agent definitions, sandbox skills -- each wrote the
same loop: ask for JSON, hand it to the real validator, and when the validator
refuses, give the refusal back and ask again. The loop is the same because the
idea is: a model told what is wrong with its draft fixes most of it, and a
model asked in advance to be careful does not.

What differs between them is the judge, so that is the one thing a caller
supplies. The plumbing they each carried a copy of lives here: the model call
and what to do when it fails, a reply that is not JSON, the growing message,
the notes a person reads afterwards, and the progress callback.

The pipeline drafter is deliberately not built on this. It makes exactly one
repair round, resends the whole draft, and keeps whichever version is less
broken -- a different procedure, not this one with different settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

from app.services import llm_json

#: Called as the loop turns: (stage, attempt, notes so far). The notes are the
#: progress worth showing -- "fixing: id must be lowercase" says more than a
#: spinner -- and the repair loop makes the slow case the common one.
Progress = Callable[[str, int, List[str]], None]


@dataclass
class Verdict:
    """What the judge made of one draft."""

    #: The candidate, when there is one worth keeping. A draft whose flaw a
    #: person can see and fix is kept even though it is being sent back.
    value: Any = None
    #: What to tell the model. Empty means the draft is accepted.
    complaint: str = ""
    #: What to tell the person, if not the complaint itself.
    note: str = ""
    #: What the model should do about the complaint.
    instruction: str = "Fix exactly that and return the whole thing again."
    #: No further attempt could help: the failure is not in the draft.
    stop: bool = False
    #: Forget any candidate kept from an earlier attempt.
    discard: bool = False


@dataclass
class Outcome:
    value: Any = None
    notes: List[str] = field(default_factory=list)
    attempts: int = 0


Judge = Callable[[Dict[str, Any], int], Awaitable[Verdict]]


def _report(on_progress: Optional[Progress], stage: str, attempt: int, notes) -> None:
    if on_progress is None:
        return
    try:
        on_progress(stage, attempt, list(notes))
    except Exception as exc:  # pragma: no cover - defensive
        # Progress is a courtesy. A caller whose reporting breaks must not
        # take the draft down with it.
        logger.warning(f"Draft progress callback failed: {exc}")


async def run(
    *,
    system: str,
    message: str,
    schema: Dict[str, Any],
    judge: Judge,
    max_attempts: int = 3,
    user_id: Any = None,
    db: Any = None,
    llm: Any = None,
    task_type: str = "balanced",
    on_progress: Optional[Progress] = None,
    snapshot_phase: Optional[str] = None,
    what: str = "draft",
) -> Outcome:
    """Ask, judge, and ask again with the judge's complaint, up to a limit.

    Returns the last candidate the judge kept (possibly None), the notes, and
    how many attempts were made. Never raises for a bad reply or an
    unreachable model: both are facts about the draft, recorded in the notes.
    """
    if llm is None:
        from app.services.llm_service import LLMService

        llm = LLMService()

    outcome = Outcome()
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        outcome.attempts = attempt
        _report(on_progress, "drafting", attempt, outcome.notes)
        try:
            extra = (
                {"snapshot_context": {"phase": snapshot_phase}}
                if snapshot_phase
                else {}
            )
            completion = await llm.generate_structured(
                system_prompt=system,
                user_message=message,
                response_schema=schema,
                task_type=task_type,
                user_id=user_id,
                db=db,
                **extra,
            )
        except Exception as exc:
            logger.warning(f"{what} call failed on attempt {attempt}: {exc}")
            outcome.notes.append(f"The model could not be reached: {exc}")
            break

        payload = llm_json.completion_object(completion)
        if not payload:
            outcome.notes.append(f"Attempt {attempt}: the reply was not JSON.")
            message = (
                f"{message}\n\nYour last reply was not a JSON object. Reply "
                "with one JSON object and nothing else."
            )
            continue

        _report(on_progress, "checking", attempt, outcome.notes)
        verdict = await judge(payload, attempt)
        if verdict.discard:
            outcome.value = None
        if verdict.value is not None:
            outcome.value = verdict.value

        if not verdict.complaint:
            break
        if verdict.stop:
            outcome.notes.append(verdict.note or verdict.complaint)
            break

        outcome.notes.append(f"Attempt {attempt}: {verdict.note or verdict.complaint}")
        message = f"{message}\n\n{verdict.complaint}\n\n{verdict.instruction}"

    _report(on_progress, "done", outcome.attempts, outcome.notes)
    return outcome
