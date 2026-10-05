"""Keep what a run learned about *calling* a tool, not just about its subject.

Methods already carry what transfers between jobs -- 415 of them in this
deployment, on kernels, prefetchers and A/B design. Not one is about how to
call a tool, and that is the knowledge runs keep paying for twice.

Measured: a gem5 study passed a mechanism at the top level of a config, was
refused with the accepted shape spelled out (``a mechanism is named inside
`caches`, like {"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}``),
corrected itself, and finished. The next study made the identical mistake and
spent five iterations on it, because the first one's lesson lived only in a
conversation that had ended.

So: when a tool refuses a call, and a later call to that same tool succeeds,
the refusal was a lesson and the run has just proved the correction works.
That pair is recorded as a method.

Two things are deliberately not recorded. The arguments that worked, because
they routinely contain whole programs and the reusable part is the *shape* the
tool described, not one kernel's source. And a refusal never followed by a
success, because a correction nobody has demonstrated is a guess -- which is
the distinction ``agent_method_record`` exists to enforce.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: How much of the tool's refusal to keep. Long enough for the corrective
#: sentence, short enough that a stack trace does not become the lesson.
MAX_REFUSAL_CHARS = 400


def _actions(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    taken = state.get("actions_taken") if isinstance(state, Mapping) else None
    return [a for a in (taken or []) if isinstance(a, Mapping)]


def _tool_of(entry: Mapping[str, Any]) -> str:
    action = entry.get("action") if isinstance(entry.get("action"), Mapping) else {}
    return str(action.get("tool") or entry.get("tool") or "").strip()


def corrections(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Refusals this run recovered from, one per tool.

    A correction is a failure that the tool itself judged -- it ran, read the
    call and said no -- followed later by a success on the same tool. A tool
    that could not run at all teaches nothing about how to call it, and a
    failure the run never recovered from proves nothing.
    """
    from app.services import agent_failure_diagnosis as diagnosis

    pending: Dict[str, Dict[str, Any]] = {}
    learned: Dict[str, Dict[str, Any]] = {}

    for entry in _actions(state):
        tool = _tool_of(entry)
        if not tool:
            continue
        result = entry.get("result") if isinstance(entry.get("result"), Mapping) else {}
        error = result.get("error")
        succeeded = bool(result.get("success")) and not error

        if succeeded:
            refusal = pending.pop(tool, None)
            if refusal and tool not in learned:
                produced = [
                    str(f.get("type")).strip()
                    for f in (result.get("findings") or [])
                    if isinstance(f, Mapping) and str(f.get("type") or "").strip()
                ]
                learned[tool] = {
                    "tool": tool,
                    "refusal": refusal["refusal"],
                    "error_class": refusal["error_class"],
                    "produced": produced,
                }
            continue

        if not error:
            continue
        # A tool that never ran says nothing about how to call it, and neither
        # does the run's own broken source.
        if diagnosis.could_not_run(error) or diagnosis.blames_the_submitted_code(error):
            continue
        pending.setdefault(
            tool,
            {
                "refusal": str(error)[:MAX_REFUSAL_CHARS],
                "error_class": diagnosis.classify_error(error),
            },
        )

    return list(learned.values())


def _classified(correction: Mapping[str, Any]) -> str:
    """Name the failure class, or say nothing when it was not classified.

    ``classify_error`` returns ``"unknown"`` for a refusal whose wording no
    bucket matches, and a sentence reading "refused for its shape (unknown)"
    tells the reader less than one that stops at "shape".
    """
    error_class = str(correction.get("error_class") or "").strip()
    if not error_class or error_class == "unknown":
        return ""
    return f" ({error_class})"


def build(
    correction: Mapping[str, Any], available_finding_types: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """A method record for one recovered refusal, or None if it cannot be built."""
    from app.services import agent_method_record

    tool = str(correction.get("tool") or "")
    refusal = str(correction.get("refusal") or "").strip()
    if not tool or not refusal:
        return None

    produced = [t for t in (correction.get("produced") or []) if t]
    # Cite what the corrected call produced. When it produced nothing citable
    # the method is still true and still worth having, so it is recorded
    # unvalidated rather than silently dropped or dressed up as evidence.
    derived_from = produced or [agent_method_record.NO_EVIDENCE]

    try:
        return agent_method_record.build_record(
            name=f"Calling {tool}: what it accepts",
            procedure=[
                f"Before calling {tool}, note the shape it refused once here: "
                f"{refusal}",
                "Make the call the way that message describes; a later call in "
                "that run succeeded.",
            ],
            prevents=(
                f"Spending iterations re-editing a call to {tool} that is "
                f"refused for its shape{_classified(correction)}, when the "
                "tool has already said what it accepts."
            ),
            derived_from=derived_from,
            available_finding_types=list(available_finding_types)
            + [agent_method_record.NO_EVIDENCE],
            applies_to=tool,
        )
    except Exception as exc:  # pragma: no cover - record validation
        logger.debug(f"Could not build a usage method for {tool}: {exc}")
        return None


async def record_corrections(job: Any, state: Mapping[str, Any], db: Any) -> int:
    """Store a method for each refusal this run recovered from. Returns how many."""
    from app.models.memory import ConversationMemory
    from app.services import agent_method_record

    available = [
        str(f.get("type")).strip()
        for f in (state.get("findings") or [])
        if isinstance(f, Mapping) and str(f.get("type") or "").strip()
    ]

    stored = 0
    for correction in corrections(state):
        record = build(correction, available)
        if not record:
            continue
        db.add(
            ConversationMemory(
                user_id=job.user_id,
                job_id=getattr(job, "id", None),
                memory_type=agent_method_record.MEMORY_TYPE,
                content=agent_method_record.render(record),
                # Below a validated subject method: knowing how to call a tool
                # matters less than knowing what is true about the thing being
                # studied, and this one is often recorded without evidence.
                importance_score=(
                    0.7 if record["status"] == agent_method_record.VALIDATED else 0.5
                ),
                tags=agent_method_record.tags_for(record),
                context={
                    "record": "method",
                    "status": record["status"],
                    "evidence": record["evidence"],
                    "learned": "tool_usage",
                },
            )
        )
        stored += 1
    if stored:
        logger.info("Recorded %s tool-usage method(s) from job %s", stored, job.id)
    return stored


__all__ = ["MAX_REFUSAL_CHARS", "build", "corrections", "record_corrections"]
