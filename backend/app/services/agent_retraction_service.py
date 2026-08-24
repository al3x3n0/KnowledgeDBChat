"""Withdraw a result, and take down what rested on it.

The retraction itself is one row. The work is the propagation, because the
whole point is that a defect found late reaches everything the defect touched
rather than only the place it was noticed.

What propagates, and why each one matters:

* **Method standing.** A method's record is the runs that carried it. Runs
  scored against retracted measurements have to leave that record, or a method
  validated entirely against defective numbers keeps recommending itself and
  the ranking that uses standing keeps promoting it.
* **Method validation.** A method whose cited evidence has all been retracted
  is no longer validated by anything. It reverts to unvalidated rather than
  disappearing -- the procedure may still be sound and only its evidence gone,
  which is a different thing from being wrong.

Propagation is computed on read rather than written into the things it touches.
A retraction can itself be withdrawn -- a measurement re-taken and found good
after all -- and a system that had already rewritten every dependent record
would have no way back. It also means a retraction added today applies to
records written last week without a migration.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_method_outcome import AgentMethodOutcome
from app.models.agent_retraction import AgentRetraction, RetractionKind


async def retract(
    db: AsyncSession,
    *,
    user_id: Any,
    kind: str,
    ref: Any,
    reason: str,
    source: Optional[str] = None,
    source_job_id: Optional[Any] = None,
) -> AgentRetraction:
    """Record that something is no longer believed, and why."""
    if str(kind) not in RetractionKind.ALL:
        raise ValueError(
            f"unknown retraction kind {kind!r}; expected one of "
            f"{', '.join(RetractionKind.ALL)}"
        )
    if not str(reason or "").strip():
        raise ValueError(
            "a retraction needs a reason: a later run has to tell a result "
            "withdrawn for a harness defect from one withdrawn because the "
            "question changed, and only the reason distinguishes them"
        )
    if not str(ref or "").strip():
        raise ValueError("a retraction needs a subject")

    row = AgentRetraction(
        user_id=user_id,
        subject_kind=str(kind),
        subject_ref=str(ref)[:300],
        reason=str(reason).strip(),
        source=str(source)[:200] if source else None,
        source_job_id=source_job_id,
    )
    db.add(row)
    await db.flush()
    return row


async def withdraw(db: AsyncSession, retraction_id: Any) -> bool:
    """Undo a retraction -- the measurement was re-taken and held after all.

    Possible only because propagation is computed on read. A system that had
    rewritten every dependent record on retraction would have nothing to
    restore them from.
    """
    row = await db.get(AgentRetraction, retraction_id)
    if row is None:
        return False
    await db.delete(row)
    await db.flush()
    return True


async def retractions(
    db: AsyncSession, user_id: Any, kind: Optional[str] = None
) -> List[AgentRetraction]:
    query = select(AgentRetraction).where(AgentRetraction.user_id == user_id)
    if kind:
        query = query.where(AgentRetraction.subject_kind == str(kind))
    rows = await db.execute(query.order_by(AgentRetraction.created_at.desc()))
    return list(rows.scalars())


async def retracted_refs(db: AsyncSession, user_id: Any, kind: str) -> Set[str]:
    rows = await retractions(db, user_id, kind)
    return {str(r.subject_ref) for r in rows}


async def reasons_for(db: AsyncSession, user_id: Any, kind: str) -> Dict[str, str]:
    rows = await retractions(db, user_id, kind)
    return {str(r.subject_ref): str(r.reason) for r in rows}


# --- propagation ----------------------------------------------------------


async def live_outcomes(
    db: AsyncSession, user_id: Any, outcomes: Iterable[AgentMethodOutcome]
) -> List[AgentMethodOutcome]:
    """The outcomes that still count toward a method's standing.

    A run scored against retracted measurements says nothing about whether the
    method worked, so it leaves the record entirely rather than counting as a
    failure. Counting it as a failure would punish a method for evidence that
    was withdrawn, which is a different claim from the method having failed.
    """
    retracted_jobs = await retracted_refs(db, user_id, RetractionKind.JOB)
    retracted_methods = await retracted_refs(db, user_id, RetractionKind.METHOD)
    if not retracted_jobs and not retracted_methods:
        return list(outcomes)

    kept = []
    for outcome in outcomes:
        if str(getattr(outcome, "job_id", "") or "") in retracted_jobs:
            continue
        if str(getattr(outcome, "method_memory_id", "") or "") in retracted_methods:
            continue
        kept.append(outcome)
    return kept


def _cited_types(record: Mapping[str, Any]) -> List[str]:
    """The finding types a method record cites.

    `evidence` is the stored key; `derived_from` is only the name of the
    parameter that produced it, and reading the parameter name off a stored
    record silently finds nothing -- which reads exactly like a method with no
    citations, so nothing would ever be retracted.
    """
    cited = record.get("evidence")
    if cited is None:
        cited = record.get("derived_from")
    if isinstance(cited, str):
        return [cited]
    if isinstance(cited, (list, tuple)):
        return [str(x) for x in cited]
    return []


async def method_evidence_status(
    db: AsyncSession, user_id: Any, record: Mapping[str, Any]
) -> Dict[str, Any]:
    """Whether a recorded method's evidence still stands.

    A method citing three finding types, two of them retracted, is weakened but
    not unvalidated -- one piece of live evidence is still evidence. Only when
    every cited type has been withdrawn does the method lose its validation.
    """
    from app.services import agent_method_record

    cited = [
        c for c in _cited_types(record) if c and c != agent_method_record.NO_EVIDENCE
    ]
    if not cited:
        return {
            "status": record.get("status"),
            "retracted_evidence": [],
            "changed": False,
        }

    retracted = await retracted_refs(db, user_id, RetractionKind.FINDING_TYPE)
    gone = [c for c in cited if c in retracted]
    if not gone:
        return {
            "status": record.get("status"),
            "retracted_evidence": [],
            "changed": False,
        }

    all_gone = len(gone) == len(cited)
    status = agent_method_record.UNVALIDATED if all_gone else record.get("status")
    return {
        "status": status,
        "retracted_evidence": gone,
        "surviving_evidence": [c for c in cited if c not in retracted],
        "changed": all_gone,
    }


async def affected(db: AsyncSession, user_id: Any) -> Dict[str, Any]:
    """What the current retractions take down, for an operator or a report."""
    jobs = await retracted_refs(db, user_id, RetractionKind.JOB)
    types = await retracted_refs(db, user_id, RetractionKind.FINDING_TYPE)
    methods = await retracted_refs(db, user_id, RetractionKind.METHOD)

    rows = await db.execute(
        select(AgentMethodOutcome).where(AgentMethodOutcome.user_id == user_id)
    )
    outcomes = list(rows.scalars())
    dropped = [
        o
        for o in outcomes
        if str(o.job_id or "") in jobs or str(o.method_memory_id or "") in methods
    ]

    by_method: Dict[str, int] = {}
    for outcome in dropped:
        by_method[str(outcome.method_name)] = (
            by_method.get(str(outcome.method_name), 0) + 1
        )

    return {
        "retracted_jobs": len(jobs),
        "retracted_finding_types": sorted(types),
        "retracted_methods": len(methods),
        "outcomes_total": len(outcomes),
        "outcomes_dropped": len(dropped),
        "methods_losing_runs": by_method,
    }


def describe(rows: Sequence[AgentRetraction]) -> List[str]:
    """One line per retraction, for a digest or a prompt."""
    lines = []
    for row in rows:
        lines.append(f"{row.subject_kind} {row.subject_ref} is retracted: {row.reason}")
    return lines


async def note_for_prompt(db: AsyncSession, user_id: Any, limit: int = 5) -> str:
    """What a run needs to know before it cites anything.

    Injected rather than left to be discovered, because a run has no way to
    tell that a finding type it is about to rely on was withdrawn last week.
    """
    rows = await retractions(db, user_id)
    if not rows:
        return ""
    lines = describe(rows[:limit])
    more = "" if len(rows) <= limit else f" (and {len(rows) - limit} more)"
    return (
        "Retracted -- do not cite these as evidence, and do not treat a result "
        "derived from them as established:\n- " + "\n- ".join(lines) + more
    )


async def apply_to_standing(
    db: AsyncSession, user_id: Any, outcomes: Iterable[AgentMethodOutcome]
) -> Dict[str, Any]:
    """Summarise a method's standing with retracted runs removed."""
    from app.services import agent_method_standing_service as standing

    kept = await live_outcomes(db, user_id, outcomes)
    summary = standing.summarize(kept)
    dropped = len(list(outcomes)) - len(kept)
    if dropped:
        summary["runs_retracted"] = dropped
        logger.debug(f"Standing excludes {dropped} retracted run(s)")
    return summary
