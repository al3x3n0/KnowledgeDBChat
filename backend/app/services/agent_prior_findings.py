"""Reading what earlier runs measured, so a new one need not measure it again.

Every run's findings are persisted, typed and complete -- 260 jobs in this
database hold them, with the numbers, the subject and the measurement source
still attached. Nothing could read them. `get_research_findings` serves
`executor._job_findings[job.id]`, an in-memory dict scoped to the run asking,
so a pipeline began every time from nothing and re-derived baselines that were
already on disk.

What crossed instead was prose. A completed job is LLM-summarised into
`finding`/`insight`/`lesson` memories and those are injected into later runs,
which carries *that* prefetching helped a strided scan and loses *by how much,
against what baseline, on which core*. That is the wrong half to keep for work
whose entire product is a number.

TWO RULES SHAPE THIS MODULE.

**Recalled evidence is citable.** `record_prediction` and `record_method`
check `derived_from` against the finding types the run has, and refuse a
citation to evidence that does not exist -- the check that caught a run
predicting from an llvm-mca result it never obtained. Recalled findings enter
the run's findings so that check keeps working unchanged, and each carries the
job it came from, so the record says what it rests on.

**A recall never reads another recall.** Recalled findings are written into
the recalling job's own results, and that job is then in the window the next
recall scans -- so without a guard a number is copied forward hop by hop, and
`_provenance` rewrites `recalled_from_job` at every hop to name the previous
*copier*. A live run recalled fifty findings all stamped as coming from one
job, which had itself measured nothing and held thirty-two copies drawn from
seven other jobs. The citation named a job that never ran the simulator. That
is worse than losing the provenance, because the record still looks complete.
Only first-hand findings are recalled; the job that measured a number is
always still in the window, so nothing is lost by skipping the copies.

**Recalled evidence never satisfies a contract.** A goal contract asking for
two `mechanism_comparison` findings is the thing that makes a run do the work.
If a recall could fill it, the cheapest way to satisfy any contract would be
to look up two old numbers and stop. Every recalled finding is stamped
`recalled: True`, and the contract counter skips those.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

#: Most findings one recall may return. A recall is meant to bring back the
#: handful of numbers a run needs to orient itself, not to replay a corpus
#: into a prompt that then has no room for the work.
MAX_RESULTS = 25

#: How many recent jobs to read before filtering. Findings live inside a JSON
#: column, so the filtering happens in Python: SQLite backs the tests and the
#: JSON operators that would push this into the database are Postgres-only.
#: Scanning a bounded window of recent jobs is portable and, at a few hundred
#: rows, not worth optimising into something the tests cannot exercise.
SCAN_LIMIT = 200


def _matches(
    finding: Dict[str, Any],
    types: Sequence[str],
    subject: str,
) -> bool:
    """Whether one stored finding answers the request."""
    if not isinstance(finding, dict):
        return False
    if not _is_first_hand(finding):
        return False
    ftype = str(finding.get("type") or "").strip()
    if not ftype:
        return False
    if types and ftype not in types:
        return False
    if subject:
        # Every word, anywhere in the record -- not the phrase, and not in
        # three chosen fields. A live run asked for "L2 prefetcher" and got
        # nothing while the database held `l2.prefetcher=StridePrefetcher`:
        # the words were there, adjacent nowhere, and in a field the filter
        # was not reading. A filter narrower than the record it searches
        # returns an empty answer that looks like an absence of evidence.
        haystack = _searchable_text(finding)
        if not all(token in haystack for token in subject.lower().split()):
            return False
    return True


def _is_first_hand(finding: Dict[str, Any]) -> bool:
    """Whether this job measured the number, rather than recalling it.

    A recall stores what it recalled into its own results, so the copies are
    in the window the next recall scans. Following them builds a chain whose
    every hop overwrites the source, until the record cites a job that never
    took the measurement.
    """
    return not bool(finding.get("recalled"))


def _searchable_text(finding: Dict[str, Any]) -> str:
    """Every string in the record, lowercased, for matching against.

    Numbers included, so a subject can name one: a caller looking for the run
    that measured 1.7054 should be able to ask for it.
    """
    parts: List[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                parts.append(str(key))
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif value is not None and not isinstance(value, bool):
            parts.append(str(value))

    walk(finding)
    return " ".join(parts).lower()


def _provenance(job: Any, finding: Dict[str, Any]) -> Dict[str, Any]:
    """The finding as the recalling run will see it.

    `recalled` is what keeps this out of the contract count, and the job it
    came from is what lets a reader tell a number this run measured from one
    it looked up. Both belong on the record rather than in a log line.
    """
    recalled = dict(finding)
    recalled["recalled"] = True
    recalled["recalled_from_job"] = str(getattr(job, "id", "") or "")
    goal = str(getattr(job, "goal", "") or "")
    recalled["recalled_from_goal"] = goal[:200]
    completed = getattr(job, "completed_at", None) or getattr(job, "created_at", None)
    recalled["recalled_at"] = completed.isoformat() if completed else None
    return recalled


async def recall(
    *,
    db: AsyncSession,
    user_id: Any,
    exclude_job_id: Any = None,
    finding_types: Optional[Sequence[str]] = None,
    subject: str = "",
    job_type: str = "",
    limit: int = 10,
) -> Dict[str, Any]:
    """Findings from this user's earlier jobs, newest first."""
    from app.models.agent_job import AgentJob

    types = [str(t).strip() for t in (finding_types or []) if str(t).strip()]
    subject = str(subject or "").strip()
    try:
        limit = max(1, min(int(limit or 10), MAX_RESULTS))
    except (TypeError, ValueError):
        limit = 10

    stmt = select(AgentJob).where(AgentJob.user_id == user_id)
    if exclude_job_id is not None:
        stmt = stmt.where(AgentJob.id != exclude_job_id)
    if job_type:
        stmt = stmt.where(AgentJob.job_type == job_type)
    stmt = stmt.order_by(AgentJob.created_at.desc()).limit(SCAN_LIMIT)

    result = await db.execute(stmt)
    jobs = list(result.scalars().all())

    matched: List[Dict[str, Any]] = []
    seen_types: Dict[str, int] = {}
    for job in jobs:
        results = getattr(job, "results", None)
        if not isinstance(results, dict):
            continue
        findings = results.get("findings")
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not _matches(finding, types, subject):
                continue
            matched.append(_provenance(job, finding))
            ftype = str(finding.get("type") or "").strip()
            seen_types[ftype] = seen_types.get(ftype, 0) + 1
            if len(matched) >= limit:
                break
        if len(matched) >= limit:
            break

    return {
        "success": True,
        "findings": matched,
        "count": len(matched),
        "types_found": seen_types,
        "jobs_scanned": len(jobs),
        "note": (
            "These were measured by earlier runs, not by this one. They can be "
            "cited in derived_from, and each says which job produced it. They "
            "do NOT count toward this run's goal contract -- recalling a "
            "number is not the same as establishing one, and a contract that "
            "a lookup could satisfy would stop asking for the work."
        ),
    }


async def available_types(
    *, db: AsyncSession, user_id: Any, exclude_job_id: Any = None
) -> Dict[str, int]:
    """What kinds of evidence this user's earlier runs actually produced.

    Offered because a caller cannot guess the vocabulary: the types are
    whatever the tools that ran chose to emit, and asking for one that was
    never produced returns nothing, with no hint that the name was the problem.
    """
    from app.models.agent_job import AgentJob

    stmt = select(AgentJob).where(AgentJob.user_id == user_id)
    if exclude_job_id is not None:
        stmt = stmt.where(AgentJob.id != exclude_job_id)
    stmt = stmt.order_by(AgentJob.created_at.desc()).limit(SCAN_LIMIT)

    result = await db.execute(stmt)
    counts: Dict[str, int] = {}
    for job in result.scalars().all():
        results = getattr(job, "results", None)
        if not isinstance(results, dict):
            continue
        findings = results.get("findings")
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict) or not _is_first_hand(finding):
                continue
            ftype = str(finding.get("type") or "").strip()
            if ftype:
                counts[ftype] = counts.get(ftype, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
