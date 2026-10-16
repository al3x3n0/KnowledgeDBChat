"""Where a run's time went: per job, per iteration, per phase and per tool.

ucb-bar/chia profiles every node call (where it ran, how long, what it
depended on) and renders a timeline and an aggregated table, because a long
experiment that is slow for an unknown reason cannot be made faster. A run
here has the same problem at a coarser grain: a pipeline that took three
hours took them somewhere -- in the queue, waiting on the model, waiting on a
simulator, or between stages waiting for a person or a beat tick -- and the
log said only when each iteration ended.

This reads what is already recorded. A run is the job and every job sharing
its root (a pipeline's stages, a swarm's roles). For each job: time queued
(created to started), time running (started to completed, or to now), the
gap after the job it follows ended (which includes any approval it waited
for), and each iteration's span from the log's timestamps, split by phase
where the runner timed them (`phase_ms`, recorded since this module was
added -- older iterations have the span only). The totals add the phases and
group the act time by tool, which is the table chia's profiler prints.

Times are what the database and the log say. A worker killed mid-run leaves
a job "running" until the sweep notices, and its running time then includes
that wait; the profile reports it as it reads and does not guess.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.models.agent_job import AgentJob
from app.utils.datetimes import as_aware_utc, utc_now

PHASES = ("observe", "think", "act", "evaluate")
#: A run larger than this is profiled for its first jobs only, and says so.
MAX_JOBS = 50


def _when(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    return as_aware_utc(value)


def _seconds(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    return round(max(0.0, (end - start).total_seconds()), 1)


def _iterations(job: AgentJob) -> List[Dict[str, Any]]:
    entries = [
        e
        for e in (job.execution_log or [])
        if isinstance(e, dict) and e.get("phase") == "iteration_complete"
    ]
    previous = _when(job.started_at)
    out: List[Dict[str, Any]] = []
    for entry in entries:
        ended = _when(entry.get("timestamp"))
        phase_ms = (
            entry.get("phase_ms") if isinstance(entry.get("phase_ms"), dict) else None
        )
        out.append(
            {
                "iteration": entry.get("iteration"),
                "tool": entry.get("action"),
                "success": entry.get("success"),
                "ended_at": ended.isoformat() if ended else None,
                "seconds": _seconds(previous, ended),
                "phase_ms": phase_ms,
            }
        )
        previous = ended or previous
    return out


def _add_phases(into: Dict[str, int], phase_ms: Optional[Dict[str, Any]]) -> None:
    for phase in PHASES:
        try:
            into[phase] += int((phase_ms or {}).get(phase) or 0)
        except (TypeError, ValueError):
            continue


def _add_tool(into: Dict[str, Dict[str, Any]], iteration: Dict[str, Any]) -> None:
    tool = str(iteration.get("tool") or "")
    if not tool:
        return
    row = into.setdefault(tool, {"tool": tool, "calls": 0, "failures": 0, "act_ms": 0})
    row["calls"] += 1
    if iteration.get("success") is False:
        row["failures"] += 1
    try:
        row["act_ms"] += int((iteration.get("phase_ms") or {}).get("act") or 0)
    except (TypeError, ValueError):
        pass


def _job_profile(
    job: AgentJob, by_id: Dict[Any, AgentJob], now: datetime
) -> Dict[str, Any]:
    created, started = _when(job.created_at), _when(job.started_at)
    completed = _when(job.completed_at)
    parent = by_id.get(job.parent_job_id)
    iterations = _iterations(job)
    phases = {phase: 0 for phase in PHASES}
    tools: Dict[str, Dict[str, Any]] = {}
    for iteration in iterations:
        _add_phases(phases, iteration["phase_ms"])
        _add_tool(tools, iteration)
    config = job.config if isinstance(job.config, dict) else {}
    # A job that stopped without completing (paused for a person, blocked)
    # ran until its last log entry, not until now. Measured to now, a
    # blocked stage of the first profiled run read 9.6 hours "running".
    paused = completed is None and str(job.status or "") not in ("running", "pending")
    logged = [
        _when(e.get("timestamp"))
        for e in (job.execution_log or [])
        if isinstance(e, dict)
    ]
    last_logged = max((t for t in logged if t), default=None)
    stopped = (last_logged or started) if paused else (now if started else None)
    return {
        "id": str(job.id),
        "parent_id": str(job.parent_job_id) if job.parent_job_id else None,
        "name": job.name,
        "stage": config.get("pipeline_stage"),
        "status": job.status,
        "created_at": created.isoformat() if created else None,
        "started_at": started.isoformat() if started else None,
        "completed_at": completed.isoformat() if completed else None,
        "queued_seconds": _seconds(created, started or (None if completed else now)),
        "ran_seconds": _seconds(started, completed or stopped),
        "paused_seconds": _seconds(stopped, now) if paused else None,
        # From the job it follows ending to this one existing: a chain's
        # handoff, and any approval the plan waited for.
        "after_parent_seconds": _seconds(
            _when(parent.completed_at) if parent is not None else None, created
        ),
        "iterations": iterations,
        "phase_ms": phases,
        "tools": sorted(tools.values(), key=lambda r: -r["act_ms"]),
    }


def build(
    jobs: Iterable[AgentJob], *, root_id: Any, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """The profile of the jobs given, as one run rooted at `root_id`."""
    now = now or utc_now()
    ordered = sorted(jobs, key=lambda j: _when(j.created_at) or now)
    truncated = len(ordered) > MAX_JOBS
    ordered = ordered[:MAX_JOBS]
    by_id = {job.id: job for job in ordered}
    profiles = [_job_profile(job, by_id, now) for job in ordered]

    phases = {phase: 0 for phase in PHASES}
    tools: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        _add_phases(phases, profile["phase_ms"])
        for row in profile["tools"]:
            total = tools.setdefault(
                row["tool"],
                {"tool": row["tool"], "calls": 0, "failures": 0, "act_ms": 0},
            )
            for key in ("calls", "failures", "act_ms"):
                total[key] += row[key]

    starts = [_when(j.created_at) for j in ordered if j.created_at]
    # Where each job stopped, so a run whose last stage is paused for a
    # person does not keep growing its wall time while nobody answers.
    ends = [
        _when(p["completed_at"])
        or (
            _when(p["started_at"]) + timedelta(seconds=p["ran_seconds"])
            if p["started_at"] and p["ran_seconds"] is not None
            else now
        )
        for p in profiles
    ]
    timed = sum(
        1 for p in profiles for i in p["iterations"] if i.get("phase_ms") is not None
    )
    untimed = sum(1 for p in profiles for i in p["iterations"]) - timed
    return {
        "root_job_id": str(root_id),
        "jobs": profiles,
        "wall_seconds": _seconds(min(starts), max(ends)) if starts and ends else None,
        "totals": {
            "queued_seconds": round(sum(p["queued_seconds"] or 0 for p in profiles), 1),
            "ran_seconds": round(sum(p["ran_seconds"] or 0 for p in profiles), 1),
            "after_parent_seconds": round(
                sum(p["after_parent_seconds"] or 0 for p in profiles), 1
            ),
            "phase_ms": phases,
            "tools": sorted(tools.values(), key=lambda r: -r["act_ms"]),
        },
        "iterations_without_phase_times": untimed,
        "truncated": truncated,
    }


async def load_run(db: AsyncSession, job: AgentJob) -> List[AgentJob]:
    """The job and every job sharing its root, with their logs."""
    root = job.root_job_id or job.id
    rows = await db.execute(
        select(AgentJob)
        .options(undefer(AgentJob.execution_log))
        .where(or_(AgentJob.id == root, AgentJob.root_job_id == root))
        .order_by(AgentJob.created_at)
        .limit(MAX_JOBS + 1)
    )
    return list(rows.scalars())


def root_of(job: AgentJob) -> UUID:
    return job.root_job_id or job.id
