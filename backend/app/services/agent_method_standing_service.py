"""Score a method by what happened to the runs that carried it.

`record_method` captures how to do something and the evidence that established
it; memory injection puts it in front of later jobs. Between those two there
was nothing: no record that a method had been reused, and no way to tell a
method that keeps preceding good work from one that keeps preceding failure.

This closes it with the same shape the calibration store uses for numbers. When
a job finishes, every method that was in its context gets a row saying what
became of that run -- whether its contract held, how many predictions it
settled and how far off they were. Standing is the aggregate, and it travels
back with the method the next time it is recalled.

Two things it deliberately does not claim:

*Carried is not followed.* A method in a run's context may have been ignored.
That is why `cited` exists and is counted separately: a run that named the
method it was building on is better evidence than one that merely had it to
hand, and merging the two would produce a number that reads as more than it is.

*A run's outcome is not the method's fault.* One row is an association, not a
cause; a good method can precede a run that failed for its own reasons. Only
the accumulation means much, so the report says how many runs it rests on and
stays quiet until there are any.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_method_outcome import AgentMethodOutcome
from app.models.agent_prediction import AgentPrediction
from app.services import agent_method_record

# Below this many runs a rate is noise dressed as a number.
MIN_RUNS_FOR_A_RATE = 3


def _method_memories(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Methods that were injected into this run, from its own state."""
    payloads = state.get("injected_memory_payloads")
    if not isinstance(payloads, list):
        return []
    methods: List[Dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        content = str(payload.get("content") or "")
        parsed = agent_method_record.parse(content)
        if not parsed:
            continue
        methods.append(
            {
                "id": str(payload.get("id") or ""),
                "name": parsed.get("name") or "",
                "content": content,
            }
        )
    return methods


def _cited_names(state: Mapping[str, Any]) -> List[str]:
    """Method names this run said it was building on."""
    cited: List[str] = []
    actions = state.get("actions_taken")
    if not isinstance(actions, list):
        return cited
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        if str(action.get("tool") or "") != "record_method":
            continue
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        builds_on = params.get("builds_on")
        if isinstance(builds_on, str):
            builds_on = [builds_on]
        if isinstance(builds_on, list):
            cited.extend(str(name).strip() for name in builds_on if str(name).strip())
    return cited


async def _prediction_scores(db: AsyncSession, job_id: Any) -> Dict[str, Any]:
    """How many claims this run settled, and how far off they were."""
    if job_id is None:
        return {"settled": 0, "mean_relative_error": None}
    rows = (
        (
            await db.execute(
                select(AgentPrediction).where(AgentPrediction.job_id == job_id)
            )
        )
        .scalars()
        .all()
    )
    errors = [
        abs(float(row.error_relative)) for row in rows if row.error_relative is not None
    ]
    return {
        "settled": len(errors),
        "mean_relative_error": (sum(errors) / len(errors)) if errors else None,
    }


async def record_outcomes_for_job(
    db: AsyncSession,
    job: Any,
    state: Mapping[str, Any],
    contract_eval: Optional[Mapping[str, Any]] = None,
) -> List[AgentMethodOutcome]:
    """Attach this run's outcome to every method it carried.

    Failures here must not fail the run: a job that did its work and could not
    be scored is still a job that did its work.
    """
    try:
        methods = _method_memories(state)
        if not methods:
            return []

        cited = {name.lower() for name in _cited_names(state)}
        contract = dict(contract_eval or {})
        missing = contract.get("missing")
        scores = await _prediction_scores(db, getattr(job, "id", None))

        written: List[AgentMethodOutcome] = []
        for method in methods:
            try:
                memory_id = UUID(method["id"])
            except (ValueError, TypeError):
                continue
            outcome = AgentMethodOutcome(
                method_memory_id=memory_id,
                method_name=(method["name"] or "unnamed")[:200],
                user_id=getattr(job, "user_id", None),
                job_id=getattr(job, "id", None),
                cited=method["name"].lower() in cited,
                contract_enabled=bool(contract.get("enabled")),
                contract_satisfied=bool(contract.get("satisfied")),
                unmet_requirements=(
                    ", ".join(str(x) for x in missing[:8])
                    if isinstance(missing, list)
                    else None
                ),
                predictions_settled=int(scores["settled"]),
                mean_relative_error=scores["mean_relative_error"],
                iterations=int(getattr(job, "iteration", 0) or 0),
            )
            db.add(outcome)
            written.append(outcome)
        await db.flush()
        return written
    except Exception as exc:  # pragma: no cover - scoring must not break a run
        logger.warning(f"could not record method outcomes: {exc}")
        return []


def summarize(outcomes: Iterable[AgentMethodOutcome]) -> Dict[str, Any]:
    """A method's standing: how often it preceded work that held up."""
    rows = list(outcomes)
    if not rows:
        return {"runs": 0}

    graded = [row for row in rows if row.contract_enabled]
    satisfied = [row for row in graded if row.contract_satisfied]
    errors = [
        float(row.mean_relative_error)
        for row in rows
        if row.mean_relative_error is not None
    ]
    summary: Dict[str, Any] = {
        "runs": len(rows),
        "cited_by": sum(1 for row in rows if row.cited),
        "graded_runs": len(graded),
        "contracts_satisfied": len(satisfied),
        "predictions_settled": sum(int(row.predictions_settled or 0) for row in rows),
    }
    # A rate over one or two runs is noise wearing a percentage sign.
    if len(graded) >= MIN_RUNS_FOR_A_RATE:
        summary["satisfied_rate"] = round(len(satisfied) / len(graded), 2)
    if errors:
        summary["mean_relative_error"] = round(sum(errors) / len(errors), 4)
    return summary


def describe(summary: Mapping[str, Any]) -> str:
    """One line for a reader, saying what the standing rests on."""
    runs = int(summary.get("runs", 0) or 0)
    if not runs:
        return "not yet used by any run"
    parts = [f"carried by {runs} run{'s' if runs != 1 else ''}"]
    if summary.get("cited_by"):
        parts.append(f"cited by {summary['cited_by']}")
    graded = int(summary.get("graded_runs", 0) or 0)
    if graded:
        parts.append(
            f"{summary.get('contracts_satisfied', 0)}/{graded} met their contract"
        )
    if summary.get("mean_relative_error") is not None:
        parts.append(
            f"mean prediction error {summary['mean_relative_error'] * 100:.0f}%"
        )
    return ", ".join(parts)


async def standing_for(
    db: AsyncSession, method_memory_ids: Iterable[Any]
) -> Dict[str, Dict[str, Any]]:
    """Standing for each method, keyed by memory id as a string."""
    wanted: List[UUID] = []
    for raw in method_memory_ids:
        try:
            wanted.append(UUID(str(raw)))
        except (ValueError, TypeError):
            continue
    if not wanted:
        return {}
    rows = (
        (
            await db.execute(
                select(AgentMethodOutcome).where(
                    AgentMethodOutcome.method_memory_id.in_(wanted)
                )
            )
        )
        .scalars()
        .all()
    )
    # Runs scored against retracted measurements leave the record entirely.
    # Counting them as failures would punish a method for evidence that was
    # withdrawn, which is a different claim from the method having failed.
    from app.services import agent_retraction_service

    user_ids = {row.user_id for row in rows if row.user_id is not None}
    for user_id in user_ids:
        mine = [r for r in rows if r.user_id == user_id]
        kept = await agent_retraction_service.live_outcomes(db, user_id, mine)
        dropped = {id(r) for r in mine} - {id(r) for r in kept}
        if dropped:
            rows = [r for r in rows if id(r) not in dropped]

    grouped: Dict[str, List[AgentMethodOutcome]] = {}
    for row in rows:
        grouped.setdefault(str(row.method_memory_id), []).append(row)
    return {key: summarize(value) for key, value in grouped.items()}


def rank(candidates: List[Any], standing: Mapping[str, Mapping[str, Any]]) -> List[Any]:
    """Order recalled methods by what became of the runs that carried them.

    Standing only ranks where standing means something. Below the threshold a
    method keeps whatever order relevance gave it: a method used once is not
    better or worse than an untried one, and sorting on a single run would
    dress noise as judgement.

    A method whose record is established and bad is moved to the back rather
    than dropped. Removing it would end its record there, and a method that
    preceded three failures may have been the wrong method or may have been
    handed three hard problems -- demotion says which is suspected without
    deciding it.
    """

    def key(candidate: Any) -> Tuple[int, float, int]:
        summary = standing.get(str(getattr(candidate, "id", "")), {})
        graded = int(summary.get("graded_runs", 0) or 0)
        if graded < MIN_RUNS_FOR_A_RATE:
            # Untried and barely-tried alike: leave them where they were.
            return (1, 0.0, 0)
        rate = float(summary.get("satisfied_rate", 0.0) or 0.0)
        # Established and good first, established and bad last, the rest in
        # between where relevance put them.
        band = 0 if rate > 0.5 else 2
        return (band, -rate, -graded)

    return sorted(candidates, key=key)


def caution(summary: Mapping[str, Any]) -> str:
    """A warning to travel with a method whose record is established and poor.

    Said as what happened rather than as a verdict: the runs that carried this
    did not meet their contracts, which is a reason to look at it and not proof
    that it is why they failed.
    """
    graded = int(summary.get("graded_runs", 0) or 0)
    if graded < MIN_RUNS_FOR_A_RATE:
        return ""
    satisfied = int(summary.get("contracts_satisfied", 0) or 0)
    if satisfied == 0:
        return (
            f"none of the {graded} runs carrying this met their contract -- "
            "worth checking before following it again"
        )
    if satisfied / graded <= 0.5:
        return f"only {satisfied} of {graded} runs carrying this met their contract"
    return ""
