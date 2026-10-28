"""A pipeline stage that ends short of its contract gets one more attempt.

A stage that fails stops the plan: the stages behind it never start, and the
run waits for a person to read the logs and restart it with a note. The note
a person writes there is almost always assembled from things the run already
knows -- what the contract still lacks, which tool produces it, why the
attempt stopped -- so the first retry does not need a person. This writes
that note from the run's own record and restarts the stage once, through
`agent_pipeline_restart.restart_from_stage`, the same path a person's restart
takes: the stage is re-fired from its completed predecessor and inherits the
same evidence it had the first time.

What it does not do, and why:

* Retry a stage whose blocker is the platform. A tool that never ran, or
  evidence no tool this job type may call produces, is not something a
  second attempt can change; that blocker is filed as coding work instead.
* Retry a stage held for a person's approval, cancelled, or failed outright.
  Only the two contract-unmet endings qualify: completed with the contract
  unmet (out of iterations) and paused as blocked on input.
* Retry a retry. The limit is per stage per run (`AGENT_STAGE_AUTO_RETRIES`,
  or a stage's own `auto_retries`), counted from the jobs the run holds, so it
  survives the attempts that spent it.

The retry spends the stage's budget again. That is the point -- the attempt
that ran out of iterations gets a fresh allowance -- and it is why the limit
defaults to one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.services.config_values import safe_int

#: The log phase an attempt that was retried ends on. The operator queue reads
#: it: a stage being retried is not waiting on anybody.
RETRIED_PHASE = "stage_retried_automatically"

#: Marks the job a retry created, with the attempt it retries.
RETRY_OF_KEY = "auto_retry_of"

MAX_NOTE_CHARS = 2000
MAX_LISTED = 5


def retry_limit(config: Optional[Mapping[str, Any]]) -> int:
    """How many automatic attempts a stage may have after its first."""
    from app.core.config import settings

    config = config if isinstance(config, Mapping) else {}
    default = safe_int(getattr(settings, "AGENT_STAGE_AUTO_RETRIES", 1), 1)
    value = config.get("auto_retries")
    return max(0, safe_int(value, default) if value is not None else default)


def qualifies(
    job: Any, *, contract_unmet: bool, blocked: Optional[Mapping[str, Any]]
) -> str:
    """ "" if this attempt may be retried, else why not (for the log)."""
    from app.services import agent_pipeline_restart

    if not contract_unmet:
        return "contract met"
    if not agent_pipeline_restart.stage_of(job):
        return "not a pipeline stage"
    status = str(getattr(job, "status", "") or "")
    phase = str(getattr(job, "current_phase", "") or "")
    blocked_on_input = status == "paused" and phase == "blocked_needs_input"
    if not (status == "completed" or blocked_on_input):
        # Held for approval, cancelled or failed: someone else's move.
        return f"ended {status}/{phase}"
    need = (blocked or {}).get("needs") if isinstance(blocked, Mapping) else None
    if isinstance(need, Mapping) and need.get("answerable_by") == "platform_change":
        return "blocked by the platform, which a second attempt cannot change"
    return ""


def attempts_used(stage_jobs: Sequence[Any], stage_id: str) -> int:
    """Automatic retries this run has already given the stage."""
    from app.services import agent_pipeline_restart

    used = 0
    for item in stage_jobs:
        job = getattr(item, "job", item)
        config = getattr(job, "config", None)
        if (
            isinstance(config, dict)
            and config.get(RETRY_OF_KEY)
            and agent_pipeline_restart.stage_of(job) == stage_id
        ):
            used += 1
    return used


def _titles(findings: Any, limit: int = MAX_LISTED) -> List[str]:
    out: List[str] = []
    for f in findings if isinstance(findings, list) else []:
        if not isinstance(f, dict):
            continue
        title = str(f.get("title") or f.get("summary") or "").strip()
        if title:
            out.append(f"[{f.get('type') or 'finding'}] {title[:160]}")
        if len(out) >= limit:
            break
    return out


def build_note(
    *,
    job: Any,
    state: Mapping[str, Any],
    contract_eval: Mapping[str, Any],
    blocked: Optional[Mapping[str, Any]],
) -> str:
    """The note a person would have written, from what the run recorded."""
    from app.services import agent_evidence_map, agent_measurement_validity

    missing = [str(m) for m in (contract_eval.get("missing") or []) if m]
    lines = [
        "AUTOMATIC RETRY. The previous attempt at this stage ended without "
        "meeting its contract. Start by producing what is missing; do not "
        "repeat orientation the last attempt already did."
    ]
    if missing:
        lines.append("Still missing: " + ", ".join(missing[:8]) + ".")

    metrics = contract_eval.get("metrics")
    validity = metrics.get("validity") if isinstance(metrics, Mapping) else None
    details = validity.get("details") if isinstance(validity, Mapping) else None
    remedies = agent_measurement_validity.explain(
        [m for m in missing if m.startswith("validity:")],
        details if isinstance(details, Mapping) else {},
    )
    lines.extend(remedies[:MAX_LISTED])

    required = [
        m.split(":", 1)[1].split(">=", 1)[0]
        for m in missing
        if not m.startswith(("validity:", "result_key:")) and ":" in m
    ]
    if required:
        how = agent_evidence_map.describe_chain(
            required, job_type=str(getattr(job, "job_type", "") or "research")
        )
        if how:
            lines.append("How to obtain it: " + " ".join(how[:MAX_LISTED]))

    if isinstance(blocked, Mapping):
        reason = str(blocked.get("reason") or "").strip()
        if reason:
            lines.append(f"The last attempt stopped because: {reason[:400]}")
        need = blocked.get("needs")
        if isinstance(need, Mapping) and need.get("question"):
            lines.append(f"Its open question: {str(need['question'])[:400]}")

    found = _titles(state.get("findings"))
    if found:
        lines.append(
            "It had established (not carried over; reproduce only what the "
            "contract needs): " + "; ".join(found) + "."
        )
    return "\n".join(lines)[:MAX_NOTE_CHARS]


async def retry(
    executor: Any,
    job: Any,
    *,
    state: Dict[str, Any],
    contract_eval: Mapping[str, Any],
    blocked: Optional[Mapping[str, Any]],
    db: Any,
) -> Optional[Any]:
    """Restart this stage once if it qualifies. Returns the new job, or None.

    Never raises: a stage that cannot be retried still has to finish the way
    it would have without this.
    """
    from loguru import logger

    from app.services import agent_pipeline_restart

    why_not = qualifies(
        job,
        contract_unmet=bool(contract_eval.get("enabled"))
        and not bool(contract_eval.get("satisfied")),
        blocked=blocked,
    )
    if why_not:
        return None

    stage_id = agent_pipeline_restart.stage_of(job)
    root_id = job.root_job_id or job.parent_job_id or job.id
    try:
        stages = await agent_pipeline_restart.load_run(root_id, db)
        limit = retry_limit(job.config)
        if attempts_used(stages, stage_id) >= limit:
            if limit:
                job.add_log_entry(
                    {
                        "phase": "stage_auto_retry_spent",
                        "stage": stage_id,
                        "reason": f"{limit} automatic retr"
                        f"{'y' if limit == 1 else 'ies'} already used",
                    }
                )
            return None

        note = build_note(
            job=job, state=state, contract_eval=contract_eval, blocked=blocked
        )
        child = await agent_pipeline_restart.restart_from_stage(
            root_job_id=root_id,
            stage_id=stage_id,
            executor=executor,
            db=db,
            note=note,
            extra_config={RETRY_OF_KEY: str(job.id)},
        )
    except agent_pipeline_restart.PipelineRestartError as error:
        job.add_log_entry(
            {
                "phase": "stage_auto_retry_refused",
                "stage": stage_id,
                "reason": error.detail[:400],
            }
        )
        return None
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning(f"Automatic retry of stage {stage_id} ({job.id}) failed: {exc}")
        return None

    return child


__all__ = [
    "RETRIED_PHASE",
    "RETRY_OF_KEY",
    "attempts_used",
    "build_note",
    "qualifies",
    "retry",
    "retry_limit",
]
