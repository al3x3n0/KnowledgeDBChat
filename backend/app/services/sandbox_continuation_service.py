"""Short polling and dispatch for agent actions waiting on independent Jobs."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import select

from app.core.config import settings
from app.models.agent_job import AgentJob
from app.services import (
    agent_job_user_cap,
    broker_queue,
    job_dispatch,
    kubernetes_sandbox,
)
from app.services import kubernetes_sandbox_artifacts as artifacts
from app.services import sandbox_continuation_state as state


def reference_record(reference: dict, config):
    name = str(reference.get("name") or "")
    if (
        not re.fullmatch(r"kdbc-step-[a-f0-9]{40}", name)
        or reference.get("namespace") != config.namespace
    ):
        raise ValueError("Invalid sandbox continuation reference")
    directory = config.root / name
    record = json.loads((directory / "record.json").read_text())
    return name, directory, record


async def ready(reference, *, api, config):
    name, directory, record = await asyncio.to_thread(
        reference_record, reference, config
    )
    if record.get("stop_reason") == "timed_out" and record["status"] != "timed_out":
        # A timeout request is durable before DELETE. Do not mistake a later
        # Job disappearance for a lost experiment, or collect a late success.
        return False
    if record["status"] in ("completed", "failed", "timed_out", "cancelled"):
        return True
    response = await api.request(
        "GET", f"/apis/batch/v1/namespaces/{config.namespace}/jobs/{name}"
    )
    if response.status_code == 404:
        # A lost create response is safe to retry by the same name. A Job
        # lost after adoption is reported by the runner without resubmission.
        return True
    response.raise_for_status()
    job = response.json()
    if (
        job["metadata"].get("annotations", {}).get("kdbc.io/request-sha256")
        != record["fingerprint"]
    ):
        raise ValueError("Sandbox continuation Job has a different request fingerprint")
    if record.get("uid") and job["metadata"]["uid"] != record["uid"]:
        raise ValueError("Sandbox continuation Job was replaced")
    if record["status"] == "prepared":
        record.update(status="running", uid=job["metadata"]["uid"])
        await asyncio.to_thread(artifacts.save, directory / "record.json", record)
    return any(
        c.get("status") == "True" and c.get("type") in ("Complete", "Failed")
        for c in job.get("status", {}).get("conditions", [])
    )


def recently_dispatched(wait: dict, now: datetime) -> bool:
    try:
        moment = datetime.fromisoformat(wait.get("dispatched_at") or "")
    except (TypeError, ValueError):
        return False
    if moment.tzinfo:
        moment = moment.replace(tzinfo=None)
    return now - moment < timedelta(minutes=2)


async def poll_waiting(db, *, api=None, config=None, limit=None, now=None) -> dict:
    """Commit before publishing, and retry lost dispatches after a grace window."""
    config = config or kubernetes_sandbox.Configuration.configured()
    config.validate()
    limit = max(1, min(int(limit or settings.KUBERNETES_SANDBOX_POLL_BATCH_SIZE), 100))
    now = now or datetime.utcnow()
    counts = {"checked": 0, "queued": 0, "cancelled": 0, "errors": 0}
    query = (
        select(AgentJob.id, AgentJob.status, AgentJob.results[state.RESULTS_KEY])
        .where(
            AgentJob.status.in_(
                ("paused", "pending", "cancelled", "failed", "completed")
            ),
            AgentJob.results[state.RESULTS_KEY]["key"].as_string() != "",
        )
        .order_by(
            AgentJob.results[state.RESULTS_KEY]["last_polled_at"]
            .as_string()
            .asc()
            .nullsfirst(),
            AgentJob.last_activity_at.asc().nullsfirst(),
        )
        .limit(limit)
    )
    candidates = (await db.execute(query)).all()
    ids = [row[0] for row in candidates]
    # Read-only discovery must not hold a snapshot/row lock across the batch.
    await db.commit()
    if not ids:
        return counts
    queued_ids = {
        str(job_id)
        for job_id, status, wait in candidates
        if status == "pending"
        and isinstance(wait, dict)
        and wait.get("auto_resume")
        and not recently_dispatched(wait, now)
    }
    # No row locks during broker I/O. Recheck the live row before using this
    # observation; a queued/reserved message already delivers this job's ID.
    presence = (
        await asyncio.to_thread(broker_queue.agent_job_message_presence, queued_ids)
        if queued_ids
        else {}
    )
    owned_api = api is None
    api = api or kubernetes_sandbox.KubernetesAPI()
    started = time.monotonic()
    try:
        for job_id in ids:
            if time.monotonic() - started > 20:
                break
            job = (
                await db.execute(
                    select(AgentJob)
                    .where(AgentJob.id == job_id)
                    .with_for_update(skip_locked=True)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one_or_none()
            wait = state.waiting(job) if job is not None else None
            if not wait or job.status not in (
                "paused",
                "pending",
                "cancelled",
                "failed",
                "completed",
            ):
                await db.commit()
                continue
            live = job.execution_lease_expires_at
            if live and live.replace(tzinfo=None) > now:
                # Rotate candidates owned by a worker too. They can remain
                # pending during startup and otherwise occupy every batch.
                state.write_wait(job, dict(wait, last_polled_at=now.isoformat()))
                await db.commit()
                continue
            terminal_owner = job.status in ("cancelled", "failed", "completed")
            if not terminal_owner and (
                not wait.get("auto_resume")
                or recently_dispatched(wait, now)
                or (job.status == "pending" and presence.get(str(job.id)) is True)
            ):
                state.write_wait(job, dict(wait, last_polled_at=now.isoformat()))
                await db.commit()
                continue
            counts["checked"] += 1
            try:
                name, directory, record = await asyncio.to_thread(
                    reference_record, wait["reference"], config
                )
                if terminal_owner:
                    removed = await kubernetes_sandbox.cancel(
                        api,
                        config,
                        name,
                        fingerprint=record["fingerprint"],
                        uid=record.get("uid", ""),
                    )
                    record["status"] = "cancelled"
                    await asyncio.to_thread(
                        artifacts.save, directory / "record.json", record
                    )
                    if removed:
                        state.write_wait(job, None)
                        counts["cancelled"] += 1
                    else:
                        state.write_wait(
                            job, dict(wait, last_polled_at=now.isoformat())
                        )
                else:
                    complete = await ready(wait["reference"], api=api, config=config)
                    if record.get("stop_reason") == "timed_out" or (
                        not complete and time.time() >= float(wait["deadline_at"])
                    ):
                        record["stop_reason"] = "timed_out"
                        await asyncio.to_thread(
                            artifacts.save, directory / "record.json", record
                        )
                        complete = await kubernetes_sandbox.cancel(
                            api,
                            config,
                            name,
                            fingerprint=record["fingerprint"],
                            uid=record.get("uid", ""),
                        )
                        if complete:
                            record["status"] = "timed_out"
                            await asyncio.to_thread(
                                artifacts.save, directory / "record.json", record
                            )
                    wait = dict(wait, last_polled_at=now.isoformat())
                    if complete:
                        job.status = "pending"
                        job.celery_task_id = None
                        wait["dispatched_at"] = now.isoformat()
                        state.write_wait(job, wait)
                        agent_job_user_cap.clear_wait(job)
                        job_dispatch.enqueue_agent_job(db, job.id, job.user_id)
                        counts["queued"] += 1
                    else:
                        state.write_wait(job, wait)
                job.last_activity_at = now
                await db.commit()
            except Exception as exc:
                counts["errors"] += 1
                await db.rollback()
                logger.warning(
                    f"Could not poll sandbox continuation for {job_id}: {exc}"
                )
                current_job = await db.get(AgentJob, job_id)
                current_wait = state.waiting(current_job) if current_job else None
                if (
                    current_wait
                    and current_wait["key"] == wait["key"]
                    and current_job.status
                    in ("paused", "pending", "cancelled", "failed", "completed")
                ):
                    # Rotate transient errors, so one failing batch cannot
                    # starve healthy experiments waiting behind it.
                    current_job.last_activity_at = now
                    state.write_wait(
                        current_job,
                        dict(
                            current_wait,
                            last_error=str(exc)[:300],
                            last_polled_at=now.isoformat(),
                            auto_resume=False
                            if isinstance(exc, ValueError)
                            else current_wait.get("auto_resume"),
                        ),
                    )
                    if isinstance(exc, ValueError) and current_job.status == "paused":
                        current_job.current_phase = "blocked_needs_input"
                        current_job.phase_details = (
                            f"Cannot reconcile experiment: {exc}"[:280]
                        )
                    await db.commit()
    finally:
        if owned_api:
            await api.close()
    return counts
