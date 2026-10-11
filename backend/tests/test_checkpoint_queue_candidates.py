"""The operator queue built from candidates is the queue built from every job.

The queue used to load every job a person had run and project each into a
row or into nothing. It now asks the database only for jobs that could become
a row. Asking for too few would hide a row with nobody told -- which is how
blocked runs once waited eleven days -- so this builds the queue both ways
over every combination of the things the projector looks at and requires
the same rows.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.api.endpoints.agent_jobs import _build_checkpoint_queue_items
from app.models import agent_job as agent_job_model
from app.models.agent_job import AgentJob
from app.modules.autonomy.application import (
    checkpoint_queue_candidates,
    checkpoint_queue_jobs,
)
from app.services.agent_job_queue_helpers import extract_approval_checkpoint

STATUSES = [
    "pending",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
    " Paused ",
]
PHASES = [
    None,
    "acting",
    "blocked_needs_input",
    " blocked_needs_input ",
    "awaiting_approval",
]
SCHEDULES = [None, "once", "recurring", "continuous", " Recurring "]
RESULTS = [
    None,
    {},
    {"findings": []},
    {"approval_checkpoint": {"message": "apply the patch?"}},
    {"approval_checkpoint": {}},
    {"approval_checkpoint": "yes"},
    {"execution_strategy": {"approval_checkpoints": {"pending": {"step": 2}}}},
    {"execution_strategy": {"approval_checkpoints": {"pending": None}}},
    {"blocked": {"reason": "no new findings", "missing": ["x"], "resumable": True}},
]
LOGS = [
    None,
    [],
    [{"phase": "acting"}],
    [{"phase": "acting"}, {"phase": "completed_contract_unmet", "missing": ["x"]}],
    ["not a dict", {"phase": "completed_contract_unmet"}],
]
QUIET_FOR = [timedelta(minutes=2), timedelta(hours=3)]


def _every_job(user_id):
    now = datetime.utcnow()
    combinations = itertools.product(
        STATUSES, PHASES, SCHEDULES, RESULTS, LOGS, QUIET_FOR
    )
    for n, (status, phase, schedule, results, log, quiet) in enumerate(combinations):
        yield AgentJob(
            id=uuid4(),
            name=f"job {n}",
            goal="a goal",
            job_type="research",
            user_id=user_id,
            status=status,
            current_phase=phase,
            schedule_type=schedule,
            results=results,
            execution_log=log,
            iteration=3,
            max_iterations=100,
            tool_calls_used=1,
            max_tool_calls=100,
            llm_calls_used=1,
            max_llm_calls=100,
            max_runtime_minutes=60,
            created_at=now - timedelta(days=1, seconds=n),
            last_activity_at=now - quiet,
        )


def _rows(jobs):
    return sorted(
        (str(row.job_id), row.item_type)
        for row in _build_checkpoint_queue_items(list(jobs), [], monitor_health_rows=[])
    )


@pytest.mark.asyncio
async def test_the_same_rows_come_from_the_candidates_as_from_every_job(
    db_session, test_user
):
    db_session.add_all(_every_job(test_user.id))
    await db_session.commit()

    everything = (
        (
            await db_session.execute(
                select(AgentJob).where(AgentJob.user_id == test_user.id)
            )
        )
        .scalars()
        .all()
    )
    candidates = await checkpoint_queue_candidates.load_queue_jobs(
        db_session, test_user.id
    )

    from_everything, from_candidates = _rows(everything), _rows(candidates)
    kinds = {kind for _, kind in from_everything}

    # Every kind of job row is really produced by this matrix...
    assert {"approval_checkpoint", "blocked_run", "contract_unmet"} <= kinds
    assert len(kinds) >= 4
    # ...none is lost by asking for candidates...
    assert from_candidates == from_everything
    # ...and asking for candidates is worth doing.
    assert len(candidates) < len(everything) * 0.8
    assert len(everything) == len(STATUSES) * len(PHASES) * len(SCHEDULES) * len(
        RESULTS
    ) * len(LOGS) * len(QUIET_FOR)


@pytest.mark.asyncio
async def test_a_job_named_by_an_inbox_item_is_loaded_whatever_its_state(
    db_session, test_user
):
    plain = AgentJob(
        id=uuid4(),
        name="monitor",
        goal="watch",
        job_type="research",
        user_id=test_user.id,
        status="completed",
    )
    db_session.add(plain)
    await db_session.commit()

    without = await checkpoint_queue_candidates.load_queue_jobs(
        db_session, test_user.id
    )
    named = await checkpoint_queue_candidates.load_queue_jobs(
        db_session, test_user.id, also=[plain.id, None]
    )

    assert without == []
    assert [job.id for job in named] == [plain.id]


@pytest.mark.asyncio
async def test_somebody_elses_jobs_are_not_candidates(db_session, test_user):
    db_session.add(
        AgentJob(
            id=uuid4(),
            name="theirs",
            goal="g",
            job_type="research",
            user_id=uuid4(),
            status="paused",
            current_phase="blocked_needs_input",
        )
    )
    await db_session.commit()

    assert (
        await checkpoint_queue_candidates.load_queue_jobs(db_session, test_user.id)
        == []
    )


# --- the column that makes the JSON reasons askable -------------------------


@pytest.mark.parametrize("results", RESULTS)
def test_the_model_reads_an_approval_checkpoint_as_the_queue_does(results):
    # Restated in the model because a model cannot import a service.
    job = AgentJob(status="paused", results=results)

    assert agent_job_model.approval_checkpoint_in(results) is bool(
        extract_approval_checkpoint(job)
    )


def test_the_model_and_the_projector_name_the_same_log_entry():
    assert (
        agent_job_model.CONTRACT_UNMET_LOG_PHASE
        == checkpoint_queue_jobs.CONTRACT_UNMET_PHASE
    )


def _flags(job):
    return (job.has_approval_checkpoint, job.has_contract_unmet_entry)


@pytest.mark.asyncio
async def test_the_flags_follow_the_job(db_session, test_user):
    job = AgentJob(
        id=uuid4(), name="j", goal="g", job_type="research", user_id=test_user.id
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    assert _flags(job) == (False, False)

    job.results = {"approval_checkpoint": {"message": "ok?"}}
    await db_session.commit()
    await db_session.refresh(job)
    assert _flags(job) == (True, False)

    # Appended in place, as the finaliser does.
    job.add_log_entry({"phase": "completed_contract_unmet", "missing": ["x"]})
    await db_session.commit()
    await db_session.refresh(job)
    assert _flags(job) == (True, True)

    job.results = {"findings": []}
    await db_session.commit()
    await db_session.refresh(job)
    assert _flags(job) == (False, True)


@pytest.mark.asyncio
async def test_a_job_edited_straight_after_it_is_created(db_session, test_user):
    # Nothing was loaded back after the insert: the flags must not depend
    # on attributes the session never read.
    job = AgentJob(
        id=uuid4(), name="j", goal="g", job_type="research", user_id=test_user.id
    )
    db_session.add(job)
    await db_session.commit()

    job.results = {"approval_checkpoint": {"message": "ok?"}}
    await db_session.commit()
    db_session.expunge_all()

    assert _flags(await db_session.get(AgentJob, job.id)) == (True, False)


@pytest.mark.asyncio
async def test_a_row_loaded_without_its_log_keeps_what_the_log_said(
    db_session, test_user
):
    # The job list defers the log. Editing results on a row loaded that way
    # must not erase the flag only the log can justify.
    from sqlalchemy.orm import defer

    job = AgentJob(
        id=uuid4(),
        name="j",
        goal="g",
        job_type="research",
        user_id=test_user.id,
        execution_log=[{"phase": "completed_contract_unmet"}],
    )
    db_session.add(job)
    await db_session.commit()
    job_id = job.id
    db_session.expunge_all()

    partial = (
        await db_session.execute(
            select(AgentJob)
            .options(defer(AgentJob.execution_log))
            .where(AgentJob.id == job_id)
        )
    ).scalar_one()
    partial.results = {"approval_checkpoint": {"message": "ok?"}}
    await db_session.commit()
    db_session.expunge_all()

    assert _flags(await db_session.get(AgentJob, job_id)) == (True, True)
