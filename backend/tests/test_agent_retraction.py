"""A result that was believed, then checked, has to take down what rested on it.

This project retracted an entire per-instruction measurement table -- four of
nine classes had been timed on chains that reached infinity within a few
iterations -- and nothing in the system noticed. Methods validated against those
numbers kept their standing. An autonomous programme accumulates poison faster
than it accumulates results unless retraction is a supported operation.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.agent_method_outcome import AgentMethodOutcome
from app.models.agent_retraction import RetractionKind
from app.services import agent_method_record
from app.services import agent_method_standing_service as standing
from app.services import agent_retraction_service as retraction


def _outcome(user_id, *, job_id=None, memory_id=None, satisfied=True):
    return AgentMethodOutcome(
        method_memory_id=memory_id or uuid4(),
        method_name="value-stable chains",
        user_id=user_id,
        job_id=job_id or uuid4(),
        cited=False,
        contract_enabled=True,
        contract_satisfied=satisfied,
        predictions_settled=1,
        mean_relative_error=0.05,
        iterations=8,
    )


@pytest.mark.asyncio
async def test_a_retraction_needs_a_reason(db_session):
    """A later run has to tell a result withdrawn for a harness defect from one
    withdrawn because the question changed."""
    with pytest.raises(ValueError) as excinfo:
        await retraction.retract(
            db_session,
            user_id=uuid4(),
            kind=RetractionKind.FINDING_TYPE,
            ref="recip_throughput",
            reason="   ",
        )

    assert "reason" in str(excinfo.value)


@pytest.mark.asyncio
async def test_an_unknown_kind_is_refused(db_session):
    with pytest.raises(ValueError) as excinfo:
        await retraction.retract(
            db_session, user_id=uuid4(), kind="vibes", ref="x", reason="because"
        )

    assert "unknown retraction kind" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_retracted_run_leaves_a_methods_record(db_session):
    """The propagation that matters: a method validated against defective
    numbers must stop being able to recommend itself on them."""
    user_id, memory_id = uuid4(), uuid4()
    bad_job = uuid4()
    for job_id in (bad_job, uuid4(), uuid4()):
        db_session.add(_outcome(user_id, job_id=job_id, memory_id=memory_id))
    await db_session.flush()

    before = await standing.standing_for(db_session, [str(memory_id)])
    assert before[str(memory_id)]["runs"] == 3

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.JOB,
        ref=str(bad_job),
        reason="the host was never verified; its controls read 0.53",
    )

    after = await standing.standing_for(db_session, [str(memory_id)])
    assert after[str(memory_id)]["runs"] == 2


@pytest.mark.asyncio
async def test_a_retracted_run_is_removed_not_counted_as_a_failure(db_session):
    """Counting it as a failure would punish the method for evidence that was
    withdrawn, which is a different claim from the method having failed."""
    user_id, memory_id = uuid4(), uuid4()
    good = uuid4()
    bad = uuid4()
    db_session.add(_outcome(user_id, job_id=good, memory_id=memory_id, satisfied=True))
    db_session.add(_outcome(user_id, job_id=bad, memory_id=memory_id, satisfied=True))
    await db_session.flush()

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.JOB,
        ref=str(bad),
        reason="measured on chains that reached infinity",
    )

    after = await standing.standing_for(db_session, [str(memory_id)])
    summary = after[str(memory_id)]
    assert summary["runs"] == 1
    assert summary["contracts_satisfied"] == 1


@pytest.mark.asyncio
async def test_retracting_a_method_removes_its_own_record(db_session):
    user_id, memory_id = uuid4(), uuid4()
    for _ in range(3):
        db_session.add(_outcome(user_id, memory_id=memory_id))
    await db_session.flush()

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.METHOD,
        ref=str(memory_id),
        reason="the procedure double-counts the prologue",
    )

    after = await standing.standing_for(db_session, [str(memory_id)])
    assert after.get(str(memory_id), {"runs": 0})["runs"] == 0


@pytest.mark.asyncio
async def test_another_users_retraction_does_not_touch_mine(db_session):
    mine, theirs = uuid4(), uuid4()
    memory_id, job_id = uuid4(), uuid4()
    db_session.add(_outcome(mine, job_id=job_id, memory_id=memory_id))
    await db_session.flush()

    await retraction.retract(
        db_session,
        user_id=theirs,
        kind=RetractionKind.JOB,
        ref=str(job_id),
        reason="not my run",
    )

    after = await standing.standing_for(db_session, [str(memory_id)])
    assert after[str(memory_id)]["runs"] == 1


# --- method evidence ------------------------------------------------------


def _record(cited):
    return agent_method_record.build_record(
        name="measure with value-stable chains",
        procedure=["keep the dependence, not the value"],
        prevents="timing infinity instead of the instruction named",
        derived_from=cited,
        available_finding_types=list(cited),
    )


@pytest.mark.asyncio
async def test_a_method_whose_evidence_is_all_retracted_is_unvalidated(db_session):
    user_id = uuid4()
    record = _record(["benchmark_measurement"])
    assert record["status"] == agent_method_record.VALIDATED

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.FINDING_TYPE,
        ref="benchmark_measurement",
        reason="taken on a host whose controls never passed",
    )

    status = await retraction.method_evidence_status(db_session, user_id, record)

    assert status["changed"] is True
    assert status["status"] == agent_method_record.UNVALIDATED
    assert status["retracted_evidence"] == ["benchmark_measurement"]


@pytest.mark.asyncio
async def test_one_surviving_citation_keeps_a_method_validated(db_session):
    """Weakened is not unvalidated: one piece of live evidence is still
    evidence."""
    user_id = uuid4()
    record = _record(["benchmark_measurement", "dynamic_profile"])

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.FINDING_TYPE,
        ref="benchmark_measurement",
        reason="defective harness",
    )

    status = await retraction.method_evidence_status(db_session, user_id, record)

    assert status["changed"] is False
    assert status["retracted_evidence"] == ["benchmark_measurement"]
    assert status["surviving_evidence"] == ["dynamic_profile"]


@pytest.mark.asyncio
async def test_a_method_with_no_evidence_is_unaffected(db_session):
    user_id = uuid4()
    record = _record([agent_method_record.NO_EVIDENCE])

    status = await retraction.method_evidence_status(db_session, user_id, record)

    assert status["changed"] is False


# --- withdrawing and reporting -------------------------------------------


@pytest.mark.asyncio
async def test_a_retraction_can_itself_be_withdrawn(db_session):
    """The measurement was re-taken and held after all. Possible only because
    propagation is computed on read."""
    user_id, memory_id, job_id = uuid4(), uuid4(), uuid4()
    db_session.add(_outcome(user_id, job_id=job_id, memory_id=memory_id))
    await db_session.flush()

    row = await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.JOB,
        ref=str(job_id),
        reason="suspected bad host",
    )
    assert (await standing.standing_for(db_session, [str(memory_id)])).get(
        str(memory_id), {"runs": 0}
    )["runs"] == 0

    assert await retraction.withdraw(db_session, row.id) is True

    after = await standing.standing_for(db_session, [str(memory_id)])
    assert after[str(memory_id)]["runs"] == 1


@pytest.mark.asyncio
async def test_nothing_is_deleted_by_a_retraction(db_session):
    """The evidence that something was once believed is itself evidence."""
    user_id, job_id, memory_id = uuid4(), uuid4(), uuid4()
    db_session.add(_outcome(user_id, job_id=job_id, memory_id=memory_id))
    await db_session.flush()

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.JOB,
        ref=str(job_id),
        reason="bad host",
    )

    summary = await retraction.affected(db_session, user_id)
    assert summary["outcomes_total"] == 1, "the row is still there"
    assert summary["outcomes_dropped"] == 1, "it just does not count"


@pytest.mark.asyncio
async def test_the_report_says_what_the_retraction_took_down(db_session):
    user_id, memory_id = uuid4(), uuid4()
    bad = uuid4()
    db_session.add(_outcome(user_id, job_id=bad, memory_id=memory_id))
    db_session.add(_outcome(user_id, memory_id=memory_id))
    await db_session.flush()

    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.JOB,
        ref=str(bad),
        reason="controls never passed",
    )
    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.FINDING_TYPE,
        ref="recip_throughput",
        reason="12-55% run to run; 1 of 9 within band",
    )

    summary = await retraction.affected(db_session, user_id)

    assert summary["retracted_jobs"] == 1
    assert summary["retracted_finding_types"] == ["recip_throughput"]
    assert summary["outcomes_dropped"] == 1
    assert summary["methods_losing_runs"]["value-stable chains"] == 1


@pytest.mark.asyncio
async def test_a_run_is_told_what_it_must_not_cite(db_session):
    """A run has no way to tell that a finding type it is about to rely on was
    withdrawn last week."""
    user_id = uuid4()
    await retraction.retract(
        db_session,
        user_id=user_id,
        kind=RetractionKind.FINDING_TYPE,
        ref="recip_throughput",
        reason="does not reproduce: 12-55% across independent runs",
    )

    note = await retraction.note_for_prompt(db_session, user_id)

    assert "recip_throughput" in note
    assert "do not cite" in note.lower()
    assert "does not reproduce" in note


@pytest.mark.asyncio
async def test_no_retractions_means_no_note(db_session):
    assert await retraction.note_for_prompt(db_session, uuid4()) == ""
