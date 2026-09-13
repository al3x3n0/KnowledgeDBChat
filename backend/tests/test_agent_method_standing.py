"""A method should earn its standing from what became of the runs that used it.

Methods were recorded, recalled and never scored, so one that misleads carried
exactly the authority of one that works. This is the calibration store's shape
applied to procedure: association, accumulated, and honest about being an
association.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.agent_method_outcome import AgentMethodOutcome
from app.services import agent_method_record
from app.services import agent_method_standing_service as standing


def _outcome(**overrides):
    row = AgentMethodOutcome(
        method_memory_id=uuid4(),
        method_name="inline-asm chains",
        cited=False,
        contract_enabled=True,
        contract_satisfied=True,
        predictions_settled=1,
        mean_relative_error=0.1,
        iterations=8,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def test_a_method_nothing_has_used_has_no_standing():
    assert standing.summarize([])["runs"] == 0
    assert standing.describe({"runs": 0}) == "not yet used by any run"


def test_standing_counts_runs_and_contracts():
    rows = [
        _outcome(contract_satisfied=True),
        _outcome(contract_satisfied=False),
        _outcome(contract_satisfied=True),
    ]

    summary = standing.summarize(rows)

    assert summary["runs"] == 3
    assert summary["graded_runs"] == 3
    assert summary["contracts_satisfied"] == 2
    assert summary["satisfied_rate"] == 0.67


def test_a_rate_is_withheld_until_it_means_something():
    """One or two runs is noise wearing a percentage sign."""
    summary = standing.summarize([_outcome(), _outcome()])

    assert summary["runs"] == 2
    assert "satisfied_rate" not in summary


def test_ungraded_runs_do_not_count_toward_a_rate():
    """A run with no contract says nothing about whether the method worked."""
    rows = [
        _outcome(contract_enabled=False, contract_satisfied=False),
        _outcome(contract_enabled=False, contract_satisfied=False),
        _outcome(contract_enabled=True, contract_satisfied=True),
    ]

    summary = standing.summarize(rows)

    assert summary["runs"] == 3
    assert summary["graded_runs"] == 1
    assert "satisfied_rate" not in summary


def test_being_cited_is_counted_apart_from_being_present():
    """Carried is not followed, and merging the two overstates the evidence."""
    rows = [_outcome(cited=True), _outcome(cited=False), _outcome(cited=False)]

    summary = standing.summarize(rows)

    assert summary["runs"] == 3
    assert summary["cited_by"] == 1


def test_prediction_error_is_averaged_across_runs():
    rows = [_outcome(mean_relative_error=0.2), _outcome(mean_relative_error=0.4)]

    assert standing.summarize(rows)["mean_relative_error"] == 0.3


def test_a_run_that_settled_nothing_scores_neither_way():
    rows = [_outcome(mean_relative_error=None, predictions_settled=0)]

    summary = standing.summarize(rows)

    assert "mean_relative_error" not in summary
    assert summary["predictions_settled"] == 0


def test_the_description_says_what_the_standing_rests_on():
    summary = standing.summarize(
        [_outcome(cited=True), _outcome(contract_satisfied=False), _outcome()]
    )

    line = standing.describe(summary)

    assert "carried by 3 runs" in line
    assert "cited by 1" in line
    assert "2/3 met their contract" in line


def _state_with_method(name: str, memory_id: str, builds_on=None):
    record = agent_method_record.build_record(
        name=name,
        procedure=["do the thing"],
        prevents="the wrong answer",
        derived_from=["none"],
        available_finding_types=[],
    )
    actions = []
    if builds_on is not None:
        actions.append(
            {
                "action": {"tool": "record_method", "params": {"builds_on": builds_on}},
                "result": {"success": True},
            }
        )
    return {
        "injected_memory_payloads": [
            {
                "id": memory_id,
                "type": "pattern",
                "content": agent_method_record.render(record),
            }
        ],
        "actions_taken": actions,
    }


class _Job:
    def __init__(self):
        self.id = uuid4()
        self.user_id = uuid4()
        self.iteration = 7


@pytest.mark.asyncio
async def test_a_finished_run_scores_every_method_it_carried(db_session):
    memory_id = str(uuid4())
    state = _state_with_method("inline-asm chains", memory_id)

    written = await standing.record_outcomes_for_job(
        db_session,
        _Job(),
        state,
        {"enabled": True, "satisfied": True, "missing": []},
    )

    assert len(written) == 1
    assert written[0].method_name == "inline-asm chains"
    assert written[0].contract_satisfied is True
    assert written[0].cited is False


@pytest.mark.asyncio
async def test_naming_the_method_records_the_stronger_claim(db_session):
    memory_id = str(uuid4())
    state = _state_with_method(
        "inline-asm chains", memory_id, builds_on=["inline-asm chains"]
    )

    written = await standing.record_outcomes_for_job(
        db_session, _Job(), state, {"enabled": True, "satisfied": True}
    )

    assert written[0].cited is True


@pytest.mark.asyncio
async def test_an_unmet_contract_is_recorded_with_what_was_missing(db_session):
    state = _state_with_method("a method", str(uuid4()))

    written = await standing.record_outcomes_for_job(
        db_session,
        _Job(),
        state,
        {"enabled": True, "satisfied": False, "missing": ["validity:records_method"]},
    )

    assert written[0].contract_satisfied is False
    assert "validity:records_method" in written[0].unmet_requirements


@pytest.mark.asyncio
async def test_memories_that_are_not_methods_are_left_alone(db_session):
    state = {
        "injected_memory_payloads": [
            {"id": str(uuid4()), "type": "insight", "content": "reranking helped"}
        ],
        "actions_taken": [],
    }

    assert await standing.record_outcomes_for_job(db_session, _Job(), state, {}) == []


@pytest.mark.asyncio
async def test_scoring_never_fails_the_run(db_session):
    """A job that did its work and could not be graded still did its work."""
    written = await standing.record_outcomes_for_job(
        db_session, _Job(), {"injected_memory_payloads": "not a list"}, {}
    )

    assert written == []


@pytest.mark.asyncio
async def test_standing_comes_back_grouped_by_method(db_session):
    memory_id = uuid4()
    for satisfied in (True, False, True):
        db_session.add(
            _outcome(method_memory_id=memory_id, contract_satisfied=satisfied)
        )
    await db_session.flush()

    result = await standing.standing_for(db_session, [str(memory_id), "not-a-uuid"])

    assert result[str(memory_id)]["runs"] == 3
    assert result[str(memory_id)]["contracts_satisfied"] == 2


@pytest.mark.asyncio
async def test_methods_are_recalled_alongside_ordinary_memories(db_session):
    """Ranked together for one budget, findings win every time: ten memories
    were injected into each recent job and not one was a method, so nothing
    recorded was ever reused or scored."""
    from app.models.memory import ConversationMemory
    from app.services.agent_job_memory_service import agent_job_memory_service

    user_id = uuid4()
    record = agent_method_record.build_record(
        name="measure with dependent chains",
        procedure=["emit a dependent chain", "anchor to a known-cost op"],
        prevents="the compiler reshaping the loop under test",
        derived_from=["none"],
        available_finding_types=[],
    )
    db_session.add(
        ConversationMemory(
            user_id=user_id,
            memory_type="pattern",
            content=agent_method_record.render(record),
            importance_score=0.9,
            tags=["method"],
        )
    )
    # Ordinary findings that would otherwise fill the whole budget.
    for index in range(12):
        db_session.add(
            ConversationMemory(
                user_id=user_id,
                memory_type="finding",
                content=f"finding number {index} about instruction fusion",
                importance_score=0.8,
            )
        )
    await db_session.commit()

    from app.models.agent_job import AgentJob, AgentJobStatus

    job = AgentJob(
        name="later job",
        goal="measure instruction latency on this host",
        job_type="research",
        user_id=user_id,
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=3,
        max_tool_calls=5,
        max_llm_calls=5,
        max_runtime_minutes=5,
    )
    db_session.add(job)
    await db_session.commit()

    recalled = await agent_job_memory_service.get_relevant_memories_for_job(
        job, str(user_id), db_session, limit=3, memory_types_override=["pattern"]
    )

    assert any(
        agent_method_record.parse(str(m.content or "")) for m in recalled
    ), "a reserved method query must return the method"


class _Memory:
    def __init__(self, name):
        self.id = uuid4()
        self.name = name


def _standing_of(memory, *, graded, satisfied):
    return {
        str(memory.id): standing.summarize(
            [
                _outcome(
                    method_memory_id=memory.id,
                    contract_enabled=True,
                    contract_satisfied=index < satisfied,
                )
                for index in range(graded)
            ]
        )
    }


def test_an_established_record_orders_ahead_of_an_established_bad_one():
    good, bad = _Memory("good"), _Memory("bad")
    table = {}
    table.update(_standing_of(good, graded=4, satisfied=4))
    table.update(_standing_of(bad, graded=4, satisfied=0))

    # Deliberately handed in the wrong order.
    ordered = standing.rank([bad, good], table)

    assert ordered[0] is good
    assert ordered[-1] is bad


def test_a_barely_used_method_keeps_the_order_relevance_gave_it():
    """One run is not evidence, and sorting on it dresses noise as judgement."""
    first, second = _Memory("first"), _Memory("second")
    table = _standing_of(second, graded=1, satisfied=1)

    assert standing.rank([first, second], table) == [first, second]


def test_an_untried_method_is_not_pushed_behind_a_tried_one():
    tried, untried = _Memory("tried"), _Memory("untried")
    table = _standing_of(tried, graded=2, satisfied=2)

    assert standing.rank([untried, tried], table) == [untried, tried]


def test_a_poor_record_is_demoted_rather_than_dropped():
    """Removing it would end its record there; it may have been the wrong
    method or may have been handed hard problems."""
    bad, plain = _Memory("bad"), _Memory("plain")
    table = _standing_of(bad, graded=3, satisfied=0)

    ordered = standing.rank([bad, plain], table)

    assert bad in ordered, "a demoted method is still recalled"
    assert ordered[-1] is bad


def test_a_caution_is_raised_only_once_the_record_is_established():
    assert (
        standing.caution(standing.summarize([_outcome(contract_satisfied=False)])) == ""
    )

    poor = standing.summarize([_outcome(contract_satisfied=False) for _ in range(3)])
    assert "none of the 3 runs" in standing.caution(poor)


def test_a_good_record_raises_no_caution():
    good = standing.summarize([_outcome(contract_satisfied=True) for _ in range(4)])

    assert standing.caution(good) == ""


def test_a_caution_says_what_happened_rather_than_passing_a_verdict():
    mixed = standing.summarize([_outcome(contract_satisfied=i == 0) for i in range(4)])

    line = standing.caution(mixed)

    assert "1 of 4 runs" in line
    assert "bad" not in line.lower() and "wrong" not in line.lower()
