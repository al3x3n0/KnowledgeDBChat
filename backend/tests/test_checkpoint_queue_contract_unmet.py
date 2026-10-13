"""A run that finished without meeting its contract must be visible.

A run that gives up early pauses as blocked_needs_input and shows in the queue.
A run that exhausts its iteration budget with the contract unmet is marked
**completed** — so the worse outcome carried the better-looking status, and
nothing surfaced it. Twenty-eight such runs sat in a live database over a
fortnight, every one reporting success while delivering nothing its contract
asked for. A pipeline's writeup stage was among them: it called the right tool,
the call succeeded, and the evidence never arrived.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from app.api.endpoints.agent_jobs import _build_checkpoint_queue_items
from app.models.agent_job import AgentJob, AgentJobStatus

UNMET = {
    "phase": "completed_contract_unmet",
    "reason": "The run ended with its goal contract unsatisfied: "
    "finding_type:synthesis_document",
    "missing": ["finding_type:synthesis_document"],
}


def _job(**kwargs) -> AgentJob:
    base = dict(
        id=uuid4(),
        name="attention-survey: writeup",
        goal="Write it up",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.COMPLETED.value,
        iteration=8,
        max_iterations=8,
        tool_calls_used=10,
        max_tool_calls=100,
        llm_calls_used=10,
        max_llm_calls=100,
        max_runtime_minutes=60,
        created_at=datetime.utcnow() - timedelta(hours=3),
        completed_at=datetime.utcnow() - timedelta(hours=1),
        execution_log=[{"phase": "iteration_complete"}, UNMET],
        results={},
    )
    base.update(kwargs)
    return AgentJob(**base)


def _rows(*jobs):
    return _build_checkpoint_queue_items(list(jobs), [], monitor_health_rows=[])


def _unmet(*jobs):
    return [row for row in _rows(*jobs) if row.item_type == "contract_unmet"]


class TestItIsVisibleAtAll:
    def test_a_completed_but_unsatisfied_run_produces_a_row(self):
        assert len(_unmet(_job())) == 1

    def test_the_row_names_what_was_missing(self):
        (row,) = _unmet(_job())
        assert row.checkpoint["missing"] == ["finding_type:synthesis_document"]

    def test_the_row_says_the_contract_was_not_met(self):
        # Not "completed". The status says the run stopped; this says it did
        # not do what it was asked.
        (row,) = _unmet(_job())
        assert row.reason_code == "contract_unmet"
        assert row.reason_label == "Finished without meeting its contract"


class TestWhatItOffers:
    def test_restart_is_recommended_and_relaunch_is_available(self):
        # restart resets iteration and progress, so the run gets its budget
        # back rather than immediately re-hitting the cap it just hit.
        (row,) = _unmet(_job())
        assert [action.action for action in row.actions] == ["restart", "relaunch"]
        assert row.recommended_action == "restart"


class TestItRanksBelowThingsThatAreWaiting:
    def test_it_is_lower_priority_than_a_blocked_run(self):
        # Nobody is waiting on this one. It is a quality signal about work
        # already reported as done, not a request for a decision.
        blocked = _job(
            status=AgentJobStatus.PAUSED.value,
            current_phase="blocked_needs_input",
            execution_log=[],
            results={"blocked": {"reason": "stuck", "missing": [], "resumable": True}},
        )
        (blocked_row,) = [r for r in _rows(blocked) if r.item_type == "blocked_run"]
        (unmet_row,) = _unmet(_job())
        assert unmet_row.priority < blocked_row.priority


class TestWhatItMustNotClaim:
    def test_a_completed_run_that_met_its_contract_is_not_listed(self):
        assert _unmet(_job(execution_log=[{"phase": "iteration_complete"}])) == []

    def test_a_still_running_job_is_not_listed(self):
        # The entry can only be written as a run finishes; matching on the log
        # alone would surface a job mid-flight.
        assert _unmet(_job(status=AgentJobStatus.RUNNING.value)) == []

    def test_a_blocked_run_is_reported_as_blocked_not_as_unmet(self):
        # Two different situations: one is waiting for a person, the other has
        # already stopped. Reporting both the same way loses that.
        blocked = _job(
            status=AgentJobStatus.PAUSED.value,
            current_phase="blocked_needs_input",
            execution_log=[UNMET],
            results={"blocked": {"reason": "stuck", "missing": [], "resumable": True}},
        )
        types = {row.item_type for row in _rows(blocked)}
        assert types == {"blocked_run"}

    def test_one_job_yields_one_row(self):
        assert len(_rows(_job())) == 1


# A stage of a plan is a different case: the stages behind it start only when
# its chain fires, and an unmet contract does not fire it. The first plan
# approved live stopped at its first stage with three behind it, and the row
# said only that one run had not met its contract.


def _stage(name, then=None):
    child = {"name": f"plan: {name}", "config": {"pipeline_stage": name}}
    if then is not None:
        child["chain_config"] = {"child_jobs": [then]}
    return child


PLAN_BEHIND = {
    "trigger_condition": "on_complete",
    "child_jobs": [_stage("apply", _stage("check", _stage("benchmark")))],
}


class TestAStageWithOthersBehindIt:
    def test_the_row_names_the_stages_that_did_not_start(self):
        (row,) = _unmet(_job(chain_config=PLAN_BEHIND, chain_triggered=False))

        assert row.checkpoint["waiting_stages"] == ["apply", "check", "benchmark"]
        assert "3 later stage(s) will not start" in row.summary
        assert "apply, check, benchmark" in row.summary
        # What was missing is still said.
        assert "synthesis_document" in row.summary

    def test_it_ranks_above_a_run_nobody_waits_on(self):
        (stopped,) = _unmet(_job(chain_config=PLAN_BEHIND, chain_triggered=False))
        (alone,) = _unmet(_job())

        assert stopped.priority > alone.priority

    def test_a_chain_that_already_fired_has_nothing_waiting(self):
        (row,) = _unmet(_job(chain_config=PLAN_BEHIND, chain_triggered=True))

        assert row.checkpoint["waiting_stages"] == []
        assert "later stage" not in row.summary

    def test_a_run_with_no_chain_is_unchanged(self):
        (row,) = _unmet(_job())

        assert row.checkpoint["waiting_stages"] == []
        assert row.priority == 60

    def test_a_malformed_chain_does_not_break_the_queue(self):
        chain = {"child_jobs": ["not a stage", {"config": "nor this"}, None]}

        (row,) = _unmet(_job(chain_config=chain, chain_triggered=False))

        assert row.checkpoint["waiting_stages"] == []
