"""A run that stopped because it needs a person must appear in the queue.

It did not. An approval checkpoint produced a queue row and a recurring job
produced a recovery row, but a one-shot run that paused as `blocked_needs_input`
matched neither branch, so it appeared in no queue at all. Six were found
waiting 8 to 11 days in a live database, one of them a pipeline stage -- which
means the whole DAG behind it had stopped with nobody told.

These use the real bound composer rather than a stub: the defect was in which
jobs the projector selected, and a test with its own fake selection would have
agreed with itself.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from app.api.endpoints.agent_jobs import _build_checkpoint_queue_items
from app.models.agent_job import AgentJob, AgentJobStatus


def _job(**kwargs) -> AgentJob:
    base = dict(
        id=uuid4(),
        name="attention-survey: writeup",
        goal="Write up the survey",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.PAUSED.value,
        current_phase="blocked_needs_input",
        iteration=6,
        max_iterations=100,
        tool_calls_used=10,
        max_tool_calls=100,
        llm_calls_used=10,
        max_llm_calls=100,
        max_runtime_minutes=60,
        created_at=datetime.utcnow() - timedelta(days=9),
        last_activity_at=datetime.utcnow() - timedelta(days=9),
        results={
            "blocked": {
                "reason": "3 consecutive rounds produced no new findings",
                "missing": ["finding_type:synthesis_document"],
                "resumable": True,
            }
        },
    )
    base.update(kwargs)
    return AgentJob(**base)


def _rows(*jobs):
    return _build_checkpoint_queue_items(list(jobs), [], monitor_health_rows=[])


def _blocked(*jobs):
    return [row for row in _rows(*jobs) if row.item_type == "blocked_run"]


class TestItIsVisibleAtAll:
    def test_a_blocked_run_produces_a_queue_row(self):
        assert len(_blocked(_job())) == 1

    def test_the_row_says_why_the_run_gave_up(self):
        # The run usually knows. That sentence is the most useful thing here.
        (row,) = _blocked(_job())
        assert "no new findings" in (row.summary or "")

    def test_the_row_names_what_is_missing(self):
        (row,) = _blocked(_job())
        assert row.checkpoint["missing"] == ["finding_type:synthesis_document"]

    def test_it_is_not_labelled_as_an_approval(self):
        # Nothing is proposed for sign-off here; something is absent. Calling it
        # an approval would tell the reader to look for a decision to confirm.
        (row,) = _blocked(_job())
        assert row.reason_code == "needs_input"
        assert row.reason_label == "Waiting on an answer"

    def test_the_age_is_carried_so_it_can_be_sorted_to_the_top(self):
        (row,) = _blocked(_job())
        assert row.age_minutes is not None and row.age_minutes > 0


class TestWhatItOffers:
    def test_resume_is_offered_when_the_run_said_it_could_resume(self):
        (row,) = _blocked(_job())
        assert [action.action for action in row.actions] == ["resume"]
        assert row.recommended_action == "resume"

    def test_no_action_is_offered_when_the_run_cannot_be_resumed(self):
        # An action that would fail is worse than none.
        job = _job(
            results={"blocked": {"reason": "gone", "missing": [], "resumable": False}}
        )
        (row,) = _blocked(job)
        assert row.actions == []
        assert row.recommended_action is None

    def test_a_run_that_recorded_nothing_still_surfaces(self):
        # Losing the row because the payload is absent would restore the bug for
        # exactly the runs that explain themselves least.
        job = _job(results={})
        (row,) = _blocked(job)
        assert row.item_type == "blocked_run"


class TestItDoesNotDisturbTheRestOfTheQueue:
    def test_an_approval_checkpoint_is_still_an_approval(self):
        job = _job(
            current_phase="awaiting_approval",
            results={
                "approval_checkpoint": {
                    "status": "pending",
                    "message": "Approve this step",
                }
            },
        )
        types = {row.item_type for row in _rows(job)}
        assert "blocked_run" not in types

    def test_one_job_never_yields_two_rows(self):
        assert len(_rows(_job())) == 1

    def test_a_paused_job_that_is_not_blocked_is_unchanged(self):
        # Previously produced nothing, and still should: nobody is waiting on an
        # answer, so a queue row would be noise.
        job = _job(current_phase="finalizing", results={})
        assert _rows(job) == []

    def test_a_running_job_is_not_swept_in(self):
        job = _job(status=AgentJobStatus.RUNNING.value)
        assert _blocked(job) == []
