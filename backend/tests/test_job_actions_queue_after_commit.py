"""An operator action queues its job once the action is committed.

Restarting a completed pipeline stage on the live stack: the action changed
the row and sent the task before committing, the worker took the message
45 ms later, read the row as still ``completed``, called it a duplicate
delivery and dropped it. The commit then set the job to ``pending``, where it
stayed.
"""

from types import SimpleNamespace

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.modules.autonomy.application import job_action_dispatch
from app.modules.autonomy.application.job_action_recovery import perform_recovery_action


class _Task:
    def __init__(self):
        self.sent = []

    def delay(self, *args):
        self.sent.append(args)


class _NoQuickStart:
    async def refined_repo_retry(self, job, **_):
        return None


def _deps(task):
    return SimpleNamespace(
        execute_agent_job_task=task,
        quick_start_relaunch_dispatcher=_NoQuickStart(),
        append_operator_intervention=lambda *args, **kwargs: None,
    )


async def _finished_stage(db, user):
    job = AgentJob(
        name="plan: implement",
        goal="Write it",
        job_type="coding",
        user_id=user.id,
        status=AgentJobStatus.COMPLETED.value,
        iteration=10,
        max_iterations=10,
        config={"pipeline_stage": "implement"},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


@pytest.mark.asyncio
async def test_a_restart_is_not_sent_before_it_is_committed(db_session, test_user):
    job = await _finished_stage(db_session, test_user)
    task = _Task()

    await perform_recovery_action(
        job, "restart", None, deps=_deps(task), db=db_session, current_user=test_user
    )

    assert job.status == AgentJobStatus.PENDING.value
    assert task.sent == [], "the worker would read the row as still completed"

    await db_session.commit()

    assert task.sent == [(str(job.id), str(test_user.id))]


@pytest.mark.asyncio
async def test_a_restart_that_is_rolled_back_sends_nothing(db_session, test_user):
    job = await _finished_stage(db_session, test_user)
    task = _Task()

    await perform_recovery_action(
        job, "restart", None, deps=_deps(task), db=db_session, current_user=test_user
    )
    await db_session.rollback()

    assert task.sent == []


@pytest.mark.asyncio
async def test_with_nothing_to_commit_it_is_sent_at_once(db_session, test_user):
    job = await _finished_stage(db_session, test_user)
    task = _Task()

    job_action_dispatch.queue_job(db_session, _deps(task), job.id, test_user.id)

    assert task.sent == [(str(job.id), str(test_user.id))]


def test_a_stand_in_session_is_sent_at_once():
    task = _Task()

    job_action_dispatch.queue_job(object(), _deps(task), "job", "user")

    assert task.sent == [("job", "user")]


def test_no_job_action_sends_its_task_directly():
    """The seam is the helper: a bare ``.delay`` here is the race again."""
    import pathlib

    folder = pathlib.Path(job_action_dispatch.__file__).parent
    offenders = [
        path.name
        for path in sorted(folder.glob("job_action_*.py"))
        if path.name != "job_action_dispatch.py"
        and "execute_agent_job_task.delay" in path.read_text()
    ]

    assert offenders == []
