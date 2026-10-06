"""Work is queued when the rows it reads are committed, and not otherwise."""

import pytest

from app.models.agent_job import AgentJob
from app.services import job_dispatch
from app.tasks.agent_job_tasks import execute_agent_job_task

pytestmark = pytest.mark.unit


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(
        execute_agent_job_task, "delay", lambda *args: calls.append(args)
    )
    return calls


def _job(user_id):
    return AgentJob(
        name="dispatch", goal="g", job_type="research", user_id=user_id, config={}
    )


async def test_nothing_is_sent_before_the_commit(db_session, test_user, sent):
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.flush()
    job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)
    assert sent == []

    await db_session.commit()
    assert sent == [(str(job.id), str(test_user.id))]


async def test_a_row_already_committed_is_sent_at_once(db_session, test_user, sent):
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.commit()
    job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)
    assert sent == [(str(job.id), str(test_user.id))]


async def test_a_failed_immediate_send_reaches_the_caller(
    db_session, test_user, monkeypatch
):
    # The chain orchestrator records chain_dispatch_failed from this error.
    def _down(*_args):
        raise ConnectionError("broker unreachable")

    monkeypatch.setattr(execute_agent_job_task, "delay", _down)
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.commit()
    with pytest.raises(ConnectionError):
        job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)


async def test_a_rollback_sends_nothing(db_session, test_user, sent):
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.flush()
    job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)
    await db_session.rollback()
    await db_session.commit()
    assert sent == []


async def test_each_message_is_sent_once(db_session, test_user, sent):
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.flush()
    job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)
    job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)
    await db_session.commit()
    await db_session.commit()
    assert len(sent) == 2


async def test_a_failed_deferred_send_does_not_undo_the_commit(
    db_session, test_user, monkeypatch
):
    def _down(*_args):
        raise ConnectionError("broker unreachable")

    monkeypatch.setattr(execute_agent_job_task, "delay", _down)
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.flush()
    job_dispatch.enqueue_agent_job(db_session, job.id, test_user.id)
    await db_session.commit()
    assert await db_session.get(AgentJob, job.id) is not None


async def test_the_backlog_start_queues_after_its_commit(db_session, test_user, sent):
    from app.models.coding_backlog import CodingBacklogItem
    from app.services.coding_backlog_items import create_orchestrator_job

    item = CodingBacklogItem(
        user_id=test_user.id, title="t", portfolio_goal="g", status="draft"
    )
    db_session.add(item)
    await db_session.flush()
    job = await create_orchestrator_job(item, db=db_session)
    assert sent == []
    await db_session.commit()
    assert sent == [(str(job.id), str(test_user.id))]
    # The timeline says where it started from, not where it ended up.
    started = item.decomposition["backlog_timeline"][-1]
    assert (started["previous_status"], started["new_status"]) == (
        "draft",
        "running",
    )
