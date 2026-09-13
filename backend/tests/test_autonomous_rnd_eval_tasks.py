from __future__ import annotations

import asyncio

import pytest
from celery.exceptions import MaxRetriesExceededError
from sqlalchemy import select

from app.core.config import settings
from app.models.agent_job import AgentJobStatus
from app.models.autonomous_rnd_eval_launch import (
    EVAL_LAUNCH_STATUS_COMPLETED,
    EVAL_LAUNCH_STATUS_FAILED,
    EVAL_LAUNCH_STATUS_RUNNING,
    AutonomousRndEvalLaunch,
)
from app.services.autonomous_rnd_eval_launch_service import (
    autonomous_rnd_eval_launch_service,
)
from app.services.autonomous_rnd_eval_service import autonomous_rnd_eval_harness
from app.tasks import autonomous_rnd_eval_tasks
from tests.conftest import TestSessionLocal


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@pytest.fixture
def celery_session(monkeypatch):
    monkeypatch.setattr(
        autonomous_rnd_eval_tasks, "create_celery_session", lambda: TestSessionLocal
    )
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED", True)


def _create_launch(db_session, user_id, *, settle: bool):
    async def _seed():
        suite = autonomous_rnd_eval_harness.load_builtin_suite("compiler_research_v1")
        launch, jobs = await autonomous_rnd_eval_launch_service.launch(
            db_session, user_id=user_id, suite=suite, trials_override=1
        )
        if settle:
            for job in jobs:
                job.status = AgentJobStatus.COMPLETED.value
        await db_session.commit()
        return launch.id

    return _run(_seed())


def test_finalize_task_requests_a_retry_while_trials_are_running(
    db_session, test_user, celery_session
):
    launch_id = _create_launch(db_session, test_user.id, settle=False)

    result = _run(autonomous_rnd_eval_tasks._async_finalize(str(launch_id)))

    assert result == {
        "finalized": False,
        "retry": True,
        "reason": "trials_running",
    }


def test_finalize_task_grades_a_settled_launch(db_session, test_user, celery_session):
    launch_id = _create_launch(db_session, test_user.id, settle=True)

    result = _run(autonomous_rnd_eval_tasks._async_finalize(str(launch_id)))

    assert result["finalized"] is True
    assert "run_id" in result

    async def _reload():
        return (
            await db_session.execute(
                select(AutonomousRndEvalLaunch).where(
                    AutonomousRndEvalLaunch.id == launch_id
                )
            )
        ).scalar_one()

    launch = _run(_reload())
    assert launch.status == EVAL_LAUNCH_STATUS_COMPLETED
    assert str(launch.run_id) == result["run_id"]


def test_finalize_task_ignores_an_unknown_launch(db_session, celery_session):
    result = _run(
        autonomous_rnd_eval_tasks._async_finalize(
            "00000000-0000-0000-0000-000000000001"
        )
    )

    assert result == {"finalized": False, "reason": "launch_missing"}


def test_exhausted_polling_abandons_the_launch_instead_of_leaving_it_running(
    db_session, test_user, celery_session, monkeypatch
):
    launch_id = _create_launch(db_session, test_user.id, settle=False)

    def _exhausted(**_kwargs):
        raise MaxRetriesExceededError()

    monkeypatch.setattr(
        autonomous_rnd_eval_tasks.finalize_autonomous_rnd_eval_launch,
        "retry",
        _exhausted,
    )

    result = autonomous_rnd_eval_tasks.finalize_autonomous_rnd_eval_launch(
        str(launch_id)
    )

    assert result["abandoned"] is True

    async def _reload():
        return (
            await db_session.execute(
                select(AutonomousRndEvalLaunch).where(
                    AutonomousRndEvalLaunch.id == launch_id
                )
            )
        ).scalar_one()

    launch = _run(_reload())
    assert launch.status == EVAL_LAUNCH_STATUS_FAILED
    assert "did not reach a terminal state" in launch.error
    assert launch.completed_at is not None


def test_abandon_is_a_no_op_once_a_launch_has_settled(
    db_session, test_user, celery_session
):
    launch_id = _create_launch(db_session, test_user.id, settle=True)
    _run(autonomous_rnd_eval_tasks._async_finalize(str(launch_id)))

    result = _run(autonomous_rnd_eval_tasks._async_abandon(str(launch_id), "stalled"))

    assert result == {"abandoned": False, "reason": "not_running"}

    async def _reload():
        return (
            await db_session.execute(
                select(AutonomousRndEvalLaunch).where(
                    AutonomousRndEvalLaunch.id == launch_id
                )
            )
        ).scalar_one()

    launch = _run(_reload())
    assert launch.status != EVAL_LAUNCH_STATUS_RUNNING
    assert launch.error is None
