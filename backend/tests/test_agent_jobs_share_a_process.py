"""Several agent jobs in one worker process, each on its own event loop.

A running job holds a worker *process* for hours while it waits on a model,
at several hundred megabytes each. Running jobs in threads of one process
shares that memory, and gives each job its own loop, so a job that blocks its
loop stalls its own heartbeat and nobody else's.

What stood in the way was state that assumed one live loop per process. These
tests use real loops in real threads, because every one of those assumptions
holds when loops are run one after another.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.tasks import agent_job_tasks
from app.utils.per_loop import PerLoop
from tests.test_golden_agent_tasks import (  # noqa: F401  (autouse fixtures)
    ScriptedActionService,
    ScriptedLLM,
    _no_celery_dispatch,
    _no_redis_feature_flags,
    _scripted_memory_service_llm,
    decision,
)


def _in_live_loops(count, body):
    """Run `body(index)` on `count` loops that are all alive at once.

    Each thread runs its coroutine up to a barrier every thread must reach, so
    no loop has finished when another starts -- the case a sequential run
    never produces. Returns the results in order; re-raises a thread's error.
    """
    together = threading.Barrier(count, timeout=30)
    results, errors = [None] * count, []

    def run(index):
        async def main():
            first = await body(index, 0)
            await asyncio.to_thread(together.wait)
            second = await body(index, 1)
            return first, second

        try:
            results[index] = asyncio.run(main())
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            errors.append(exc)
            together.abort()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    if errors:
        raise errors[0]
    return results


class TestOneValuePerLiveLoop:
    def test_a_loop_keeps_its_value_while_another_loop_is_alive(self):
        made = []
        held = PerLoop(lambda: made.append(1) or object())

        async def body(_index, _round):
            return held.get()

        results = _in_live_loops(3, body)

        assert all(first is second for first, second in results)
        assert len({id(first) for first, _ in results}) == 3
        assert len(made) == 3

    def test_a_closed_loops_value_is_dropped(self):
        held = PerLoop(object)

        async def use():
            return held.get()

        asyncio.run(use())
        asyncio.run(use())

        assert len(held) == 1  # the second run swept the first's

    def test_the_redis_client_is_not_evicted_by_another_loop(self):
        # It dropped every client but the caller's on each access: with two
        # jobs in one process that is a new connection per call, never closed.
        from app.core import cache

        async def body(_index, _round):
            return await cache.get_redis_client()

        results = _in_live_loops(3, body)

        assert all(first is second for first, second in results)
        assert len({id(first) for first, _ in results}) == 3

    def test_the_model_call_semaphore_is_not_remade_by_another_loop(self):
        # Remade on every access, it never has a holder and limits nothing.
        from app.services import llm_concurrency

        async def body(_index, _round):
            return llm_concurrency._local_semaphore()

        results = _in_live_loops(3, body)

        assert all(first is second for first, second in results)

    def test_the_model_service_has_a_client_per_loop(self):
        from app.services.llm_service import LLMService

        service = LLMService()

        async def body(_index, _round):
            return service.client

        results = _in_live_loops(2, body)

        assert all(first is second for first, second in results)
        assert results[0][0] is not results[1][0]

    def test_a_client_assigned_by_hand_is_used_on_every_loop(self):
        from app.services.llm_service import LLMService

        service = LLMService()
        fake = object()
        service.client = fake

        async def body(_index, _round):
            return service.client

        assert _in_live_loops(2, body) == [(fake, fake), (fake, fake)]


class TestTheWallClockIsTheTasksOwn:
    """Celery's soft limit is a signal; a thread pool never delivers it."""

    @pytest.mark.asyncio
    async def test_a_job_past_its_wall_clock_ends_as_a_soft_time_limit(self):
        cleaned_up = []

        async def job():
            try:
                await asyncio.sleep(30)
            finally:
                cleaned_up.append(True)

        with pytest.raises(SoftTimeLimitExceeded):
            await agent_job_tasks._within_wall_clock(job(), 0.05)

        assert cleaned_up == [True]
        assert agent_job_tasks.is_terminal_task_error(SoftTimeLimitExceeded())

    @pytest.mark.asyncio
    async def test_a_job_inside_it_returns_what_it_returned(self):
        async def job():
            return {"status": "completed"}

        assert await agent_job_tasks._within_wall_clock(job(), 5) == {
            "status": "completed"
        }


async def _golden_job(database: Path, index: int):
    """One real executor loop, on this thread's loop, against its own file."""
    from app.core.database import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    executor = AutonomousAgentExecutor()
    llm = ScriptedLLM(
        [
            decision(tool="search_documents", params={"query": f"job {index} q{n}"})
            for n in range(3)
        ]
        + [decision(goal_achieved=True, reasoning=f"job {index} is done")]
    )
    actions = ScriptedActionService(
        {
            "search_documents": lambda action: {
                "success": True,
                "findings": [
                    {
                        "title": f"job {index}: {action['params']['query']}",
                        "category": "insight",
                        "content": f"evidence for job {index}",
                    }
                ],
            }
        }
    )
    executor.llm_service = llm
    executor.decision_parser.llm_service = llm
    executor.action_service = actions
    try:
        async with sessions() as db:
            job = AgentJob(
                name=f"Threaded {index}",
                goal=f"Find evidence, job {index}",
                job_type="research",
                user_id=uuid4(),
                status=AgentJobStatus.RUNNING.value,
                config={},
                max_iterations=6,
                max_tool_calls=20,
                max_llm_calls=40,
                max_runtime_minutes=10,
            )
            db.add(job)
            await db.commit()
            await executor._run_autonomous_loop(job, None, None, db, None)
            findings = (job.results or {}).get("findings") or []
            return {
                "status": job.status,
                "searches": [
                    str(call.get("params", {}).get("query"))
                    for call in actions.calls
                    if call.get("tool") == "search_documents"
                ],
                "findings": [str(row.get("title")) for row in findings],
            }
    finally:
        await engine.dispose()


def test_four_jobs_run_at_once_in_one_process_and_do_not_mix():
    """The real loop, four times over, with every loop alive together."""
    count = 4
    together = threading.Barrier(count, timeout=120)
    results, errors = [None] * count, []

    with tempfile.TemporaryDirectory() as directory:

        def run(index):
            async def main():
                # Every job is on its loop before any of them works.
                await asyncio.to_thread(together.wait)
                return await _golden_job(Path(directory) / f"{index}.db", index)

            try:
                results[index] = asyncio.run(main())
            except BaseException as exc:  # noqa: BLE001 - reported to the test
                errors.append(exc)
                together.abort()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

    assert not errors, errors
    for index, outcome in enumerate(results):
        assert outcome is not None, f"job {index} did not finish"
        assert outcome["status"] == AgentJobStatus.COMPLETED.value
        assert outcome["searches"] == [f"job {index} q{n}" for n in range(3)]
        # Nothing another job found ended up in this one's results.
        assert outcome["findings"], f"job {index} recorded no findings"
        assert all(title.startswith(f"job {index}:") for title in outcome["findings"])
