"""Work started and not awaited is still kept, and still reported.

The event loop refers to its tasks weakly. `asyncio.create_task(coro())` on a
line by itself can be garbage-collected while it runs, which just stops it:
the admin model download would sit at "downloading" for ever. And an error in
one is reported only when the task is collected.
"""

from __future__ import annotations

import ast
import asyncio
import gc
from pathlib import Path
from uuid import uuid4

import pytest

from app.services import agent_service as agent_service_module
from app.services.agent_service import AgentService
from app.utils import background

APP = Path(__file__).resolve().parents[1] / "app"


def test_no_task_is_started_and_dropped():
    dropped, started = [], 0
    for path in sorted(APP.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            call = node.value if isinstance(node, ast.Expr) else None
            is_start = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"create_task", "ensure_future"}
            )
            started += int(is_start)
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"create_task", "ensure_future"}
            ):
                dropped.append(f"{path.relative_to(APP.parent)}:{node.lineno}")

    assert started > 5  # the scan sees the calls it is about
    assert dropped == [], (
        "task started with nothing holding it; use app.utils.background.spawn: "
        f"{dropped}"
    )


@pytest.mark.asyncio
async def test_a_spawned_task_survives_collection_and_is_then_forgotten():
    finished = asyncio.Event()

    async def work():
        await asyncio.sleep(0.05)
        finished.set()

    before = background.running()
    background.spawn(work(), name="kept")
    gc.collect()

    assert background.running() == before + 1
    await asyncio.wait_for(finished.wait(), timeout=2)
    await asyncio.sleep(0)
    assert background.running() == before


@pytest.mark.asyncio
async def test_what_a_spawned_task_raises_is_logged(monkeypatch):
    logged = []

    class _Logger:
        def opt(self, **_kwargs):
            return self

        def error(self, message):
            logged.append(message)

    monkeypatch.setattr(background, "logger", _Logger())

    async def work():
        raise ValueError("the download broke")

    task = background.spawn(work(), name="doomed")
    await asyncio.wait([task])
    await asyncio.sleep(0)

    assert logged and "doomed" in logged[0] and "the download broke" in logged[0]


@pytest.mark.asyncio
async def test_chat_memory_extraction_does_not_borrow_the_requests_session(
    monkeypatch,
):
    # It outlives the request. Handed the request's session, it used one that
    # was closed under it when the reply was sent.
    opened, used = [], []

    class _Session:
        async def __aenter__(self):
            opened.append(self)
            return self

        async def __aexit__(self, *exc):
            opened.append("closed")

    class _Memory:
        async def extract_and_store_memories(self, **kwargs):
            used.append(kwargs["db"])

    monkeypatch.setattr(agent_service_module, "async_session_factory", _Session)
    service = AgentService.__new__(AgentService)
    service.memory_integration = _Memory()

    await service._extract_memories_background(
        user_id=uuid4(), conversation_id=uuid4(), messages=[], preferences=None
    )

    assert used == [opened[0]]
    assert opened[-1] == "closed"
