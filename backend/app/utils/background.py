"""Start work that nobody awaits, without losing it.

``asyncio.create_task(coro())`` on a line by itself is a task only the event
loop refers to, and the loop holds its tasks weakly: the garbage collector may
take one that is still running, which simply stops it. Nothing is raised and
nothing is logged. If it does finish with an error, the error is reported
only when the task is collected, as "Task exception was never retrieved".

`spawn` keeps the task until it ends and logs what it raised.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Set

from loguru import logger

_running: Set["asyncio.Task[Any]"] = set()


def _finished(task: "asyncio.Task[Any]") -> None:
    _running.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.opt(exception=error).error(
            f"Background task {task.get_name()} failed: {error}"
        )


def spawn(coro: Coroutine[Any, Any, Any], *, name: str) -> "asyncio.Task[Any]":
    """Run `coro` on the running loop and keep it alive until it is done."""
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _running.add(task)
    task.add_done_callback(_finished)
    return task


def running() -> int:
    """How many spawned tasks have not ended."""
    return len(_running)


__all__ = ["running", "spawn"]
