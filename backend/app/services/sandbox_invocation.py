"""Carry a journaled tool identity through nested sandbox calls."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterator, Optional


@dataclass
class SandboxInvocation:
    job_id: str
    key: str
    cancelled: Optional[Callable[[], Awaitable[bool]]] = None
    calls: int = 0
    preparations: int = 0
    runs: list = field(default_factory=list)
    defer: bool = False
    deadline_at: Optional[float] = None
    resume_reference: Optional[dict] = None
    workspaces: dict[str, int] = field(default_factory=dict)

    def next_key(self) -> str:
        self.calls += 1
        return f"{self.job_id}:{self.key}:{self.calls}"

    def preparation_key(self) -> str:
        self.preparations += 1
        return f"{self.job_id}:{self.key}:files:{self.preparations}"

    def workspace_key(self, workdir: str) -> int:
        """Stable workspace order within an action, including cache-only calls."""
        return self.workspaces.setdefault(
            os.path.realpath(workdir), len(self.workspaces) + 1
        )


current: ContextVar[Optional[SandboxInvocation]] = ContextVar(
    "sandbox_invocation", default=None
)
_synchronous_wait: ContextVar[bool] = ContextVar(
    "sandbox_synchronous_wait", default=False
)


def synchronous_wait_required() -> bool:
    return _synchronous_wait.get()


@contextmanager
def synchronous_wait() -> Iterator[None]:
    """Keep nested Jobs inside a caller's resource lifetime (for example a lock)."""
    token = _synchronous_wait.set(True)
    try:
        yield
    finally:
        _synchronous_wait.reset(token)


class SandboxStepPending(BaseException):
    """A suspended tool call, deliberately outside ordinary failure handlers."""

    def __init__(self, reference: dict):
        self.reference = dict(reference)
        super().__init__("Waiting for an independent sandbox experiment")


@contextmanager
def invocation(
    job_id: str,
    key: str,
    *,
    cancelled: Optional[Callable[[], Awaitable[bool]]] = None,
    defer: bool = False,
    deadline_at: Optional[float] = None,
    resume_reference: Optional[dict] = None,
) -> Iterator[SandboxInvocation]:
    context = SandboxInvocation(
        job_id,
        key,
        cancelled,
        defer=defer,
        deadline_at=deadline_at,
        resume_reference=dict(resume_reference) if resume_reference else None,
    )
    token = current.set(context)
    try:
        yield context
    finally:
        current.reset(token)
