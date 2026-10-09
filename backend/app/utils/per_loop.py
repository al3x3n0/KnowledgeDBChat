"""One value per running event loop, for things that cannot cross loops.

An asyncio lock, semaphore or network client belongs to the loop it was first
used on. A process-wide singleton holding one is wrong the moment a second
loop appears, and a Celery worker makes a new loop for every task.

Three places had already met this and each kept its own table keyed by
``id(loop)``. All three assumed a process runs its loops one after another:
on each access they dropped every entry but the caller's, reasoning that
another loop's entry must be a dead one. With two loops alive at once -- two
jobs in two threads of one worker -- each access evicts the other's live
value: a Redis connection opened per call and never closed, a semaphore that
is new every time and so limits nothing.

Here an entry is dropped only when its loop has closed or been collected.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class PerLoop(Generic[T]):
    """`get()` returns the running loop's value, making it on first use."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._values: Dict[int, Tuple["weakref.ref[asyncio.AbstractEventLoop]", T]] = {}
        # Loops in different threads reach this table at the same time.
        self._guard = threading.Lock()

    def _sweep(self) -> None:
        for key, (ref, _value) in list(self._values.items()):
            loop = ref()
            if loop is None or loop.is_closed():
                # Its loop is gone: the value cannot be used again, and cannot
                # be closed cleanly from here either. Let it be collected.
                del self._values[key]

    def get(self) -> T:
        loop = asyncio.get_running_loop()
        with self._guard:
            self._sweep()
            held = self._values.get(id(loop))
            if held is None or held[0]() is not loop:
                held = (weakref.ref(loop), self._factory())
                self._values[id(loop)] = held
            return held[1]

    def peek(self) -> Optional[T]:
        """The running loop's value if it has one; never makes one."""
        loop = asyncio.get_running_loop()
        with self._guard:
            held = self._values.get(id(loop))
            return held[1] if held is not None and held[0]() is loop else None

    def pop(self) -> Optional[T]:
        """Forget the running loop's value and return it, for closing."""
        loop = asyncio.get_running_loop()
        with self._guard:
            held = self._values.pop(id(loop), None)
            return held[1] if held is not None and held[0]() is loop else None

    def clear(self) -> None:
        with self._guard:
            self._values.clear()

    def __len__(self) -> int:
        with self._guard:
            return len(self._values)


__all__ = ["PerLoop"]
