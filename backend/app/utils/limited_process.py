"""Run a command under resource limits without a fork hook.

Three places capped a locally run program with ``preexec_fn``: a function
Python runs in the child between ``fork`` and ``exec``. That is documented as
unsafe once a process has threads -- the child is a copy of one thread, holding
whatever locks the others held, and the hook here began with an ``import``,
which takes one. Every such call was already made from a worker thread, and a
worker that runs each job in its own thread has them all the time.

Here the limits are set by a small launcher that is itself the program
``subprocess`` starts: a fresh interpreter sets its own limits and then
replaces itself with the command. Nothing runs in a forked copy of this
process.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence

#: Runs as ``python -I -S -c LAUNCHER '<limits json>' command...``.
LAUNCHER = (
    "import json, os, resource, sys\n"
    "for name, value in json.loads(sys.argv[1]).items():\n"
    "    try:\n"
    "        resource.setrlimit(getattr(resource, name), (value, value))\n"
    "    except Exception:\n"
    "        pass\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)


def process_limits(
    *, cpu_seconds: int, memory_bytes: int, file_bytes: int, open_files: int
) -> Dict[str, int]:
    """The limits a locally run program gets; no core dump, ever."""
    return {
        "RLIMIT_CPU": int(cpu_seconds),
        "RLIMIT_CORE": 0,
        "RLIMIT_FSIZE": int(file_bytes),
        "RLIMIT_NOFILE": int(open_files),
        "RLIMIT_AS": int(memory_bytes),
    }


def with_limits(command: Sequence[str], limits: Dict[str, int]) -> List[str]:
    """``command`` wrapped so it starts under ``limits``.

    A limit the platform refuses is skipped, as it was before. Where there
    are no POSIX limits the command is returned as it came.
    """
    if os.name != "posix":
        return list(command)
    return [sys.executable, "-I", "-S", "-c", LAUNCHER, json.dumps(limits), *command]


__all__ = ["LAUNCHER", "process_limits", "with_limits"]
