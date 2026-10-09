"""A child process is given a time limit of its own.

`await asyncio.wait_for(asyncio.to_thread(run), timeout)` reads as a limit and
is only a limit on the waiting. When it passes, the coroutine moves on and the
thread stays blocked on a child nobody will kill. The system-status route did
this for `docker version`: with the daemon hung, each request kept one of the
default executor's threads for ever, and that executor is what every
`asyncio.to_thread` in the process shares -- object storage, password hashing.

The limit has to be the child's: `subprocess.run(..., timeout=...)` kills it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
RUNNERS = {"run", "check_output", "check_call", "call"}


def _unlimited_calls():
    found, examined = [], 0
    for path in sorted(APP.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in RUNNERS
                and isinstance(func.value, ast.Name)
                and func.value.id in {"subprocess", "_subprocess"}
            ):
                continue
            examined += 1
            # `**kwargs` may carry it; that is the caller's to get right.
            if not any(k.arg in ("timeout", None) for k in node.keywords):
                found.append(f"{path.relative_to(APP.parent)}:{node.lineno}")
    return found, examined


def test_every_subprocess_call_names_a_timeout():
    found, examined = _unlimited_calls()

    # A check that examines nothing passes for ever.
    assert examined > 10
    assert found == [], (
        "subprocess call with no timeout; a hung child holds its thread for "
        f"ever: {found}"
    )


def test_a_timeout_on_the_child_ends_the_child():
    # The control for the rule: this is what the limit buys.
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5
        )
