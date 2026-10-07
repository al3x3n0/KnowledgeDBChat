"""No blocking call runs directly on the event loop.

A synchronous subprocess, sleep or HTTP call inside ``async def`` stops every
other coroutine in the process until it returns. In the API that is every
request being served: an MCP ``docker_execute`` ran ``docker pull`` (up to ten
minutes) and ``docker info`` (up to ten seconds) inline, an upload ran ffprobe
(up to thirty), and each login ran bcrypt (about a quarter second of CPU). In
a worker running several jobs on one loop it is every job. Each now goes
through ``asyncio.to_thread``.

The check reads source: a call to a known blocking function made directly in
an ``async def`` body. A call inside a nested ``def`` or ``lambda`` is not
counted, since that is how such work is handed to a thread; neither is a
function passed uncalled, as in ``to_thread(subprocess.run, cmd)``.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"

#: (module, function) pairs from the standard library and requests.
BLOCKING_MODULE_CALLS = {
    ("subprocess", "run"),
    ("subprocess", "check_output"),
    ("subprocess", "check_call"),
    ("subprocess", "call"),
    ("time", "sleep"),
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "request"),
}

#: This codebase's own synchronous helpers that block: each shells out or
#: burns CPU for long enough to matter.
BLOCKING_HELPERS = {
    "probe_duration_seconds",  # ffprobe, up to 30s
    "is_docker_available",  # docker info, up to 10s
    "_pull_image_sync",  # docker pull, up to 10 minutes
    "_render_graphviz",  # dot
    "compile_to_pdf",  # a LaTeX run
    "_run_behavioral_demo",  # a sandboxed program run
    "hash_password",  # bcrypt
    "verify_password",  # bcrypt
    "get_password_hash",  # bcrypt
}


def _direct(node):
    for child in ast.iter_child_nodes(node):
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            continue
        yield child
        yield from _direct(child)


def _blocking_calls(tree: ast.AST):
    for function in ast.walk(tree):
        if not isinstance(function, ast.AsyncFunctionDef):
            continue
        for node in _direct(function):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Attribute):
                owner = target.value
                if (
                    isinstance(owner, ast.Name)
                    and (owner.id.lstrip("_"), target.attr) in BLOCKING_MODULE_CALLS
                ):
                    yield function.name, node.lineno, f"{owner.id}.{target.attr}"
                elif target.attr in BLOCKING_HELPERS:
                    yield function.name, node.lineno, target.attr
            elif isinstance(target, ast.Name) and target.id in BLOCKING_HELPERS:
                yield function.name, node.lineno, target.id


def test_no_blocking_call_runs_on_the_event_loop():
    found = []
    for path in sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function, line, call in _blocking_calls(tree):
            found.append(f"{path.relative_to(APP)}:{line} {call} in {function}")
    assert not found, (
        "Blocking calls inside async functions stall every other coroutine in "
        "the process; wrap each in `await asyncio.to_thread(...)`:\n  "
        + "\n  ".join(found)
    )


def test_the_check_finds_what_it_is_looking_for():
    # The control: each shape the rule names is caught, and each way of
    # handing the work to a thread is not.
    source = """
import asyncio, subprocess, time

async def bad(svc):
    subprocess.run(["x"])
    time.sleep(1)
    svc.verify_password("a", "b")
    probe_duration_seconds("f")

async def fine(svc):
    await asyncio.to_thread(subprocess.run, ["x"])
    await asyncio.to_thread(svc.verify_password, "a", "b")
    await asyncio.to_thread(lambda: subprocess.run(["x"]))

    def helper():
        subprocess.run(["x"])

def sync_is_fine():
    subprocess.run(["x"])
"""
    calls = [call for _fn, _line, call in _blocking_calls(ast.parse(source))]
    assert sorted(calls) == sorted(
        ["subprocess.run", "time.sleep", "verify_password", "probe_duration_seconds"]
    )
