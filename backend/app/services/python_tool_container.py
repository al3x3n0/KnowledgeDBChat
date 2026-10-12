"""Run a custom Python tool's code in the confined sandbox container.

A `python` tool is code a person or an agent wrote, and it ran in the
server's own process under RestrictedPython. That was hardened once (the
"safe" modules led to `os`), but an in-process sandbox is the list of doors
found so far, and a loop that never ended kept its thread after the timeout
stopped waiting for it.

It now runs where every other piece of submitted code runs: one `docker run`
built by `agent_sandbox_runtime` -- no network, no capabilities, an
unprivileged user, memory, CPU and process limits, and a timeout that removes
the container. Inside that the code is ordinary Python: it may import the
standard library and read and write files, because the boundary is the
container and not a list of names.

**Its working directory is the caller's.** `/work` is the directory a job, a
conversation or a user already has for sandbox skills
(`sandbox_skill_runtime.run_dir`), so a file a skill built is there for a
Python tool to read, and what the tool writes is there for the next call.

The contract with the code is unchanged: it is given `input` (a dict) and
answers by assigning `output` (a dict) or `result`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.services import agent_sandbox_runtime, sandbox_skill_runtime

#: Where one call keeps its own files, inside the working directory.
CALL_DIR = ".pytool"
#: The most an answer may weigh. A tool returns a value, not a dataset; data
#: belongs in a file in the working directory.
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
#: Seconds a tool may ask for.
MIN_TIMEOUT, DEFAULT_TIMEOUT, MAX_TIMEOUT = 1, 30, 300
#: How much of what the code printed is handed back when it fails.
OUTPUT_TAIL_CHARS = 2000

# Runs inside the container. Standard library only, and nothing from this
# codebase: it is the one file the tool's author did not write.
RUNNER = r"""
import json, sys, traceback

base = sys.argv[1]
with open(base + "/input.json", encoding="utf-8") as handle:
    inputs = json.load(handle)
with open(base + "/tool.py", encoding="utf-8") as handle:
    source = handle.read()

scope = {"__name__": "__tool__", "input": inputs, "output": {}, "result": None}
try:
    compiled = compile(source, "<tool>", "exec")
except SyntaxError as exc:
    answer = {"ok": False, "kind": "compile",
              "error": "%s (line %s)" % (exc.msg, exc.lineno)}
else:
    try:
        exec(compiled, scope)
        answer = {"ok": True, "value": scope.get("output") or scope.get("result")}
    except BaseException as exc:  # SystemExit and KeyboardInterrupt included
        frames = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(frames) if f.filename == "<tool>"), None)
        answer = {"ok": False, "kind": "runtime",
                  "error": "%s: %s" % (type(exc).__name__, exc), "line": line}

with open(base + "/output.json", "w", encoding="utf-8") as handle:
    json.dump(answer, handle, default=str)
"""


class PythonToolFailed(Exception):
    """The tool's code did not produce an answer. The message says why."""


class PythonToolUnavailable(PythonToolFailed):
    """The code never ran: the sandbox is off, missing, or could not start."""


def image() -> str:
    from app.core.config import settings

    return str(getattr(settings, "CUSTOM_TOOL_PYTHON_IMAGE", "") or "").strip()


def clamp_timeout(value: Any) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        seconds = DEFAULT_TIMEOUT
    return max(MIN_TIMEOUT, min(seconds, MAX_TIMEOUT))


def _prepare(workdir: Path, code: str, inputs: Dict[str, Any]) -> Path:
    call = workdir / CALL_DIR / uuid.uuid4().hex[:12]
    call.mkdir(parents=True)
    (call / "tool.py").write_text(code, encoding="utf-8")
    (call / "runner.py").write_text(RUNNER, encoding="utf-8")
    (call / "input.json").write_text(json.dumps(inputs, default=str), encoding="utf-8")
    return call


def _collect(call: Path) -> Optional[Dict[str, Any]]:
    target = call / "output.json"
    try:
        if target.stat().st_size > MAX_OUTPUT_BYTES:
            raise PythonToolFailed(
                f"The tool's answer is larger than {MAX_OUTPUT_BYTES // 1024} KB. "
                "Write large data to a file in the working directory and "
                "return its name."
            )
        return json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        return None


def _discard(call: Path) -> None:
    shutil.rmtree(call, ignore_errors=True)
    try:
        call.parent.rmdir()  # only if no other call is using it
    except OSError:
        pass


async def run(
    code: str,
    inputs: Dict[str, Any],
    *,
    workdir: Optional[Path] = None,
    timeout_seconds: Any = DEFAULT_TIMEOUT,
) -> Any:
    """Run `code` with `inputs` and return what it assigned to output/result.

    `workdir` is the caller's working directory; without one the call gets a
    directory of its own, removed afterwards.
    """
    if not str(code or "").strip():
        raise PythonToolFailed("No code provided")
    if not agent_sandbox_runtime.execution_enabled():
        raise PythonToolUnavailable(
            "Sandboxed execution is disabled on this deployment "
            "(ENABLE_UNSAFE_CODE_EXECUTION=false), so a Python tool cannot run. "
            "This is a deployment setting; no change to the code will help."
        )
    tool_image = image()
    if tool_image not in agent_sandbox_runtime.allowed_images():
        raise PythonToolUnavailable(
            f"The Python tool image '{tool_image}' (CUSTOM_TOOL_PYTHON_IMAGE) "
            "is not one of SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES, so a "
            "Python tool cannot run. This is a deployment setting."
        )

    timeout = clamp_timeout(timeout_seconds)
    own_directory = workdir is None
    if own_directory:
        workdir = await asyncio.to_thread(
            sandbox_skill_runtime.run_dir, f"pytool-{uuid.uuid4().hex[:12]}"
        )
    call = await asyncio.to_thread(_prepare, workdir, code, dict(inputs or {}))
    inside = f"/work/{CALL_DIR}/{call.name}"
    try:
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                f"python3 {inside}/runner.py {inside}",
                str(workdir),
                image=tool_image,
                timeout_seconds=timeout,
            )
        except asyncio.TimeoutError:
            raise PythonToolFailed(
                f"Execution timed out after {timeout}s and its container was removed"
            ) from None
        except FileNotFoundError:
            raise PythonToolUnavailable(
                "The docker client is not installed where this runs, so a "
                "Python tool cannot start. This is a deployment problem."
            ) from None

        never_ran = agent_sandbox_runtime.could_not_run(returncode, stderr, tool_image)
        if never_ran:
            raise PythonToolUnavailable(never_ran)

        answer = await asyncio.to_thread(_collect, call)
        if answer is None:
            # The runner did not get as far as writing an answer: killed for
            # memory, or the interpreter itself failed.
            tail = (stderr or stdout or "").strip()[-OUTPUT_TAIL_CHARS:]
            raise PythonToolFailed(
                f"The tool ended without an answer (exit {returncode})"
                + (f": {tail}" if tail else "")
            )
        if not answer.get("ok"):
            if answer.get("kind") == "compile":
                raise PythonToolFailed(
                    f"Code compilation errors: {answer.get('error')}"
                )
            where = f" (line {answer['line']})" if answer.get("line") else ""
            printed = (stdout or "").strip()[-OUTPUT_TAIL_CHARS:]
            raise PythonToolFailed(
                f"{answer.get('error')}{where}"
                + (f"\nPrinted before it failed:\n{printed}" if printed else "")
            )
        return answer.get("value")
    finally:
        await asyncio.to_thread(_discard, call)
        if own_directory:
            await asyncio.to_thread(shutil.rmtree, workdir, True)


__all__ = [
    "CALL_DIR",
    "PythonToolFailed",
    "PythonToolUnavailable",
    "clamp_timeout",
    "run",
]
