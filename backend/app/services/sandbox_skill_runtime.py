"""Running a skill's commands in the sandbox, and deciding what came back.

The agent chooses the command; this decides whether the outcome is evidence.
That split is the point of a skill. A procedure the agent adapts is what makes
the workflow agentic rather than a fixed call, and a result checked against
fields the *skill* declared is what keeps "agentic" from meaning "whatever the
model says happened".

Two things follow from that.

**The result is read from the sandbox, never from the call.** A run does not
report a number; it leaves `result.json` behind, and that file is checked
against the declared fields. A stale one is deleted before every run, so a
result can only be the result of the command that just ran.

**A skill may name a judge.** With `judge_command` set, the file an agent's own
command wrote is discarded and the judge -- authored with the skill, not by the
run -- writes the result instead. Without one, the agent's command is the
author of its own evidence, and the finding says so (`judged_by`), because
those are not equally strong and a reader should be able to tell them apart.

Everything runs under `agent_sandbox_runtime`: no network, no capabilities, an
unprivileged uid, one writable directory. There is no second copy of that
posture here to drift.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from loguru import logger

from app.services import agent_sandbox_runtime, sandbox_skill_manifest

#: Where per-job skill directories live. Under the temp dir on purpose: with
#: the Docker socket mounted, TMPDIR is the one path that exists at the same
#: location inside this container and on the host, and the daemon resolves
#: bind mounts on the host.
ROOT_NAME = "kdbc-skills"

#: A run directory nobody has touched for this long belongs to a job that is
#: over. Pruned opportunistically rather than at job end: a job can die without
#: reaching any finaliser, and its directory would otherwise stay for ever.
STALE_AFTER_SECONDS = 24 * 3600

#: How much output a caller is handed. The tail, because that is where a
#: failing command says why.
OUTPUT_TAIL_CHARS = 6000

#: A control is meant to be the smallest thing that works. A dry run that needs
#: longer than this is not a control, and holding a request open for it is how
#: a proxy timeout becomes the error the author sees.
DRY_RUN_TIMEOUT_SECONDS = 120

# Aliases so tests can enable execution and stand in for the daemon, the same
# seam `agent_compiler_sandbox` exposes.
_execution_enabled = agent_sandbox_runtime.execution_enabled
_run_in_sandbox = agent_sandbox_runtime.run_in_sandbox


@dataclass
class SkillRun:
    """What one invocation did."""

    #: Whether the command executed at all. False means nothing about the
    #: command was tested -- the sandbox was disabled, the image missing, the
    #: daemon unreachable -- and no edit to the command will help.
    ran: bool = False
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    #: The result, only when it satisfied every declared field.
    result: Optional[Dict[str, Any]] = None
    result_problems: List[str] = field(default_factory=list)
    #: "judge_command" or "command": who wrote the result that was accepted.
    judged_by: str = ""
    #: Why it could not run, or why a run that did is not a success.
    error: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.ran and self.returncode == 0 and not self.error


def _tail(text: str) -> str:
    text = text or ""
    if len(text) <= OUTPUT_TAIL_CHARS:
        return text
    return "...[truncated]...\n" + text[-OUTPUT_TAIL_CHARS:]


def _open_up(path: Path) -> None:
    """Let the sandbox's unprivileged uid use a path this process created."""
    try:
        os.chmod(path, 0o777 if path.is_dir() else 0o666)
    except OSError:  # pragma: no cover - best effort on foreign filesystems
        pass


def root_dir() -> Path:
    return Path(tempfile.gettempdir()) / ROOT_NAME


def prune_stale(now: Optional[float] = None) -> int:
    """Remove job directories nothing has touched for a day. Returns the count."""
    base = root_dir()
    if not base.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - STALE_AFTER_SECONDS
    removed = 0
    for entry in base.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


#: The most a stage inherits from the stage before it. A copy, so it costs
#: disk per stage; past this the inheritance is refused and said to be, rather
#: than quietly filling a laptop's disk one stage at a time.
MAX_INHERIT_BYTES = 256 * 1024 * 1024

#: How many file names a run is shown. Enough to see what is there; a build
#: tree's thousands of objects are not something a model should read.
LISTING_LIMIT = 60

_INHERITANCE_NOTE = ".kdbc-inherited"


def _safe_key(job_key: Any) -> str:
    return "".join(c for c in str(job_key) if c.isalnum() or c in "-_")


def _tree_bytes(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _not_inherited(_directory: str, names: List[str]) -> List[str]:
    # The skill's own files are rewritten before every run, a result belongs
    # to the command that produced it, and the note describes the parent.
    return [
        n
        for n in names
        if n
        in (
            sandbox_skill_manifest.SKILL_DIR,
            sandbox_skill_manifest.RESULT_FILE,
            _INHERITANCE_NOTE,
        )
    ]


def _inherit(target: Path, parent: Path) -> str:
    """Copy what the previous stage left, and say what happened."""
    size = _tree_bytes(parent)
    if size > MAX_INHERIT_BYTES:
        return (
            f"The previous stage left {size // (1024 * 1024)} MB, over the "
            f"{MAX_INHERIT_BYTES // (1024 * 1024)} MB a stage may inherit, so "
            "this directory starts empty. Rebuild what you need."
        )
    copied = 0
    for entry in parent.iterdir():
        if _not_inherited(str(parent), [entry.name]):
            continue
        destination = target / entry.name
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, destination, symlinks=True, ignore=_not_inherited)
        else:
            shutil.copy2(entry, destination, follow_symlinks=False)
        copied += 1
    # Copied files belong to this process; the sandbox runs as another uid and
    # must be able to rebuild over them.
    for dirpath, dirs, files in os.walk(target):
        for name in [*dirs, *files]:
            full = Path(dirpath) / name
            if not full.is_symlink():
                _open_up(full)
    if not copied:
        return ""
    return (
        f"Started with a copy of the {copied} top-level item(s) the previous "
        "stage left in its working directory. Changes here do not reach it."
    )


def run_dir(job_key: Any, parent_key: Any = None) -> Path:
    """The directory one job works in, created if need be.

    One per job, shared by every skill that job uses, and persistent across
    its calls: a procedure is several commands -- write, build, run, measure --
    and each needs what the last one left.

    A job with a parent is a later stage of a pipeline, and its directory
    starts as a **copy** of the parent's. Copied rather than shared because
    sibling stages run at the same time: two of them in one directory would
    each rewrite ``skill/`` and delete the other's ``result.json``. A copy
    also leaves the earlier stage's directory as it was, so restarting a later
    stage starts from the same place twice.

    Only ever done once, when the directory is first made. A stage that has
    started working must not have its files replaced underneath it.
    """
    base = root_dir()
    base.mkdir(parents=True, exist_ok=True)
    _open_up(base)
    target = base / (_safe_key(job_key) or "job")
    created = not target.exists()
    target.mkdir(exist_ok=True)
    _open_up(target)
    # Touch it, so a long job's directory is not mistaken for an abandoned one.
    os.utime(target, None)

    parent_name = _safe_key(parent_key) if parent_key else ""
    if created and parent_name:
        parent = base / parent_name
        note = ""
        if parent.is_dir() and parent != target:
            try:
                note = _inherit(target, parent)
            except OSError as exc:
                note = (
                    "The previous stage's files could not be copied "
                    f"({str(exc)[:200]}), so this directory starts empty."
                )
        elif parent != target:
            # Either the previous stage never used a skill, or its directory
            # was pruned while this stage waited. The two look the same from
            # here, and a stage expecting files should know not to.
            note = (
                "The previous stage left no working directory (it used no "
                "skill, or its files were cleaned up after "
                f"{STALE_AFTER_SECONDS // 3600} hours), so this one starts empty."
            )
        if note:
            (target / _INHERITANCE_NOTE).write_text(note, encoding="utf-8")
    return target


def inheritance_note(workdir: Path) -> str:
    """What this directory started from, or "" if it started empty."""
    try:
        return (workdir / _INHERITANCE_NOTE).read_text(encoding="utf-8")
    except OSError:
        return ""


def list_files(workdir: Path) -> Dict[str, Any]:
    """What is in a working directory, as a run should see it.

    A later stage is told files exist only if something tells it; without
    this it rebuilds what it was handed, or -- worse -- assumes a file is
    there because the goal mentioned it.
    """
    names: List[str] = []
    truncated = False
    for dirpath, dirs, files in os.walk(workdir):
        relative = Path(dirpath).relative_to(workdir)
        if relative == Path("."):
            dirs[:] = [d for d in dirs if d != sandbox_skill_manifest.SKILL_DIR]
        dirs.sort()
        for name in sorted(files):
            if name == _INHERITANCE_NOTE:
                continue
            if len(names) >= LISTING_LIMIT:
                truncated = True
                break
            names.append(str(relative / name) if str(relative) != "." else name)
        if truncated:
            break
    return {"files": names, "truncated": truncated}


def _place(base: Path, files: Mapping[str, str]) -> None:
    for relative, content in files.items():
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        for parent in target.relative_to(base).parents:
            if str(parent) != ".":
                _open_up(base / parent)
        target.write_text(content, encoding="utf-8")
        _open_up(target)


def reject_caller_files(files: Any) -> str:
    """Why these files cannot be written into a run directory, or ""."""
    if files is None:
        return ""
    if not isinstance(files, Mapping):
        return (
            "files must be an object mapping a relative path to the file's "
            f"text, got {type(files).__name__}"
        )
    for key, value in files.items():
        path = str(key or "").strip()
        if not sandbox_skill_manifest.PATH_PATTERN.match(path) or ".." in path.split(
            "/"
        ):
            return (
                f"file path {path!r} is not allowed: use a relative path of "
                "letters, digits, dot, underscore and hyphen, with no '..'"
            )
        if path == sandbox_skill_manifest.RESULT_FILE:
            return (
                f"{sandbox_skill_manifest.RESULT_FILE} cannot be supplied as a "
                "file: it has to be produced by the command that ran, not "
                "written by the call"
            )
        if path.split("/")[0] == sandbox_skill_manifest.SKILL_DIR:
            return (
                f"{path!r} is inside {sandbox_skill_manifest.SKILL_DIR}/, which "
                "holds the skill's own files and is rewritten before every "
                "run. Write your files beside it."
            )
        if not isinstance(value, str):
            return f"file {path!r} must be text, got {type(value).__name__}"
        if len(value) > sandbox_skill_manifest.MAX_FILE_CHARS:
            return (
                f"file {path!r} is {len(value)} characters; at most "
                f"{sandbox_skill_manifest.MAX_FILE_CHARS}"
            )
    return ""


def _read_result(manifest: Mapping[str, Any], workdir: Path, run: SkillRun) -> None:
    path = workdir / sandbox_skill_manifest.RESULT_FILE
    if not path.is_file():
        run.result_problems = [
            f"the command finished but left no {sandbox_skill_manifest.RESULT_FILE} "
            "in the working directory, so there is no result to record"
        ]
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        run.result_problems = [
            f"{sandbox_skill_manifest.RESULT_FILE} is not valid JSON: {exc}"
        ]
        return
    problems = sandbox_skill_manifest.result_problems(manifest, payload)
    if problems:
        run.result_problems = problems
        return
    run.result = dict(payload)


async def _sandbox(
    command: str, workdir: Path, *, image: str, timeout: int, run: SkillRun
) -> bool:
    """Run one command. Returns False when the run is over (never ran, timed out)."""
    try:
        returncode, stdout, stderr = await _run_in_sandbox(
            command, str(workdir), image=image, timeout_seconds=timeout
        )
    except asyncio.TimeoutError:
        run.ran = True
        run.timed_out = True
        run.error = (
            f"The command did not finish within {timeout} seconds and its "
            "container was removed."
        )
        return False
    except FileNotFoundError:
        run.error = (
            "The docker client is not installed where this runs, so no sandboxed "
            "command can start. This is a deployment problem and no change to "
            "the command will help."
        )
        return False

    run.returncode = returncode
    run.stdout = _tail(stdout)
    run.stderr = _tail(stderr)
    explained = agent_sandbox_runtime.explain_sandbox_exit(returncode, stderr, image)
    if returncode == agent_sandbox_runtime.DOCKER_COULD_NOT_START:
        # The container never started: nothing about the command was tested.
        run.error = explained
        return False
    run.ran = True
    return True


async def execute(
    manifest: Mapping[str, Any],
    *,
    workdir: Path,
    command: str,
    files: Optional[Mapping[str, str]] = None,
    collect_result: bool = False,
    timeout_seconds: Optional[int] = None,
    image_allowed: bool = True,
) -> SkillRun:
    """Run one command for a skill, and optionally collect its result."""
    run = SkillRun()
    image = str(manifest.get("image") or "")

    if not _execution_enabled():
        run.error = (
            "Sandbox execution is disabled on this server "
            "(ENABLE_UNSAFE_CODE_EXECUTION=false), so no skill can run. This is "
            "a server setting and retrying will fail the same way."
        )
        return run
    if not image_allowed:
        run.error = agent_sandbox_runtime.image_not_allowlisted(image)
        return run

    command = str(command or "").strip()
    if not command:
        run.error = "command is empty: say what to run in the sandbox."
        return run

    ceiling = int(
        manifest.get("timeout_seconds")
        or sandbox_skill_manifest.DEFAULT_TIMEOUT_SECONDS
    )
    timeout = ceiling
    try:
        asked = int(timeout_seconds) if timeout_seconds else 0
    except (TypeError, ValueError):
        # A model passing "60s" should get the skill's own limit, not a crash.
        asked = 0
    if asked > 0:
        timeout = max(sandbox_skill_manifest.MIN_TIMEOUT_SECONDS, min(asked, ceiling))

    # The skill's own files are rewritten every time, so a run cannot leave a
    # modified helper behind for the next call -- including a modified judge.
    skill_dir = workdir / sandbox_skill_manifest.SKILL_DIR
    shutil.rmtree(skill_dir, ignore_errors=True)
    skill_dir.mkdir(parents=True, exist_ok=True)
    _open_up(skill_dir)
    _place(skill_dir, manifest.get("files") or {})
    if files:
        _place(workdir, files)

    result_path = workdir / sandbox_skill_manifest.RESULT_FILE
    result_path.unlink(missing_ok=True)

    if not await _sandbox(command, workdir, image=image, timeout=timeout, run=run):
        return run
    if not collect_result:
        return run

    if run.returncode != 0:
        run.result_problems = [
            f"the command exited {run.returncode}, so no result was collected"
        ]
        return run

    judge = str(manifest.get("judge_command") or "").strip()
    if judge:
        # Whatever the run's own command wrote is not the result. Only the
        # judge's output counts, and it starts from a clean slate.
        result_path.unlink(missing_ok=True)
        judged = SkillRun()
        if not await _sandbox(judge, workdir, image=image, timeout=timeout, run=judged):
            run.result_problems = [
                "the skill's judge could not complete: " + (judged.error or "unknown")
            ]
            return run
        if judged.returncode != 0:
            run.result_problems = [
                f"the skill's judge exited {judged.returncode}: "
                + (judged.stderr or judged.stdout)[-600:].strip()
            ]
            return run
        run.judged_by = "judge_command"
    else:
        run.judged_by = "command"

    _read_result(manifest, workdir, run)
    if run.result is None:
        run.judged_by = ""
    return run


#: Programs worth knowing about before writing a skill for an image. A short
#: list of the things a procedure is likely to call, not an inventory.
PROBE_CANDIDATES = (
    "clang clang++ gcc g++ cc rustc cargo python3 make cmake "
    "opt llc lld llvm-mca llvm-objdump llvm-size llvm-nm llvm-bolt "
    "objdump size nm readelf strip valgrind perf gem5 gem5.opt "
    "git jq bc awk sed"
).split()

_PROBED: Dict[str, List[str]] = {}


async def probe_tools(image: str) -> List[str]:
    """Which of the usual programs this image actually has.

    A drafter told only an image's name guesses its contents, and guesses the
    common case: the first live draft called `gcc` in an image that ships
    clang, was shown `No such file or directory: 'gcc'`, and called `gcc`
    again. Asking the image costs one container start and is remembered for
    the life of the process, since an image's contents do not change under a
    tag while this is running.

    Returns [] when nothing could be asked. Silence is not evidence that an
    image is empty, so a caller must treat [] as "unknown", never as "none".
    """
    if image in _PROBED:
        return list(_PROBED[image])
    if not _execution_enabled():
        return []
    script = "for t in %s; do command -v $t >/dev/null 2>&1 && echo $t; done; true" % (
        " ".join(PROBE_CANDIDATES)
    )
    base = root_dir()
    base.mkdir(parents=True, exist_ok=True)
    _open_up(base)
    workdir = Path(tempfile.mkdtemp(prefix="probe_", dir=str(base)))
    _open_up(workdir)
    try:
        returncode, stdout, _ = await _run_in_sandbox(
            script, str(workdir), image=image, timeout_seconds=60
        )
    except Exception as exc:
        logger.warning(f"Could not probe {image}: {exc}")
        return []
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if returncode != 0:
        return []
    found = [t for t in stdout.split() if t in PROBE_CANDIDATES]
    _PROBED[image] = found
    return list(found)


async def dry_run(
    manifest: Mapping[str, Any], *, image_allowed: bool
) -> Dict[str, Any]:
    """Run a skill's control, and say whether the skill works.

    This is what stands between a skill and being offered to a run. A control
    that passes shows the image is present, the helper files are where the
    procedure says, the judge runs, and the result has the declared shape --
    each of which otherwise fails mid-run in a way that reads as the agent's
    mistake.
    """
    control = manifest.get("control") or {}
    slug = str(manifest.get("id") or "skill")
    base = root_dir()
    base.mkdir(parents=True, exist_ok=True)
    _open_up(base)
    workdir = Path(tempfile.mkdtemp(prefix=f"dryrun_{slug}_", dir=str(base)))
    _open_up(workdir)
    try:
        run = await execute(
            manifest,
            workdir=workdir,
            command=str(control.get("command") or ""),
            files=control.get("files") or None,
            collect_result=True,
            timeout_seconds=DRY_RUN_TIMEOUT_SECONDS,
            image_allowed=image_allowed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Skill dry run for {slug} raised: {exc}")
        run = SkillRun(error=f"The dry run itself failed: {str(exc)[:300]}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    passed = run.ok and run.result is not None
    if passed:
        detail = "The control ran and left a result with every declared field."
    elif run.error:
        detail = run.error
    elif run.returncode not in (0, None):
        detail = (
            f"The control command exited {run.returncode}: "
            + (run.stderr or run.stdout)[-600:].strip()
        )
    else:
        detail = "; ".join(run.result_problems) or "The control produced no result."

    return {
        "ok": passed,
        #: Distinguishes "the skill is wrong" from "nothing could be tested".
        #: Only the first is something its author can fix.
        "ran": run.ran,
        "detail": detail,
        "returncode": run.returncode,
        "stdout": run.stdout[-2000:],
        "stderr": run.stderr[-2000:],
        "result": run.result,
        "judged_by": run.judged_by,
        "content_hash": sandbox_skill_manifest.content_hash(manifest),
    }
