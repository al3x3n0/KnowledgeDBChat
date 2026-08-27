"""One definition of how submitted code is confined when it runs.

Every sandboxed tool -- compiling a snippet, benchmarking one, elaborating an
architecture description -- runs under the same posture: no network, all
capabilities dropped, an unprivileged uid, bounded memory and pids, and a
per-run directory that is the only writable path. Keeping that in one place is
the point: a second copy is a second thing to weaken by accident, and the
weakening would be invisible in review because each copy still looks careful.

Execution is gated by ENABLE_UNSAFE_CODE_EXECUTION, and the image must be on
SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MEMORY = "2048m"
DEFAULT_CPUS = "2"
DEFAULT_PIDS_LIMIT = "256"

#: How long to wait for the daemon to remove a container we have abandoned.
#: Short on purpose: the caller is already reporting a timeout, and a cleanup
#: that hangs would turn one stuck run into a stuck request.
REMOVE_TIMEOUT_SECONDS = 20


def allowed_images() -> List[str]:
    from app.core.config import settings

    raw = getattr(settings, "SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES", "") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def execution_enabled() -> bool:
    from app.core.config import settings

    return bool(getattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))


def image_not_allowlisted(image: str) -> str:
    """Why this call cannot run, and why retrying it differently will not help.

    The old wording -- "Image X is not allowlisted. Allowed: Y" -- reads as an
    instruction to use Y, and a caller that cannot choose an image reads it as
    one anyway. In a live run the critic advised retrying "with the allowlisted
    image explicitly set", the agent worked out from the tool schema that there
    was no such parameter, tried regardless, failed again, and spent two of its
    five iterations on it. The image is a server setting; say so, so the only
    remaining move is the right one.
    """
    allowed = allowed_images()
    return (
        f"Image {image} is not allowlisted on this server, so this tool cannot "
        "run. The image is chosen by the server, not by the caller: there is no "
        "parameter to change and retrying will fail the same way. "
        + (
            f"Allowlisted images: {', '.join(allowed)}. "
            "Use a tool that runs in one of those, or ask an operator to "
            "allowlist this one."
            if allowed
            else "No images are allowlisted at all; ask an operator."
        )
    )


def docker_command(
    *,
    image: str,
    workdir: str,
    script: str,
    timeout_seconds: int,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
    name: str = "",
) -> List[str]:
    """Build the confined `docker run` invocation for one sandboxed script.

    ``name`` is what makes an abandoned run recoverable. Without it the only
    handle on a container is the client process, and killing that leaves the
    container running: the daemon owns it, not us.
    """
    return [
        "docker",
        "run",
        "--rm",
        *(("--name", name) if name else ()),
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        DEFAULT_PIDS_LIMIT,
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--user",
        "65534:65534",
        "-v",
        f"{workdir}:/work:rw",
        "-w",
        "/work",
        image,
        "/bin/sh",
        "-lc",
        script,
    ]


async def remove_container(name: str) -> bool:
    """Force-remove a container we have stopped waiting for.

    Best effort by construction: the container may already be gone (every run
    carries --rm), the daemon may be busy, and either way the caller is on its
    way to reporting a failure. What it must not do is raise into that path and
    replace a truthful timeout with a confusing cleanup error.
    """
    if not name:
        return False
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "--force",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.wait(), timeout=REMOVE_TIMEOUT_SECONDS)
        return process.returncode == 0
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not remove sandbox container {name}: {exc}")
        return False


async def run_in_sandbox(
    script: str,
    workdir: str,
    *,
    image: str,
    timeout_seconds: int,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> Tuple[int, str, str]:
    """Run one script in the sandbox, returning (returncode, stdout, stderr).

    A run that outlives its timeout is torn down here rather than left to the
    caller. `process.kill()` alone -- what this did before -- kills the `docker
    run` client and not the container behind it, and the container goes on
    holding its --cpus share indefinitely. One orphaned gem5 burned 150% CPU
    for an hour on this machine and corrupted every wall-clock measurement
    taken while it ran, which is the expensive part: the leak is silent, and it
    lands in the numbers rather than in an error.

    Cancellation gets the same treatment as a timeout. A job cancelled mid-run
    abandons its container exactly as thoroughly.
    """
    name = f"kdbc-sandbox-{uuid.uuid4().hex[:16]}"
    process = await asyncio.create_subprocess_exec(
        *docker_command(
            image=image,
            workdir=workdir,
            script=script,
            timeout_seconds=timeout_seconds,
            memory=memory,
            cpus=cpus,
            name=name,
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        process.kill()
        try:
            # Shielded, because the cancellation path is where this matters
            # most and awaiting plainly inside a cancelled task can be
            # interrupted before the removal is even sent. Shield runs it as
            # its own task, so a second cancellation abandons the wait rather
            # than the cleanup.
            await asyncio.shield(remove_container(name))
        except asyncio.CancelledError:
            pass
        raise
    return (
        process.returncode,
        (stdout or b"").decode("utf-8", "replace"),
        (stderr or b"").decode("utf-8", "replace"),
    )
