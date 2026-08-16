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
from typing import List, Tuple

DEFAULT_MEMORY = "2048m"
DEFAULT_CPUS = "2"
DEFAULT_PIDS_LIMIT = "256"


def allowed_images() -> List[str]:
    from app.core.config import settings

    raw = getattr(settings, "SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES", "") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def execution_enabled() -> bool:
    from app.core.config import settings

    return bool(getattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))


def docker_command(
    *,
    image: str,
    workdir: str,
    script: str,
    timeout_seconds: int,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> List[str]:
    """Build the confined `docker run` invocation for one sandboxed script."""
    return [
        "docker",
        "run",
        "--rm",
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


async def run_in_sandbox(
    script: str,
    workdir: str,
    *,
    image: str,
    timeout_seconds: int,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> Tuple[int, str, str]:
    """Run one script in the sandbox, returning (returncode, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        *docker_command(
            image=image,
            workdir=workdir,
            script=script,
            timeout_seconds=timeout_seconds,
            memory=memory,
            cpus=cpus,
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        process.kill()
        raise
    return (
        process.returncode,
        (stdout or b"").decode("utf-8", "replace"),
        (stderr or b"").decode("utf-8", "replace"),
    )
