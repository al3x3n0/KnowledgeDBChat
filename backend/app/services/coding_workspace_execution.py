"""Run commands on a disposable copy of a coding workspace."""

import asyncio
import re
import shlex
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import settings
from app.core.feature_flags import get_flag
from app.services import agent_sandbox_runtime, sandbox_skill_runtime
from app.utils.tree_copy import without_git_or_links


async def run_workspace_command(
    workspace: Path,
    command: str,
    *,
    timeout: int,
    env: dict[str, Any] | None = None,
) -> tuple[int, str, str]:
    enabled = await get_flag("unsafe_code_execution_enabled")
    if enabled is None:
        enabled = settings.ENABLE_UNSAFE_CODE_EXECUTION
    if not enabled:
        raise ValueError("Coding commands require isolated execution to be enabled")
    image = settings.CODING_WORKSPACE_SANDBOX_IMAGE
    if image not in agent_sandbox_runtime.allowed_images():
        raise ValueError(agent_sandbox_runtime.image_not_allowlisted(image))

    # Let the caller configure the program without inheriting worker variables.
    entries = {
        "HOME": "/tmp",
        "CI": "true",
        "GOCACHE": "/tmp/go-cache",
        "CARGO_HOME": "/tmp/cargo",
        **(env or {}),
    }
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in entries):
        raise ValueError("Invalid environment variable name")
    exports = " ".join(
        f"{name}={shlex.quote(str(value))}" for name, value in entries.items()
    )
    script = f"export {exports}\n{command}"

    root = sandbox_skill_runtime.root_dir()
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="coding-", dir=root) as directory:
        await asyncio.to_thread(
            shutil.copytree,
            workspace,
            directory,
            ignore=without_git_or_links,
            dirs_exist_ok=True,
        )
        code, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
            script,
            directory,
            image=image,
            timeout_seconds=timeout,
        )
    startup_error = agent_sandbox_runtime.explain_sandbox_exit(code, stderr, image)
    if startup_error:
        raise ValueError(startup_error)
    return code, stdout, stderr
