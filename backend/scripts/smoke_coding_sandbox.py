"""Exercise the application's Docker coding sandbox on a synthetic repository."""

import argparse
import asyncio
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch
from uuid import uuid4

PROBE = """
import os
from pathlib import Path

assert os.geteuid() == 65534
assert set(os.listdir("/sys/class/net")) == {"lo"}
assert "KDBC_SMOKE_WORKER_SENTINEL" not in os.environ
assert not Path(".git").exists()
assert not Path("outside_link").exists()
Path("sandbox-only.txt").write_text("disposable")
print("Unprivileged, offline, disposable workspace")
"""

CHECK_COMMAND = """
set -eu
python3 probe.py
pytest -q test_python.py
npm test --silent
go test ./...
cargo test --offline -q
printf 'int main(void) { return 0; }\n' | cc -x c - -o c-smoke
./c-smoke
printf 'C compile and run passed\n'
"""


def populate(workspace: Path) -> None:
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "src").mkdir()
    outside = workspace.parent / "outside.txt"
    outside.write_text("Synthetic file outside the repository", encoding="utf-8")
    (workspace / "outside_link").symlink_to(outside)

    files = {
        "probe.py": PROBE,
        "test_python.py": "def test_example():\n    assert 2 + 2 == 4\n",
        "package.json": '{"scripts":{"test":"node --test smoke.test.cjs"}}\n',
        "smoke.test.cjs": (
            "const test = require('node:test');\n"
            "const assert = require('node:assert/strict');\n"
            "test('example', () => assert.equal(2 + 2, 4));\n"
        ),
        "go.mod": "module smoke.example\n\ngo 1.24\n",
        "smoke_test.go": (
            'package smoke\nimport "testing"\n'
            'func TestExample(t *testing.T) { if 2 + 2 != 4 { t.Fatal("arithmetic") } }\n'
        ),
        "Cargo.toml": '[package]\nname="smoke"\nversion="0.1.0"\nedition="2024"\n',
        "src/lib.rs": "#[test]\nfn example() { assert_eq!(2 + 2, 4); }\n",
    }
    for path, content in files.items():
        (workspace / path).write_text(content, encoding="utf-8")


@contextmanager
def track_container():
    from app.services import agent_sandbox_runtime

    name = "kdbc-coding-smoke-" + uuid4().hex[:12]
    with patch.object(agent_sandbox_runtime, "new_container_name", return_value=name):
        yield
    result = subprocess.run(
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--filter",
            f"name=^/{name}$",
            "--quiet",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=20,
    )
    if result.stdout.strip():
        raise RuntimeError(f"Sandbox left container {name} behind")


async def check(workspace: Path) -> None:
    from app.agent_core import research_programs
    from app.core.cache import cache_service
    from app.services.coding_workspace_execution import run_workspace_command
    from app.services.sandbox_invocation import invocation

    research_programs.initialise(())
    with patch.object(cache_service, "get", new=AsyncMock(return_value=None)):
        with track_container():
            code, stdout, stderr = await run_workspace_command(
                workspace, CHECK_COMMAND, timeout=180
            )
        print(stdout, end="", flush=True)
        print(stderr, end="", file=sys.stderr, flush=True)
        if code:
            raise RuntimeError(f"Sandbox toolchain checks exited with {code}")
        if (workspace / "sandbox-only.txt").exists():
            raise RuntimeError("Sandbox writes reached the source workspace")
        print("Sandbox writes stayed in the disposable copy", flush=True)

        with track_container():
            try:
                await run_workspace_command(workspace, "sleep 30", timeout=3)
            except asyncio.TimeoutError:
                pass
            else:
                raise RuntimeError("Sandbox command did not time out")
        print("Timeout removed its container", flush=True)

        deadline = asyncio.get_running_loop().time() + 2

        async def cancelled() -> bool:
            return asyncio.get_running_loop().time() >= deadline

        with track_container(), invocation("smoke", "cancel", cancelled=cancelled):
            try:
                await run_workspace_command(workspace, "sleep 30", timeout=20)
            except InterruptedError:
                pass
            else:
                raise RuntimeError("Sandbox command did not respond to cancellation")
        print("Cancellation removed its container", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Locally built coding sandbox image"
    )
    image = parser.parse_args().image
    subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        timeout=15,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    config = {
        "DEBUG": "true",
        "ENABLE_UNSAFE_CODE_EXECUTION": "true",
        "SANDBOX_EXECUTION_BACKEND": "docker",
        "SANDBOX_MAX_CONCURRENT_RUNS": "0",
        "CODING_WORKSPACE_SANDBOX_IMAGE": image,
        "SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES": image,
        "KDBC_SMOKE_WORKER_SENTINEL": uuid4().hex,
    }
    directory = Path.cwd()
    with TemporaryDirectory(prefix="kdbc-coding-smoke-") as scratch:
        # Resolve app settings away from the developer's .env file.
        os.chdir(scratch)
        try:
            workspace = Path(scratch) / "repo"
            populate(workspace)
            with patch.dict(os.environ, config):
                asyncio.run(check(workspace))
        finally:
            os.chdir(directory)


if __name__ == "__main__":
    main()
