"""Run the Dockerfile's program-install shell with a stand-in package installer.

No Docker daemon or network is needed. A failing pip must fail the image build,
even when cleanup succeeds; otherwise the requested program is silently absent.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _install(tmp_path, targets, pip_exit=0, private=False):
    dockerfile = (ROOT / "backend/Dockerfile").read_text()
    start = dockerfile.index("RUN --mount=type=bind,from=programs")
    instruction = dockerfile[start:].split("\n\n", 1)[0].replace("\\\n", " ")
    command = re.sub(r"\ARUN(?:\s+--mount=\S+)+\s+", "", instruction)
    # The build secret is disposable test data, even if the runner has secrets.
    secret = tmp_path / "github_token"
    if private:
        secret.write_text("test-build-token")
    command = command.replace("/run/secrets/github_token", str(secret))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    trace = tmp_path / "trace"
    for name in ("pip", "rm", "find", "git", "cat"):
        script = bin_dir / name
        script.write_text(
            "#!/bin/sh\n"
            # Record command names only: git arguments may hold credentials.
            f"printf '%s\\n' '{name}' >> \"$BUILD_TRACE\"\n"
            + ("printf '%s' 'test-build-token'\n" if name == "cat" else "")
            + f"exit {pip_exit if name == 'pip' else 0}\n"
        )
        script.chmod(0o755)
    run = subprocess.run(
        ["/bin/sh", "-c", command],
        env={
            **os.environ,
            "PATH": str(bin_dir),
            "BUILD_TRACE": str(trace),
            "RESEARCH_PROGRAMS": targets,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return run, trace.read_text().splitlines() if trace.exists() else []


@pytest.mark.parametrize("private", [False, True])
def test_program_install_failure_fails_the_build_and_cleans_credentials(
    tmp_path, private
):
    run, calls = _install(tmp_path, "/programs", pip_exit=19, private=private)

    assert run.returncode == 19, run.stderr
    assert "pip" in calls
    assert "find" not in calls  # Do not continue after the installer failed.
    assert "rm" in calls


def test_core_only_build_skips_program_installation(tmp_path):
    run, calls = _install(tmp_path, "", pip_exit=19)

    assert run.returncode == 0, run.stderr
    assert "pip" not in calls


def test_successful_install_still_cleans_up(tmp_path):
    run, calls = _install(tmp_path, "/programs", private=True)

    assert run.returncode == 0, run.stderr
    assert "git" in calls
    assert "pip" in calls
    assert "find" in calls
    assert calls[-1] == "rm"


def test_local_checkout_builds_and_its_entry_point_loads(tmp_path):
    """A real source build, then discovery in a fresh interpreter.

    The mount must permit the metadata writes exercised here. This checks the
    declared mount configuration and packaging without needing a Docker daemon.
    """
    dockerfile = (ROOT / "backend/Dockerfile").read_text()
    mount = re.search(
        r"--mount=(type=bind,from=programs,target=/programs[^\s]*)", dockerfile
    )
    assert mount is not None
    options = mount.group(1).split(",")
    assert (
        "rw" in options or "readwrite" in options
    ), "Local package builds need writable sources"

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "kdbc-local-build-probe"\nversion = "0.1.0"\n'
        '[project.entry-points."kdbc.research_programs"]\n'
        'local_build_probe = "build_probe:PROGRAM"\n'
        '[tool.setuptools]\npy-modules = ["build_probe"]\n'
    )
    (source / "build_probe.py").write_text(
        "from app.agent_core.research_programs import ResearchProgram\n"
        "PROGRAM = ResearchProgram(name='local_build_probe', version='0.1.0')\n"
    )
    target = tmp_path / "installed"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-build-isolation",
            "--no-deps",
            "--no-compile",
            "--target",
            str(target),
            str(source),
        ],
        env={
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_CONFIG_FILE": os.devnull,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert list(
        source.glob("*.egg-info")
    ), "The package backend writes into the source mount"

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from app.agent_core.research_programs import status; "
                "report = status(); "
                "assert any(p['name'] == 'local_build_probe' and p['version'] == '0.1.0' "
                "for p in report['loaded']), report"
            ),
        ],
        cwd=ROOT / "backend",
        env={
            **os.environ,
            "PYTHONPATH": f"{target}{os.pathsep}{ROOT / 'backend'}",
            "RESEARCH_PROGRAMS_DISABLED": "",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
