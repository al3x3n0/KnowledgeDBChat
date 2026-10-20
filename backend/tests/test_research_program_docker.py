"""Opt-in BuildKit smoke test using an already built backend image.

TEST_RESEARCH_PROGRAM_DOCKER_IMAGE=knowledge_db_backend:latest pytest \
    tests/test_research_program_docker.py --no-cov

Builds a disposable image with the real program-install instruction and current
core code. It never starts the application stack or uses its entrypoint.
"""

import os
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE_IMAGE = os.environ.get("TEST_RESEARCH_PROGRAM_DOCKER_IMAGE", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not BASE_IMAGE, reason="No backend image selected for Docker smoke test"
    ),
]


def _docker(*args, timeout=180):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(scope="module")
def smoke_image(tmp_path_factory):
    folder = tmp_path_factory.mktemp("program-docker")
    source = folder / "program"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "kdbc-docker-smoke"\nversion = "0.1.0"\n'
        '[project.entry-points."kdbc.research_programs"]\n'
        'docker_smoke = "docker_smoke:PROGRAM"\n'
        '[tool.setuptools]\npy-modules = ["docker_smoke"]\n'
    )
    (source / "docker_smoke.py").write_text(
        textwrap.dedent(
            """
        from app.agent_core.research_programs import ResearchProgram
        from app.agent_core.tool_specs.spec import ToolSpec
        from app.services.agent_tool_providers.base import FunctionToolProvider

        async def measure(params, ctx):
            return {"success": True, "data": {"measured": True}}

        def build(executor):
            return FunctionToolProvider(name="docker_smoke", handlers={"docker_smoke_measure": measure})

        PROGRAM = ResearchProgram(
            name="docker_smoke", version="0.1.0", providers=(build,),
            tool_specs=(ToolSpec(name="docker_smoke_measure", description="Smoke measurement",
                                parameters={"type": "object", "properties": {}},
                                produces=("docker_smoke_measurement",)),),
            sandbox_images=("kdbc-docker-smoke:1",),
            evidence_tools=("docker_smoke_measure",),
        )
    """
        )
    )
    context = folder / "context"
    context.mkdir()
    # Send only source code, never the developer's backend/.env or runtime data.
    for path in (ROOT / "backend/app").rglob("*.py"):
        dest = context / path.relative_to(ROOT / "backend")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    dockerfile = (ROOT / "backend/Dockerfile").read_text()
    start = dockerfile.index("RUN --mount=type=bind,from=programs")
    install = dockerfile[start:].split("\n\n", 1)[0]
    (context / "Dockerfile").write_text(
        "ARG SMOKE_BASE\nFROM scratch AS programs\nFROM ${SMOKE_BASE}\n"
        'ARG RESEARCH_PROGRAMS="/programs"\n'
        f"{install}\nCOPY app /app/app\n"
    )
    tag = f"kdbc-program-smoke:{uuid.uuid4().hex}"
    args = [
        "build",
        "--pull=false",
        "--build-context",
        f"programs={source}",
        "--build-arg",
        f"SMOKE_BASE={BASE_IMAGE}",
        "--tag",
        tag,
        str(context),
    ]
    try:
        built = _docker(*args)
        assert built.returncode == 0, built.stdout + built.stderr
        assert sorted(p.name for p in source.iterdir()) == [
            "docker_smoke.py",
            "pyproject.toml",
        ]
        yield tag, args
    finally:
        _docker("image", "rm", tag, timeout=30)


@pytest.mark.parametrize("disabled", [False, True])
def test_installed_program_contributions_and_execution(smoke_image, disabled):
    tag, _ = smoke_image
    probe = textwrap.dedent(
        """
        import asyncio
        import os
        from app.agent_core import research_programs, tool_specs
        from app.services.agent_tool_providers.base import AgentToolExecutionContext, AgentToolRegistry
        from app.services.agent_evidence_bundle import evidence_tools
        from app.services.agent_sandbox_runtime import allowed_images

        enabled = not bool(os.environ.get("RESEARCH_PROGRAMS_DISABLED"))
        report = research_programs.status()
        assert any(p["name"] == "docker_smoke" for p in report["loaded"]) == enabled, report
        assert ("docker_smoke" in report["disabled"]) != enabled, report
        assert ("docker_smoke_measure" in tool_specs.spec_names()) == enabled
        assert ("docker_smoke_measure" in evidence_tools()) == enabled
        assert ("kdbc-docker-smoke:1" in allowed_images()) == enabled

        async def check():
            registry = AgentToolRegistry(research_programs.build_providers(None))
            ctx = AgentToolExecutionContext(mode="autonomous", db=None, service=None)
            handled, result = await registry.try_execute("docker_smoke_measure", {}, ctx)
            assert handled == enabled
            if enabled:
                assert result == {"success": True, "data": {"measured": True}}, result
        asyncio.run(check())
    """
    )
    result = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        "--env",
        f"RESEARCH_PROGRAMS_DISABLED={'docker_smoke' if disabled else ''}",
        tag,
        "-c",
        probe,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_install_target_fails_the_image_build(smoke_image):
    _, args = smoke_image
    # No usable image is produced; do not replace the successful smoke tag.
    failed_args = args[:]
    tag_index = failed_args.index("--tag")
    del failed_args[tag_index : tag_index + 2]
    failed_args[-1:-1] = ["--build-arg", "RESEARCH_PROGRAMS=/programs/missing"]
    result = _docker(*failed_args)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "/programs/missing" in result.stdout + result.stderr
