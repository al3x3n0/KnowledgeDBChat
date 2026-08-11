"""Tests for the compiler sandbox tools.

The docker-backed paths are exercised through their guards and parsing; the
container itself is not run here.
"""

import pytest

from app.services import agent_compiler_sandbox as sandbox


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
    monkeypatch.setattr(
        sandbox, "_allowed_images", lambda: [sandbox.DEFAULT_IMAGE, "python:3.11-slim"]
    )


def test_flags_allow_ordinary_compiler_options():
    assert sandbox._clean_flags("-O3 -ffast-math -march=armv8-a") is not None
    assert sandbox._clean_flags("") == ""


@pytest.mark.parametrize(
    "flags",
    ["-O2; rm -rf /", "-O2 && curl evil", "-O2 `id`", "-O2 $(whoami)", "-O2 | sh"],
)
def test_flags_reject_shell_metacharacters(flags):
    assert sandbox._clean_flags(flags) is None


@pytest.mark.asyncio
async def test_injection_attempt_is_refused_before_running(enabled):
    result = await sandbox.compile_c_snippet(code="int main(){}", flags="-O2; id")

    assert "unsupported characters" in result["error"]


@pytest.mark.asyncio
async def test_execution_disabled_is_reported_not_silently_skipped(monkeypatch):
    monkeypatch.setattr(sandbox, "_execution_enabled", lambda: False)

    result = await sandbox.compile_c_snippet(code="int main(){}")

    assert "ENABLE_UNSAFE_CODE_EXECUTION" in result["error"]


@pytest.mark.asyncio
async def test_image_outside_the_allowlist_is_refused(enabled):
    result = await sandbox.compile_c_snippet(
        code="int main(){}", image="ghcr.io/somewhere/else:latest"
    )

    assert "not allowlisted" in result["error"]


@pytest.mark.asyncio
async def test_empty_code_is_refused(enabled):
    assert "code is required" in (await sandbox.compile_c_snippet(code="  "))["error"]


@pytest.mark.asyncio
async def test_oversized_code_is_refused(enabled):
    result = await sandbox.compile_c_snippet(code="x" * (sandbox.MAX_CODE_CHARS + 1))

    assert "exceeds" in result["error"]


@pytest.mark.asyncio
async def test_emit_rejects_a_value_it_cannot_map(enabled):
    result = await sandbox.compile_c_snippet(code="int main(){}", emit="binary")

    assert "emit must be one of" in result["error"]


@pytest.mark.parametrize("alias", ["assembly", "ASM", " s ", "llvm-ir", "llvm"])
@pytest.mark.asyncio
async def test_emit_accepts_obvious_synonyms(enabled, monkeypatch, alias):
    """An agent asked for "assembly" and lost an iteration to the rejection."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "  uaddw v0.2d, v0.2d, v2.2s\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(code="int main(){}", emit=alias)

    assert result["success"] is True


def test_codegen_counts_describe_aarch64_output():
    counts = sandbox.count_codegen(
        "  uaddw v0.2d, v0.2d, v2.2s\n  b.ge .LBB0_2\n  csel w8, w9, w10, lt\n  bl foo\n"
    )

    assert counts["vector_ops"] >= 1
    assert counts["conditional_branches"] >= 1
    assert counts["conditional_selects"] >= 1
    assert counts["calls"] >= 1


def test_codegen_counts_describe_x86_output():
    counts = sandbox.count_codegen(
        "  vpaddd %xmm0, %xmm1, %xmm2\n  je .L2\n  call foo\n"
    )

    assert counts["vector_ops"] >= 1
    assert counts["conditional_branches"] >= 1


@pytest.mark.asyncio
async def test_a_program_that_exits_non_zero_is_not_reported_as_success(
    enabled, monkeypatch
):
    """The timing loop ends with echo, so its status would otherwise mask the
    program's own failure."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 91, "", "program exited 3"

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.benchmark_c_snippet(code="int main(){return 3;}")

    assert result["success"] is False
    assert "exited non-zero" in result["error"]


@pytest.mark.asyncio
async def test_timings_are_separated_from_program_output(enabled, monkeypatch):
    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "hello\n__elapsed_ms__ 42\nhello\n__elapsed_ms__ 37\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.benchmark_c_snippet(code="int main(){}", repeat=2)

    assert result["data"]["all_ms"] == [42, 37]
    assert result["data"]["fastest_ms"] == 37
    assert result["data"]["stdout"] == "hello\nhello"


@pytest.mark.asyncio
async def test_compile_failure_returns_the_compiler_error(enabled, monkeypatch):
    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 1, "", "snippet.c:1:5: error: expected identifier"

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(code="int broken(")

    assert result["success"] is False
    assert "expected identifier" in result["compiler_stderr"]


@pytest.mark.asyncio
async def test_compile_reports_a_finding_so_the_run_records_what_it_learned(
    enabled, monkeypatch
):
    """Without a finding the loop harvests nothing, and the job's own summary
    then reports that it produced no results."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "  uaddw v0.2d, v0.2d, v2.2s\n  b.ge .L2\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(code="int main(){}", flags="-O3")

    findings = result["findings"]
    assert findings[0]["type"] == "codegen_measurement"
    assert findings[0]["codegen"]["vector_ops"] >= 1
    assert "-O3" in findings[0]["title"]


@pytest.mark.asyncio
async def test_benchmark_reports_a_finding_too(enabled, monkeypatch):
    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "__elapsed_ms__ 42\n__elapsed_ms__ 37\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.benchmark_c_snippet(code="int main(){}", repeat=2)

    assert result["findings"][0]["fastest_ms"] == 37
