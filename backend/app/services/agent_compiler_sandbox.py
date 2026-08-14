"""Compile and run C snippets inside the compiler research sandbox.

The agent needs to see what a compiler actually emitted. Timing alone is not
enough: a loop can be vectorized or if-converted, leaving no branch to
mispredict, and a benchmark measuring "branch prediction" in that state
measures noise.

Runs use the same posture as the experiment runner: no network, all
capabilities dropped, an unprivileged uid, and a per-run directory that is the
only writable path. The image must be on
SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES, and execution is gated by
ENABLE_UNSAFE_CODE_EXECUTION like every other code path that runs submitted
code.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"
MAX_CODE_CHARS = 20000
MAX_OUTPUT_CHARS = 12000
DEFAULT_TIMEOUT_SECONDS = 120

# Counted off the generated assembly to describe what the compiler did.
# aarch64 and x86-64 spellings both appear so the same tool works on either.
CODEGEN_PATTERNS = {
    # The x86 alternative matches a register, which starts with '%', so it
    # cannot sit behind the \b that the mnemonic alternatives need.
    "vector_ops": r"(\b(uaddw|addv|faddp|addp|v\d+\.\d+[bhsd])\b|%[xyz]mm\d+)",
    "conditional_branches": r"\b(b\.(eq|ne|ge|gt|le|lt|hs|lo)|j(e|ne|g|ge|l|le|a|b)\b)",
    "conditional_selects": r"\b(csel|csinc|cinc|cneg|csneg|cmov\w*)\b",
    "calls": r"\b(bl|call)\b",
}


# Flags are interpolated into a shell command, so restrict them to the
# characters real compiler flags use rather than trying to quote arbitrary text.
SAFE_FLAGS = re.compile(r"^[-A-Za-z0-9_=+., /]*$")

EMIT_ALIASES = {
    "asm": "asm",
    "assembly": "asm",
    "s": "asm",
    "ir": "ir",
    "llvm": "ir",
    "llvm-ir": "ir",
    "llvm_ir": "ir",
    "bitcode": "ir",
}


# Compiler complaints an agent cannot act on without knowing this sandbox, and
# the flag that does work here. The remedy matters more than the diagnosis: a
# run that was told only "Compilation failed" re-sent -march=native four times
# and never measured the -O3 codegen it had been asked for.
COMPILER_ERROR_REMEDIES = (
    (
        re.compile(r"does not support '-march=native'"),
        "This sandbox targets aarch64, where clang rejects -march=native. "
        "Use -mcpu=native instead.",
    ),
    (
        re.compile(r"unsupported option '-m(avx\w*|sse\d*)'", re.IGNORECASE),
        "x86 ISA flags do not apply on this aarch64 sandbox. Use -mcpu=native, "
        "or plain -O3, and read the emitted assembly for the vector width.",
    ),
)


def explain_compiler_failure(stderr: str) -> str:
    """Build a failure message the caller can act on.

    The compiler already said what was wrong; repeating "Compilation failed"
    and filing the reason in a separate field means the reason may never reach
    whoever decides the next call.
    """
    first_line = next(
        (line.strip() for line in (stderr or "").splitlines() if line.strip()), ""
    )
    message = "Compilation failed"
    if first_line:
        message += f": {first_line[:400]}"
    for pattern, remedy in COMPILER_ERROR_REMEDIES:
        if pattern.search(stderr or ""):
            return f"{message} — {remedy}"
    return message


MAX_REPORTED_METRICS = 12
MAX_REPORTED_VALUES = 20
_REPORTED_METRIC = re.compile(
    r"^\s*([A-Za-z][\w .%/-]{0,40}?)\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*$"
)


def parse_reported_metrics(output: str) -> Dict[str, List[float]]:
    """Collect the numbers the benchmarked program printed about itself.

    A harness that prints "gflops=1.646" has already done the arithmetic that
    makes its timings meaningful. The finding carried only the elapsed
    milliseconds, so that figure was dropped on the floor and runs concluded
    "no GFLOP/s was reported" about a program that had reported it four times.

    Only plain key=value numbers are taken, and they are carried as printed:
    what a key means is the program's business, not this module's.
    """
    metrics: Dict[str, List[float]] = {}
    for line in (output or "").splitlines():
        match = _REPORTED_METRIC.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        if name not in metrics and len(metrics) >= MAX_REPORTED_METRICS:
            continue
        values = metrics.setdefault(name, [])
        if len(values) < MAX_REPORTED_VALUES:
            values.append(float(match.group(2)))
    return metrics


def _clean_flags(flags: str) -> Optional[str]:
    """Return usable flags, or None if they contain shell metacharacters."""
    candidate = (flags or "").strip()
    return candidate if SAFE_FLAGS.match(candidate) else None


def _allowed_images() -> List[str]:
    from app.core.config import settings

    raw = getattr(settings, "SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES", "") or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _execution_enabled() -> bool:
    from app.core.config import settings

    return bool(getattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))


def _docker_command(
    *, image: str, workdir: str, script: str, timeout_seconds: int
) -> List[str]:
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
        "256",
        "--memory",
        "2048m",
        "--cpus",
        "2",
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


def describe_subject(code: str, label: str = "") -> str:
    """Name what was compiled, for the finding this run will record.

    A measurement that does not say what it measured cannot be compared with
    another. An agent surveying five kernels got back five findings all reading
    "clang -O3: N vector ops", could not map them to its kernels, and spent its
    remaining iterations measuring them again.
    """
    explicit = (label or "").strip()
    if explicit:
        return explicit[:80]
    # Fall back to the function names the snippet defines.
    names = re.findall(
        r"^\s*(?:static\s+|inline\s+)*[A-Za-z_][\w\s\*]*?([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{",
        code or "",
        re.MULTILINE,
    )
    unique = list(dict.fromkeys(names))
    return ", ".join(unique[:3]) if unique else "unnamed snippet"


def count_codegen(assembly: str) -> Dict[str, int]:
    """Summarize what the compiler emitted, so timings can be trusted or not."""
    return {
        name: len(re.findall(pattern, assembly))
        for name, pattern in CODEGEN_PATTERNS.items()
    }


async def _run(script: str, workdir: str, *, image: str, timeout_seconds: int):
    process = await asyncio.create_subprocess_exec(
        *_docker_command(
            image=image,
            workdir=workdir,
            script=script,
            timeout_seconds=timeout_seconds,
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


def _preflight(code: str, image: str) -> Optional[Dict[str, Any]]:
    """Reject a request before spending a container on it."""
    if not code.strip():
        return {"error": "code is required"}
    if len(code) > MAX_CODE_CHARS:
        return {"error": f"code exceeds {MAX_CODE_CHARS} characters"}
    if not _execution_enabled():
        return {
            "error": (
                "Sandboxed execution is disabled on this server "
                "(ENABLE_UNSAFE_CODE_EXECUTION is false)."
            )
        }
    if image not in _allowed_images():
        return {
            "error": (
                f"Image {image} is not allowlisted. Allowed: "
                f"{', '.join(_allowed_images()) or 'none'}"
            )
        }
    return None


async def compile_c_snippet(
    *,
    code: str,
    flags: str = "-O2",
    emit: str = "asm",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile C and return the generated code plus codegen counts."""
    blocked = _preflight(code, image)
    if blocked:
        return blocked

    # Accept the obvious synonyms. A caller asking for "assembly" means "asm",
    # and rejecting it costs an iteration to learn a vocabulary difference.
    emit = EMIT_ALIASES.get((emit or "asm").strip().lower())
    if emit is None:
        return {
            "error": ("emit must be one of: " f"{', '.join(sorted(set(EMIT_ALIASES)))}")
        }
    emit_flag = "-S" if emit == "asm" else "-S -emit-llvm"
    safe_flags = _clean_flags(flags)
    if safe_flags is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}

    with tempfile.TemporaryDirectory(prefix="compile_snippet_") as workdir:
        Path(workdir, "snippet.c").write_text(code, encoding="utf-8")
        script = (
            f"clang {safe_flags} {emit_flag} "
            "-o out.txt snippet.c 2>compile_err.txt; "
            "rc=$?; cat compile_err.txt >&2; "
            "if [ $rc -eq 0 ]; then cat out.txt; fi; exit $rc"
        )
        try:
            returncode, stdout, stderr = await _run(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"error": f"Compilation timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"compile_c_snippet failed: {exc}")
            return {"error": f"Compilation failed: {exc}"}

    if returncode != 0:
        return {
            "success": False,
            "error": explain_compiler_failure(stderr),
            "compiler_stderr": stderr[:MAX_OUTPUT_CHARS],
            "flags": flags,
        }

    codegen = count_codegen(stdout)
    subject = describe_subject(code, label)
    return {
        "success": True,
        "data": {
            "subject": subject,
            "flags": flags,
            "emit": emit,
            "output": stdout[:MAX_OUTPUT_CHARS],
            "truncated": len(stdout) > MAX_OUTPUT_CHARS,
            "codegen": codegen,
            "compiler_warnings": stderr[:2000] or None,
        },
        # The loop harvests "findings"; without one a run that measured
        # something records nothing, and downstream summaries report that the
        # job produced no results.
        "findings": [
            {
                "type": "codegen_measurement",
                # Name the subject first: an unlabelled measurement cannot be
                # compared with the next one.
                "title": (
                    f"{subject} @ clang {flags}: "
                    f"{codegen['vector_ops']} vector ops, "
                    f"{codegen['conditional_branches']} conditional branches"
                ),
                "subject": subject,
                "flags": flags,
                "codegen": codegen,
            }
        ],
    }


async def benchmark_c_snippet(
    *,
    code: str,
    flags: str = "-O2",
    repeat: int = 3,
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile and run a self-contained C program, reporting its fastest trial."""
    blocked = _preflight(code, image)
    if blocked:
        return blocked

    repeat = max(1, min(int(repeat or 3), 10))
    safe_flags = _clean_flags(flags)
    if safe_flags is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}

    with tempfile.TemporaryDirectory(prefix="bench_snippet_") as workdir:
        Path(workdir, "snippet.c").write_text(code, encoding="utf-8")
        # sh needs "{ ...; }" with single braces; "{{" is not grouping and made
        # the exit-90 branch fire even when the compile had succeeded.
        script = (
            f"clang {safe_flags} -o bench snippet.c 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            f"for i in $(seq 1 {repeat}); do "
            "  s=$(date +%s%N); ./bench; rc=$?; e=$(date +%s%N); "
            # Without this the loop's exit status is echo's, so a program that
            # failed would be reported as a successful benchmark.
            '  if [ $rc -ne 0 ]; then echo "program exited $rc" >&2; exit 91; fi; '
            '  echo "__elapsed_ms__ $(( (e - s) / 1000000 ))"; '
            "done"
        )
        try:
            returncode, stdout, stderr = await _run(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"error": f"Benchmark timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"benchmark_c_snippet failed: {exc}")
            return {"error": f"Benchmark failed: {exc}"}

    if returncode == 90:
        return {
            "success": False,
            "error": explain_compiler_failure(stderr),
            "compiler_stderr": stderr[:MAX_OUTPUT_CHARS],
            "flags": flags,
        }
    if returncode != 0:
        return {
            "success": False,
            "error": (
                "The program ran but exited non-zero"
                if returncode == 91
                else f"Benchmark failed with exit code {returncode}"
            ),
            "stdout": stdout[:MAX_OUTPUT_CHARS],
            "stderr": stderr[:MAX_OUTPUT_CHARS],
        }

    timings: List[int] = []
    program_output: List[str] = []
    for line in stdout.splitlines():
        if line.startswith("__elapsed_ms__ "):
            try:
                timings.append(int(line.split()[1]))
            except (IndexError, ValueError):
                continue
        else:
            program_output.append(line)

    reported_metrics = parse_reported_metrics("\n".join(program_output))

    return {
        "success": True,
        "data": {
            "flags": flags,
            "repeat": repeat,
            # The fastest trial is the least contaminated by scheduling noise.
            "fastest_ms": min(timings) if timings else None,
            "all_ms": timings,
            "reported_metrics": reported_metrics,
            "stdout": "\n".join(program_output)[:MAX_OUTPUT_CHARS],
            "note": (
                "Wall-clock only; the sandbox has no performance counters. "
                "Check codegen with compile_c_snippet before attributing a "
                "difference to a microarchitectural effect."
            ),
        },
        "findings": [
            {
                "type": "benchmark_measurement",
                "subject": describe_subject(code, label),
                "title": (
                    f"{describe_subject(code, label)} @ clang {flags}: fastest "
                    f"{min(timings)} ms of {len(timings)} trials"
                    if timings
                    else f"{describe_subject(code, label)} @ clang {flags}: "
                    "ran with no timing recorded"
                ),
                "flags": flags,
                "fastest_ms": min(timings) if timings else None,
                "all_ms": timings,
                "reported_metrics": reported_metrics,
            }
        ],
    }
