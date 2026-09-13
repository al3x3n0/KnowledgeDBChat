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
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.config import settings
from app.services import agent_measurement_lock, agent_sandbox_runtime, agent_toolchains

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"
MAX_CODE_CHARS = 20000
MAX_OUTPUT_CHARS = 12000
DEFAULT_TIMEOUT_SECONDS = 120

#: Trials per benchmark. Three cannot support a dispersion estimate: one stall
#: in three samples decided a live swarm's verdict. Five lets the slowest be
#: discarded and still leaves four to measure with -- on the run that failed,
#: `[46, 47, 48, 49, 110]` reads as 6% spread where `[61, 109, 46]` read as
#: 137%. Safe to raise only because the trial loop also stops on a time
#: budget, so a slow program takes fewer trials rather than timing out.
DEFAULT_REPEAT = 5

#: The share of the run's timeout the trials may consume. The rest is for
#: compiling and the startup probe. Without this, raising DEFAULT_REPEAT would
#: turn a program that took 30s a trial from a slow benchmark into a failed
#: one.
TRIAL_BUDGET_SHARE = 0.5

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
    # Every one of these was hit by a live run trying to hand-write a timing
    # harness: the model knew the technique and reached for x86 spellings of
    # it. Naming the aarch64 equivalent is the difference between one retry
    # and seven.
    (
        re.compile(r"unknown FP unit"),
        "-mfpmath is an x86 option and has no meaning on aarch64, where "
        "floating point is not optional. Drop it; -O2 or -O3 alone is enough.",
    ),
    (
        re.compile(
            r"couldn't allocate (?:output|input) register for constraint '([xyt])'"
        ),
        'That inline-assembly register constraint is x86. On AArch64 use "w" '
        'for a floating-point or SIMD register, "r" for a general-purpose '
        "one, and name registers as s0/d0/v0.4s rather than xmm0.",
    ),
    (
        re.compile(r"implicitly declaring library function '(sqrtf?|fabsf?|powf?)'"),
        "The maths function has no declaration: add #include <math.h>. If the "
        "link then fails, pass -lm in flags.",
    ),
    (
        re.compile(r"undefined (?:reference|symbol).{0,40}\b(sqrtf?|powf?|logf?)\b"),
        "The maths library is not linked by default here. Add -lm to flags.",
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


# llvm-mca complaints a caller cannot act on without knowing the directive
# syntax. A run guessing "an unknown -mcpu is the usual cause" sent an agent to
# check its cpu name four times while mca had been saying, plainly, that its
# region markers did not match.
MCA_ERROR_REMEDIES = (
    (
        re.compile(r"invalid region end directive|unable to find an active region"),
        "A region marker did not pair up. Every '# LLVM-MCA-BEGIN name' needs a "
        "matching '# LLVM-MCA-END' (named or bare) after it.",
    ),
    (
        re.compile(r"invalid region start directive"),
        "The begin marker is '# LLVM-MCA-BEGIN name' as an assembly comment, "
        "and it must appear in assembly rather than in C.",
    ),
    (
        re.compile(r"unable to get target for"),
        "The target triple was not understood. It belongs in 'target' as "
        "something like aarch64-linux-gnu; the core model goes in 'cpu'.",
    ),
    (
        re.compile(r"unsupported CPU|invalid -mcpu|not a recognized processor"),
        "That core model is unknown to this LLVM. 'llc -march=aarch64 "
        "-mcpu=help' lists them; neoverse-n1 and cortex-a78 are present.",
    ),
)


def explain_mca_failure(stderr: str, returncode: int) -> str:
    """Say what llvm-mca actually complained about, and how to fix it."""
    first_line = next(
        (line.strip() for line in (stderr or "").splitlines() if line.strip()), ""
    )
    message = f"llvm-mca failed with exit code {returncode}"
    if first_line:
        message += f": {first_line[:300]}"
    for pattern, remedy in MCA_ERROR_REMEDIES:
        if pattern.search(stderr or ""):
            return f"{message} — {remedy}"
    return message


# A cycle count belongs to a specific core model, so the model is required
# rather than defaulted: "1801 cycles" with no core named is not a measurement
# anyone can check or reproduce.
SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.+-]{1,64}$")
# The architecture part of a target triple. Checked because a caller reaching
# for "the thing I am analysing" naturally puts a label or a cpu name here.
KNOWN_TARGET_ARCHITECTURES = {
    "aarch64",
    "aarch64_be",
    "arm",
    "armeb",
    "thumb",
    "x86_64",
    "i386",
    "i686",
    "riscv32",
    "riscv64",
    "mips",
    "mips64",
    "mipsel",
    "powerpc",
    "powerpc64",
    "ppc64le",
    "sparc",
    "sparcv9",
    "s390x",
    "wasm32",
    "wasm64",
}
DEFAULT_ANALYSIS_TARGET = "aarch64-linux-gnu"
MAX_MCA_ITERATIONS = 10000

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


# The confinement posture lives in one module so a second copy cannot drift
# into being weaker than this one. These aliases keep the existing names, which
# tests monkeypatch to enable execution without a real Docker daemon.
_allowed_images = agent_sandbox_runtime.allowed_images
_execution_enabled = agent_sandbox_runtime.execution_enabled
_docker_command = agent_sandbox_runtime.docker_command


async def _run(script: str, workdir: str, *, image: str, timeout_seconds: int):
    return await agent_sandbox_runtime.run_in_sandbox(
        script, workdir, image=image, timeout_seconds=timeout_seconds
    )


#: Rust function declarations, which the C pattern above cannot see.
_RUST_FN = re.compile(
    r"^\s*(?:pub\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?:extern\s+\"[^\"]*\"\s+)?fn\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)


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

    # Rust is tried first, and the order is load-bearing. The C pattern below
    # happily matches `fn main() {`, so a Rust snippet resolved to "main" --
    # its harness -- while the function actually being measured went unnamed.
    # A `fn <name>` line start does not occur in C, so trying it first costs
    # the C path nothing.
    #
    # `main` is dropped when anything else is present: in a benchmark it is
    # always the harness, never the subject. The C branch deliberately keeps
    # its existing behaviour instead of gaining the same rule, because the
    # labels it has already produced are recorded on findings that runs have
    # compared against each other, and changing them now would silently split
    # one subject into two.
    rust_names = list(dict.fromkeys(_RUST_FN.findall(code or "")))
    if rust_names:
        interesting = [n for n in rust_names if n != "main"] or rust_names
        return ", ".join(interesting[:3])

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
        from app.services import agent_sandbox_runtime

        return {"error": agent_sandbox_runtime.image_not_allowlisted(image)}
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
    requested_emit = (emit or "asm").strip()
    emit = EMIT_ALIASES.get(requested_emit.lower())
    if emit is None:
        # Say what was rejected, not only what is accepted. Measured live: a
        # run asked for emit='counts' three times running, because counts are
        # exactly what this tool returns -- they come back in every reply, and
        # `emit` chooses the LISTING beside them. An error that lists the legal
        # values without naming the illegal one leaves the caller to guess
        # which of its arguments was wrong.
        return {
            "error": (
                f"emit={requested_emit!r} is not one of: "
                f"{', '.join(sorted(set(EMIT_ALIASES)))}. "
                "Note that the codegen counts (vector ops, branches, selects, "
                "calls) are returned on every call whatever `emit` is -- it "
                "only chooses which listing comes with them."
            )
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
            "image": image,
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


# A machine busier than roughly one runnable task per CPU cannot give a stable
# wall-clock reading. The thresholds are deliberately loose: the point is to
# separate "quiet enough to compare numbers" from "these timings are noise",
# not to grade the host.
LOAD_BUSY = 0.7
LOAD_SATURATED = 1.5

# Spread across trials that a difference has to beat to mean anything. Repeated
# runs of the same benchmark on this host varied by up to 44%, which is larger
# than most effects worth proposing an instruction for.
SPREAD_UNSTABLE = 0.25
#: How much wider the untrimmed spread must be before the run is told a
#: trial stalled. Below this the two figures are the same measurement.
OUTLIER_GAP = 0.25

#: Below this, a wall-clock timing on a contended host is measuring the
#: scheduler. Serialising the measurement, taking five trials and discarding
#: the slowest all failed to stabilise a ~30 ms kernel here: spreads stayed
#: above 100% and two roles that agreed exactly on 23 ms were still reported
#: unresolvable. No statistic fixes a signal beneath the noise floor -- only a
#: bigger workload does, which is the same lesson `startup_share` teaches for
#: interpreted languages.
NOISE_FLOOR_MS = 250


#: Above this share, the reported time is mostly the interpreter booting. A
#: third is enough to distort any comparison between two implementations, long
#: before it makes the number meaningless on its own.
STARTUP_DOMINATES = 0.33


#: Trials dropped before measuring spread, as a fraction of the sample. Only
#: the SLOWEST are dropped, and that asymmetry is the whole point: wall-clock
#: interference can add time to a trial and can never subtract it, so the
#: slowest sample is the most contaminated one by construction and the fastest
#: is the cleanest. A dispersion measure that treats the maximum as signal is
#: measuring the host's worst moment, not the code.
TRIM_FRACTION = 0.25


def robust_spread(timings: List[int]) -> Dict[str, Any]:
    """How much this timing actually moved, with the host's hiccups removed.

    `(max - min) / min` over three trials is one stall away from meaningless.
    Measured, on a swarm role timing a 46 ms kernel: `all_ms=[61, 109, 46]`
    reported 137% spread -- and the middle trial was the slow one, so it was
    not even warm-up. Two trials agreed and one blip decided the number. That
    spread then became the consensus tolerance, pushed the merge past the
    50% ceiling, and summoned a person to look at a machine hiccup.

    So the reported spread is taken over the sample with the slowest quarter
    discarded, and the untrimmed figure is reported beside it. Nothing is
    hidden: a big gap between the two IS the finding that the host stalled,
    and it is surfaced as a warning rather than folded into the number.

    `fastest_ms` is unaffected -- trimming only ever removes slow trials.
    """
    out: Dict[str, Any] = {}
    values = sorted(int(t) for t in (timings or []) if t is not None)
    if not values or values[0] <= 0:
        return out
    if len(values) == 1:
        # One trial is a number, not a measurement.
        out["single_trial"] = True
        return out

    raw = (values[-1] - values[0]) / values[0]
    out["trial_spread_raw"] = round(raw, 3)

    discard = max(1, int(len(values) * TRIM_FRACTION)) if len(values) >= 3 else 0
    kept = values[: len(values) - discard] if discard else values
    spread = (kept[-1] - kept[0]) / kept[0] if kept[0] > 0 and len(kept) > 1 else 0.0

    out["trial_spread"] = round(spread, 3)
    out["trials"] = len(values)
    if discard:
        out["trials_discarded"] = discard
    if len(values) < 4:
        # Two or three trials cannot support a dispersion estimate worth much,
        # trimmed or not. Said out loud so a tight spread over three samples is
        # not read as precision it has not earned.
        out["few_trials"] = True
    return out


def measurement_quality(
    load_average: Optional[float],
    cpu_count: Optional[int],
    timings: Optional[List[int]] = None,
    startup_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Describe how trustworthy a wall-clock timing taken just now is.

    Three independent signals, because each catches what the others miss: the
    machine's load says whether something else was competing, the spread across
    trials says whether the result actually held still, and -- for an
    interpreted language -- the interpreter's own startup says how much of the
    number was paid before the algorithm began. A quiet machine can still
    produce unstable timings, a busy one can occasionally produce tight ones by
    luck, and a perfectly stable timing on an idle host can still be almost
    entirely CPython starting up.
    """
    quality: Dict[str, Any] = {}
    if load_average is not None and cpu_count:
        per_cpu = load_average / max(1, cpu_count)
        quality["load_average"] = round(load_average, 2)
        quality["cpu_count"] = cpu_count
        quality["load_per_cpu"] = round(per_cpu, 2)
        if per_cpu >= LOAD_SATURATED:
            quality["measurement_environment"] = "saturated"
        elif per_cpu >= LOAD_BUSY:
            quality["measurement_environment"] = "busy"
        else:
            quality["measurement_environment"] = "quiet"

    if timings:
        # Spread with the host's worst moments trimmed off; see robust_spread.
        # `single_trial` still comes from here -- a live run benchmarked once,
        # was refused by a contract requiring error bars, and had nothing in
        # the tool's own output to tell it why.
        quality.update(robust_spread(list(timings)))

    # What the process paid before reaching the first line of the algorithm.
    # Measured for `python3 -c pass` in the same container, around the same
    # trials, for the same reason the load average is: a floor read from
    # somewhere else is not this run's floor.
    if startup_ms is not None and startup_ms >= 0:
        quality["interpreter_startup_ms"] = startup_ms
        if timings:
            fastest = min(timings)
            quality["startup_share"] = (
                round(min(1.0, startup_ms / fastest), 3) if fastest > 0 else 1.0
            )

    environment = quality.get("measurement_environment")
    spread = quality.get("trial_spread")
    warnings = []
    if environment == "saturated":
        warnings.append(
            f"The host was saturated during this measurement "
            f"({quality['load_per_cpu']} runnable tasks per CPU): the timing "
            "reflects competition for the machine as much as the code."
        )
    elif environment == "busy":
        warnings.append(
            f"The host was busy during this measurement "
            f"({quality['load_per_cpu']} runnable tasks per CPU); treat small "
            "differences as noise."
        )
    if quality.get("single_trial"):
        warnings.append(
            "Only one trial was run, so this reports no spread and cannot show "
            "whether the number is stable. Pass repeat=5 or more; on this host "
            "repeated runs of the same benchmark have varied by up to 44%."
        )
    # A big gap between the trimmed and untrimmed spread means one trial
    # stalled. That is worth saying rather than hiding: the number is usable,
    # and the host is not quiet. Surfaced as its own warning so a reader is
    # never told a timing held still when one trial did not.
    raw_spread = quality.get("trial_spread_raw")
    if (
        spread is not None
        and raw_spread is not None
        and raw_spread >= spread + OUTLIER_GAP
    ):
        dropped = int(quality.get("trials_discarded", 0) or 0)
        warnings.append(
            f"{dropped} of {int(quality.get('trials', 0) or 0)} trials stalled "
            f"(all trials spanned {raw_spread * 100:.0f}%, the rest "
            f"{spread * 100:.0f}%); the reported spread excludes the slowest, "
            "which interference can only ever have made slower."
        )
    if quality.get("few_trials") and not quality.get("single_trial"):
        warnings.append(
            "Too few trials for the spread to mean much. Pass repeat=5 or more "
            "before treating this spread as the measurement's precision."
        )
    fastest = min(timings) if timings else None
    if (
        fastest is not None
        and fastest < NOISE_FLOOR_MS
        and spread is not None
        and spread >= SPREAD_UNSTABLE
    ):
        warnings.append(
            f"At {fastest} ms this workload is too small to time reliably on "
            f"this host: the trials still varied by {spread * 100:.0f}% after "
            "discarding the slowest. Scale the work up until the fastest trial "
            f"clears ~{NOISE_FLOOR_MS} ms; more trials cannot recover a signal "
            "smaller than the scheduling noise around it."
        )
    if spread is not None and spread >= SPREAD_UNSTABLE:
        warnings.append(
            f"Trials varied by {spread * 100:.0f}%, so any difference smaller "
            "than that is not evidence. Repeat on a quiet machine before "
            "concluding anything from it."
        )
    share = quality.get("startup_share")
    if share is not None and share >= STARTUP_DOMINATES:
        warnings.append(
            f"{share * 100:.0f}% of this timing is the interpreter starting up "
            f"({quality['interpreter_startup_ms']} ms of "
            f"{min(timings or [0])} ms), not the algorithm. Give the program "
            "enough work that the run clears that floor, or the number "
            "compares startup costs rather than implementations."
        )
    if warnings:
        quality["measurement_warning"] = " ".join(warnings)
    return quality


async def benchmark_c_snippet(
    *,
    code: str,
    flags: str = "",
    repeat: int = DEFAULT_REPEAT,
    label: str = "",
    language: str = agent_toolchains.DEFAULT_LANGUAGE,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile and run a self-contained program, reporting its fastest trial.

    The build recipe comes from `agent_toolchains` rather than being spelled
    out here, because `check_implementation` has to compile the same source the
    same way. If the two diverged -- a different optimisation level, say -- the
    binary that was verified would not be the binary that was timed, and the
    correctness check would be certifying a different program.
    """
    chain = agent_toolchains.resolve(language)
    if chain is None:
        return {"error": agent_toolchains.unsupported_language(language)}

    blocked = _preflight(code, image)
    if blocked:
        return blocked

    repeat = max(1, min(int(repeat or DEFAULT_REPEAT), 10))
    flags = flags or chain.default_flags
    safe_flags = _clean_flags(flags)
    if safe_flags is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    # After sanitising, never before: what enforce() adds is ours and known
    # safe, and running it through the caller's filter could only reject it.
    safe_flags = agent_toolchains.enforce(chain, safe_flags)

    with tempfile.TemporaryDirectory(prefix="bench_snippet_") as workdir:
        Path(workdir, chain.source_file).write_text(code, encoding="utf-8")
        build = agent_toolchains.build_script(chain, safe_flags)
        # sh needs "{ ...; }" with single braces; "{{" is not grouping and made
        # the exit-90 branch fire even when the compile had succeeded.
        script = (
            f"{build} 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            # `repeat` is a maximum, not a promise. The loop also stops once
            # the trials have eaten their share of the run's timeout, so a
            # program that takes 30s a trial runs two of them instead of
            # turning the whole measurement into a timeout with no result at
            # all. A partial sample is still a measurement; a timeout is not.
            f"__t0=$(date +%s%N); __budget={int(timeout_seconds * TRIAL_BUDGET_SHARE * 1000000000)}; "
            # One untimed run before the clock starts. Standard hygiene on
            # real hardware, where a cold page cache and an unwarmed CPU make
            # the first trial the slowest.
            #
            # It is NOT what was wrong here, and the reported warmup_ms is how
            # that was settled rather than assumed: on this host the warm-up
            # run came back at 30 ms against a 29 ms fastest trial, and the
            # timed trials then ASCENDED -- [29, 36, 40, 61, 78]. A descending
            # sequence in an earlier run looked like warm-up and was
            # coincidence. Kept because it costs one run in six and makes the
            # question checkable; not kept as a fix for anything measured.
            "__warm=$(date +%s%N); ./prog >/dev/null 2>&1; "
            "__warm_end=$(date +%s%N); "
            'echo "__warmup_ms__ $(( (__warm_end - __warm) / 1000000 ))"; '
            f"for i in $(seq 1 {repeat}); do "
            "  s=$(date +%s%N); ./prog; rc=$?; e=$(date +%s%N); "
            # Without this the loop's exit status is echo's, so a program that
            # failed would be reported as a successful benchmark.
            '  if [ $rc -ne 0 ]; then echo "program exited $rc" >&2; exit 91; fi; '
            '  echo "__elapsed_ms__ $(( (e - s) / 1000000 ))"; '
            "  if [ $(( e - __t0 )) -ge $__budget ]; then break; fi; "
            "done; "
            # Sampled in the same container, around the same trials: a timing
            # taken while the machine is busy is not a property of the code,
            # and nothing else in this pipeline would ever notice. Measured
            # here: competing work pushed identical runs from 10s to over
            # 150s, and an orphaned container raised load for an hour.
            # `unknown`, not 0 and not 1. A fallback that yields a NUMBER
            # turns "could not read the load" into "the load was zero", and
            # zero classifies as quiet -- so an unreadable probe would bless a
            # timing taken on a saturated host as taken on an idle one, which
            # is the one thing this sampling exists to prevent. A non-numeric
            # value fails the parse, leaves load_average unset, and the
            # environment is simply not claimed.
            'echo "__loadavg__ $(cut -d" " -f1 /proc/loadavg 2>/dev/null'
            ' || echo unknown)"; '
            'echo "__cpus__ $(nproc 2>/dev/null || echo unknown)"'
        )
        # For an interpreted language, what an EMPTY program costs -- taken in
        # the same container, right after the trials, exactly as the load
        # average is. Timing `./prog` times the whole process, so a Python
        # result carries a floor it did not earn: measured here, a 22 ms run
        # was 9 ms of CPython starting up. Without this the number reads as the
        # algorithm's cost and two implementations get compared on their
        # shared startup.
        if chain.startup_probe:
            script += (
                "; s=$(date +%s%N); " + chain.startup_probe + "; "
                "e=$(date +%s%N); "
                'echo "__startup_ms__ $(( (e - s) / 1000000 ))"'
            )
        # One timing at a time on this host. Two roles of the same swarm
        # started benchmarking in the same second and timed each other's
        # contention; see agent_measurement_lock for that measurement.
        try:
            async with agent_measurement_lock.exclusive_measurement(
                wait_seconds=float(
                    getattr(settings, "AGENT_MEASUREMENT_LOCK_WAIT_SECONDS", 180)
                ),
                ttl_seconds=agent_measurement_lock.ttl_for(timeout_seconds),
                enabled=bool(getattr(settings, "AGENT_MEASUREMENT_LOCK_ENABLED", True)),
            ) as lock_outcome:
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
    load_average: Optional[float] = None
    cpu_count: Optional[int] = None
    startup_ms: Optional[int] = None
    warmup_ms: Optional[int] = None
    for line in stdout.splitlines():
        if line.startswith("__elapsed_ms__ "):
            try:
                timings.append(int(line.split()[1]))
            except (IndexError, ValueError):
                continue
        elif line.startswith("__loadavg__ "):
            try:
                load_average = float(line.split()[1])
            except (IndexError, ValueError):
                continue
        elif line.startswith("__cpus__ "):
            try:
                cpu_count = int(line.split()[1])
            except (IndexError, ValueError):
                continue
        elif line.startswith("__startup_ms__ "):
            try:
                startup_ms = int(line.split()[1])
            except (IndexError, ValueError):
                continue
        elif line.startswith("__warmup_ms__ "):
            try:
                warmup_ms = int(line.split()[1])
            except (IndexError, ValueError):
                continue
        else:
            program_output.append(line)

    quality = measurement_quality(load_average, cpu_count, timings, startup_ms)
    # The discarded first run, reported rather than dropped silently. How much
    # slower it was than the fastest trial is the size of the warm-up effect on
    # this host, which is the evidence that discarding it was right -- and, if
    # the two are close, the evidence that it was not needed.
    if warmup_ms is not None and warmup_ms >= 0:
        quality["warmup_ms"] = warmup_ms
        quality["warmup_discarded"] = True
    # Whether this timing had the machine to itself. Carried on the result so
    # a reader -- and the swarm's consensus, which decides what "agree" means
    # from exactly these fields -- can tell a serialised measurement from one
    # that may have shared the CPU with another role's benchmark.
    quality.update(lock_outcome.as_quality())
    if not lock_outcome.held and lock_outcome.detail:
        quality["measurement_warning"] = " ".join(
            part
            for part in (quality.get("measurement_warning"), lock_outcome.detail)
            if part
        )

    reported_metrics = parse_reported_metrics("\n".join(program_output))

    return {
        "success": True,
        "data": {
            "image": image,
            "language": chain.language,
            "flags": flags,
            "repeat": repeat,
            # The fastest trial is the least contaminated by scheduling noise.
            "fastest_ms": min(timings) if timings else None,
            "all_ms": timings,
            "reported_metrics": reported_metrics,
            "stdout": "\n".join(program_output)[:MAX_OUTPUT_CHARS],
            **quality,
            "note": (
                "Wall-clock only; the sandbox has no performance counters. "
                "Check codegen with compile_c_snippet before attributing a "
                "difference to a microarchitectural effect."
                + (
                    f" {quality['measurement_warning']}"
                    if quality.get("measurement_warning")
                    else ""
                )
            ),
        },
        "findings": [
            {
                "type": "benchmark_measurement",
                "subject": describe_subject(code, label),
                "title": (
                    f"{describe_subject(code, label)} @ {chain.language} "
                    f"{flags}: fastest {min(timings)} ms of {len(timings)} "
                    "trials"
                    if timings
                    else f"{describe_subject(code, label)} @ {chain.language} "
                    f"{flags}: ran with no timing recorded"
                ),
                # On the finding because a timing is only comparable with
                # another taken the same way, and two languages are not the
                # same way.
                "language": chain.language,
                "flags": flags,
                "fastest_ms": min(timings) if timings else None,
                "all_ms": timings,
                "reported_metrics": reported_metrics,
                # Carried on the finding, not just the tool result, so a goal
                # contract can bound it (validity.bounds on load_per_cpu or
                # trial_spread) and refuse a run whose numbers were taken on a
                # machine too busy to measure anything.
                **quality,
            }
        ],
    }


def _mca_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the per-region summary out of llvm-mca's JSON report."""
    regions = payload.get("CodeRegions")
    region = regions[0] if isinstance(regions, list) and regions else {}
    summary = region.get("SummaryView") if isinstance(region, dict) else {}
    return summary if isinstance(summary, dict) else {}


async def analyze_snippet_cycles(
    *,
    code: str = "",
    asm: str = "",
    cpu: str = "",
    flags: str = "-O3",
    target: str = DEFAULT_ANALYSIS_TARGET,
    iterations: int = 100,
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Model how a code sequence issues on a named core, without running it.

    This is how a proposed instruction can be evidenced at all: the hardware
    does not exist, so it cannot be benchmarked, but the sequence it would
    replace can be costed against a published scheduling model, and so can the
    sequence that replaces it. Wall clock cannot do that, and on this sandbox
    it cannot do much anyway -- the microarchitecture image says plainly that
    PMU access needs privileges the sandbox drops.

    Pass ``code`` to compile and analyse, or ``asm`` to analyse a sequence
    directly -- the second is what a hypothetical costing needs, since the
    instruction being proposed cannot be produced by any compiler here.
    """
    source = asm or code
    blocked = _preflight(source, image)
    if blocked:
        return blocked

    # Region markers are assembly comments. In C they are preprocessor
    # directives and the compile dies on "invalid preprocessing directive",
    # which says nothing about what to do instead. A caller asked to fence a
    # loop reaches for them naturally, so catch it here rather than in clang.
    if not asm and "LLVM-MCA-" in code:
        return {
            "error": (
                "LLVM-MCA region markers are assembly comments and cannot appear "
                "in C: pass the fenced assembly as 'asm' instead. Compile first "
                "with compile_c_snippet, add '# LLVM-MCA-BEGIN name' and "
                "'# LLVM-MCA-END' around the loop in the output, then analyse "
                "that."
            )
        }
    if asm and code:
        # Both were supplied and only one is used; saying which prevents a
        # caller reading a number as being about the other.
        logger.info("analyze_snippet_cycles: asm given alongside code; using asm")

    cpu = str(cpu or "").strip()
    if not cpu:
        return {
            "error": (
                "cpu is required: a cycle count is a property of a specific core "
                "model. Pass one llvm-mca knows, e.g. neoverse-n1 or cortex-a78."
            )
        }
    target = str(target or DEFAULT_ANALYSIS_TARGET).strip()
    for name, value in (("cpu", cpu), ("target", target)):
        if not SAFE_MODEL_NAME.match(value):
            return {"error": f"{name} contains unsupported characters: {value!r}"}
    # A label or a core name in `target` reaches llvm-mca as a triple and comes
    # back as "unable to get target for 'norm'", which names neither the
    # parameter at fault nor what belongs in it.
    if not target.split("-")[0].lower() in KNOWN_TARGET_ARCHITECTURES:
        return {
            "error": (
                f"target should be a target triple such as "
                f"{DEFAULT_ANALYSIS_TARGET}, not {target!r}. The core model "
                f"goes in 'cpu' (you passed cpu={cpu!r}), and a name for the "
                "run goes in 'label'."
            )
        }

    safe_flags = _clean_flags(flags)
    if safe_flags is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    try:
        iteration_count = max(1, min(int(iterations), MAX_MCA_ITERATIONS))
    except (TypeError, ValueError):
        iteration_count = 100

    with tempfile.TemporaryDirectory(prefix="analyze_snippet_") as workdir:
        if asm:
            # llvm-mca's directive parser does not terminate the region name at
            # end of file, so assembly whose last line is "# LLVM-MCA-END loop"
            # with no newline is read as region "loo" and rejected. That cost a
            # live run four calls chasing an error about markers that were
            # correct. Nobody should have to know this: end the file properly.
            Path(workdir, "snippet.s").write_text(
                asm if asm.endswith("\n") else asm + "\n", encoding="utf-8"
            )
            compile_step = ""
        else:
            Path(workdir, "snippet.c").write_text(code, encoding="utf-8")
            compile_step = (
                f"clang --target={target} {safe_flags} -S -o snippet.s snippet.c "
                "2>compile_err.txt || "
                "{ cat compile_err.txt >&2; exit 90; }; "
            )
        script = (
            compile_step + f"llvm-mca -mtriple={target} -mcpu={cpu} "
            f"-iterations={iteration_count} -json snippet.s 2>mca_err.txt; "
            "rc=$?; cat mca_err.txt >&2; exit $rc"
        )
        try:
            returncode, stdout, stderr = await _run(
                script, workdir, image=image, timeout_seconds=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"error": f"Analysis timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"analyze_snippet_cycles failed: {exc}")
            return {"error": f"Analysis failed: {exc}"}

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
            "error": explain_mca_failure(stderr, returncode),
            "stderr": stderr[:MAX_OUTPUT_CHARS],
            "cpu": cpu,
        }

    try:
        summary = _mca_summary(json.loads(stdout))
    except (ValueError, TypeError) as exc:
        return {
            "success": False,
            "error": f"Could not read the llvm-mca report: {exc}",
            "stdout": stdout[:MAX_OUTPUT_CHARS],
        }
    if not summary:
        return {
            "success": False,
            "error": "llvm-mca reported no code region to analyse",
            "stdout": stdout[:MAX_OUTPUT_CHARS],
        }

    total_cycles = summary.get("TotalCycles")
    reported_iterations = summary.get("Iterations") or iteration_count
    cycles_per_iteration = (
        round(float(total_cycles) / float(reported_iterations), 3)
        if isinstance(total_cycles, (int, float)) and reported_iterations
        else None
    )
    # How many instructions the fenced region actually contains. A weak check
    # rather than a strong one: a run that hand-wrote its assembly instead of
    # analysing the compiler's output was 40% off in cycles while differing by
    # a single instruction, so this catches a wildly wrong region and not a
    # subtly wrong one.
    instructions_value = summary.get("Instructions")
    instructions_per_iteration = (
        round(float(instructions_value) / float(reported_iterations), 3)
        if isinstance(instructions_value, (int, float)) and reported_iterations
        else None
    )
    # mca's own warnings change what the number means -- a region that swept up
    # a return or the function prologue is not the loop the caller asked about.
    warnings = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    analysed = asm or ""
    if "LLVM-MCA-BEGIN" not in analysed and any(
        "return instruction" in line for line in warnings
    ):
        # Costing a whole function reads as costing its loop, and the two differ
        # by a lot: the same saxpy came out at 24.14 cycles as a function and
        # 7.18 as its inner loop, because the prologue and scalar tail were
        # being averaged in.
        warnings.append(
            "This estimate covers the whole sequence including prologue and "
            "return, not a loop. Fence the region of interest with "
            "'# LLVM-MCA-BEGIN name' and '# LLVM-MCA-END' comments in the "
            "assembly and analyse that instead."
        )
    subject = describe_subject(code or asm, label)

    return {
        "success": True,
        "data": {
            "image": image,
            "cpu": cpu,
            "target": target,
            "flags": flags if not asm else "",
            "source": "asm" if asm else "c",
            "iterations": reported_iterations,
            "total_cycles": total_cycles,
            "cycles_per_iteration": cycles_per_iteration,
            "instructions": summary.get("Instructions"),
            "instructions_per_iteration": instructions_per_iteration,
            "total_uops": summary.get("TotaluOps"),
            "ipc": summary.get("IPC"),
            "uops_per_cycle": summary.get("uOpsPerCycle"),
            "dispatch_width": summary.get("DispatchWidth"),
            "block_rthroughput": summary.get("BlockRThroughput"),
            "warnings": warnings[:10],
            "note": (
                "Modelled, not executed: these are llvm-mca's estimates for "
                f"{cpu}, and they assume the whole region issues from a warm "
                "front end with no cache misses."
            ),
        },
        "findings": [
            {
                "type": "cycle_model_measurement",
                "subject": subject,
                # The core model belongs in the title: a cycle count quoted
                # without it cannot be compared with anything.
                "title": (
                    f"{subject} @ {cpu}"
                    + (f" (clang {flags})" if not asm else " (given assembly)")
                    + f": {cycles_per_iteration} cycles/iteration, "
                    f"IPC {round(float(summary.get('IPC') or 0), 3)}"
                ),
                "cpu": cpu,
                "target": target,
                "flags": flags if not asm else "",
                "cycles_per_iteration": cycles_per_iteration,
                "instructions_per_iteration": instructions_per_iteration,
                "total_cycles": total_cycles,
                "instructions": summary.get("Instructions"),
                "block_rthroughput": summary.get("BlockRThroughput"),
                "warnings": warnings[:5],
            }
        ],
    }


# A fused instruction still has to do the work of the operations it subsumes,
# so it cannot be cheaper than the slowest of them; and if it were as expensive
# as the whole sequence there would be no reason to build it. Those two facts
# bound the saving without anyone having to invent a cost for an instruction
# that does not exist yet.
FUSION_STAND_IN_NOTE = (
    "The saving is reported as a range because llvm-mca cannot cost an "
    "instruction that does not exist. The floor is the slowest operation the "
    "fused form still has to perform, which it cannot beat; the ceiling is the "
    "sequence it replaces, which it must beat to be worth building. A single "
    "number here would be the cost of whichever stand-in instruction was "
    "picked, and picking a slow one manufactures a negative result."
)


def fusion_saving_bounds(
    sequence_cycles: float, constituent_cycles: Dict[str, float]
) -> Dict[str, Any]:
    """Bound what fusing a sequence could save, per occurrence.

    Returns the best case (the fused instruction costs as much as the slowest
    operation it subsumes) and the worst (it costs what the sequence already
    costs, so nothing is saved). A candidate whose best case is zero cannot
    pay however it is implemented, which is a verdict this can reach without
    modelling the instruction at all.
    """
    costs = {
        k: float(v) for k, v in (constituent_cycles or {}).items() if v is not None
    }
    if not costs or sequence_cycles is None:
        return {"error": "no constituent costs to bound the saving with"}

    floor = max(costs.values())
    best = max(0.0, float(sequence_cycles) - floor)
    return {
        "sequence_cycles_per_occurrence": round(float(sequence_cycles), 3),
        "slowest_constituent": max(costs, key=lambda k: costs[k]),
        "slowest_constituent_cycles": round(floor, 3),
        "max_saving_per_occurrence": round(best, 3),
        "min_saving_per_occurrence": 0.0,
        "worth_pursuing": best > 0.0,
        "constituent_cycles": {k: round(v, 3) for k, v in sorted(costs.items())},
        "note": FUSION_STAND_IN_NOTE,
    }


SAFE_PATTERN = re.compile(r"^[a-z0-9_. |>,]+$")


async def cost_fusion_candidate(
    *,
    pattern: str,
    cpu: str = "",
    copies: int = 20,
    mode: str = "dependent",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Cost a mined fusion candidate, and bound what fusing it could save.

    Costs the sequence as it stands, then each operation it contains on its
    own, and reports the saving as a range. The alternative -- naming a real
    instruction to stand for the fused one and reporting the difference -- puts
    the answer at the mercy of that choice: two live runs concluded a candidate
    was a regression on exactly that basis, and neither conclusion could be
    checked.

    The assembly is generated by candidate-coster inside the image, so the
    sequence being costed here is built by the same code that builds it for a
    human, rather than by a second implementation that can drift from it.
    """
    spec = str(pattern or "").strip().lower()
    if not spec:
        return {
            "error": (
                "pattern is required: the shape to cost, as find_fusion_candidates "
                "spells it, e.g. 'fsqrt fdiv | 0>1'."
            )
        }
    if not SAFE_PATTERN.match(spec):
        # Raw assembly is the mistake worth naming: a run with no method
        # reached for this tool correctly and then fed it the instructions the
        # candidate was found in, three times, because "unsupported
        # characters" describes the string rather than what belongs there.
        looks_like_assembly = any(
            marker in spec for marker in (",", "[", "\n", ";")
        ) and any(register in spec for register in (" s", " d", " v", " x", " w"))
        if looks_like_assembly:
            return {
                "error": (
                    "This looks like assembly. What belongs here is the shape "
                    "of a candidate as find_fusion_candidates spells it -- its "
                    "`pattern` field, such as 'fsqrt fdiv | 0>1': mnemonics, "
                    "then which result feeds which. Registers and operands are "
                    "chosen when the sequence is rendered, and passing the "
                    "ones it happened to use costs a different program."
                )
            }
        return {"error": f"pattern contains unsupported characters: {pattern!r}"}
    cpu = str(cpu or "").strip()
    if not cpu or not SAFE_MODEL_NAME.match(cpu):
        return {
            "error": (
                "cpu is required and must be a model llvm-mca knows, e.g. "
                "neoverse-n1: a cycle count is a property of a specific core."
            )
        }
    if mode not in ("dependent", "independent"):
        return {
            "error": (
                "mode must be 'dependent' (the chain's latency, what a "
                "loop-carried computation meets) or 'independent' (throughput, "
                "what an unrolled loop meets). They disagree by a lot."
            )
        }
    copies = max(2, min(int(copies or 20), 200))
    # One pass over the region, with the length coming from `copies` instead.
    # llvm-mca's own repetitions do not carry the chain: the first copy reads
    # borrowed inputs rather than the last copy's result, so repeating the
    # region lets the model overlap the repetitions and a *dependent* fmul-fadd
    # pair came back at 1.4 cycles when fmul's latency alone is 3. Length in
    # copies keeps the dependence real; length in iterations does not.
    mca_iterations = 1

    mnemonics = [m for m in spec.split("|")[0].split() if m]
    if len(mnemonics) < 2:
        return {"error": "a fusion candidate needs at least two instructions"}
    pieces = [("sequence", spec)] + [
        (f"op{index}", mnemonic)
        for index, mnemonic in enumerate(dict.fromkeys(mnemonics))
    ]

    with tempfile.TemporaryDirectory(prefix="fusion_cost_") as workdir:
        steps = []
        for tag, piece in pieces:
            steps.append(
                f"candidate-coster emit --pattern '{piece}' --mode {mode} "
                f"--copies {copies} --out-dir {tag} > /dev/null 2>{tag}.err "
                f'|| {{ echo "EMITFAIL {tag} $(head -1 {tag}.err)"; }}; '
                f"if [ -f {tag}/baseline.s ]; then "
                f"llvm-mca -mtriple={DEFAULT_ANALYSIS_TARGET} -mcpu={cpu} "
                f"-iterations={mca_iterations} "
                f"-json {tag}/baseline.s > {tag}.json 2>{tag}.mcaerr "
                f'&& echo "OK {tag}" || echo "MCAFAIL {tag} $(head -1 {tag}.mcaerr)"; '
                f"fi; "
            )
        try:
            returncode, stdout, stderr = await _run(
                "".join(steps), workdir, image=image, timeout_seconds=timeout_seconds
            )
        except asyncio.TimeoutError:
            return {"error": f"Costing timed out after {timeout_seconds}s"}
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"cost_fusion_candidate failed: {exc}")
            return {"error": f"Costing failed: {exc}"}

        failures = [
            line
            for line in stdout.splitlines()
            if line.startswith(("EMITFAIL", "MCAFAIL"))
        ]
        costs: Dict[str, float] = {}
        for tag, piece in pieces:
            report = Path(workdir, f"{tag}.json")
            if not report.exists():
                continue
            try:
                summary = _mca_summary(json.loads(report.read_text(encoding="utf-8")))
            except Exception:
                continue
            total = summary.get("TotalCycles")
            instructions = summary.get("Instructions")
            if not total or not instructions:
                continue
            per_copy = float(total) / max(1, copies * mca_iterations)
            costs[tag] = per_copy

    if "sequence" not in costs:
        return {
            "success": False,
            "error": (
                "The sequence could not be costed. "
                + ("; ".join(failures[:3]) if failures else str(stderr)[:200])
            ),
        }

    constituents = {
        piece: costs[tag] for tag, piece in pieces if tag != "sequence" and tag in costs
    }
    bounds = fusion_saving_bounds(costs["sequence"], constituents)
    if "error" in bounds:
        return {"success": False, "error": bounds["error"], "failures": failures[:5]}

    subject = (label or "").strip() or spec
    return {
        "success": True,
        "data": {
            "pattern": spec,
            "cpu": cpu,
            "mode": mode,
            "copies": copies,
            "mca_iterations": mca_iterations,
            **bounds,
            "failures": failures[:5],
        },
        "findings": [
            {
                "type": "fusion_cost_bound",
                "subject": subject,
                "title": (
                    f"{spec} @ {cpu}: {bounds['sequence_cycles_per_occurrence']} "
                    f"cycles now, at best {bounds['max_saving_per_occurrence']} "
                    f"saved by fusing (floor {bounds['slowest_constituent']} at "
                    f"{bounds['slowest_constituent_cycles']})"
                ),
                "max_saving_per_occurrence": bounds["max_saving_per_occurrence"],
                "sequence_cycles_per_occurrence": bounds[
                    "sequence_cycles_per_occurrence"
                ],
                "worth_pursuing": bounds["worth_pursuing"],
                "measurement_source": f"llvm-mca {cpu}",
                "category": "insight",
            }
        ],
    }
