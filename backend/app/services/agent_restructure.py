"""Judge an application-specific restructuring: is it the same program, and faster?

The scanner finds shapes a known pass handles, and `build_llvm_pass` turns a
shape into a pass. Both stay inside what a compiler may assume about *any*
program. The larger optimisations live outside it: this function is only ever
called with a loop-invariant divisor, this array is always sorted, this table
is rebuilt with the same contents on every call. A compiler cannot use facts
about one application; a person -- or a model -- reading that application can.
That is what this module judges, and the judgement is the part that must not
be left to whoever proposed the change.

THE HARNESS. The caller supplies three things: the kernel (the application's
code, one C file), a driver (`main`, which reads a workload from stdin, calls
the kernel and prints what it computed) and inputs. A candidate replaces the
kernel and nothing else. The driver is the caller's and is compiled once, so a
candidate cannot win by changing what is measured or what is printed.

WHAT IS COMPARED, IN ORDER, AND WHY THE ORDER MATTERS.

  1. Both build. A candidate that does not compile has its compiler errors
     returned verbatim; an ORIGINAL that does not build means the harness is
     wrong, which is reported as that rather than blamed on the candidate.
  2. Same output on every input (differential execution). Exact by default;
     with a relative tolerance only when the candidate declares it changes
     results, and then the largest drift is reported. A diverging candidate is
     never timed -- the fastest program is one that computes something else.
  3. Timed, original and candidate INTERLEAVED trial by trial, so a load spike
     on this shared host lands on both arms instead of deciding the verdict.
  4. Against a compiler ceiling: the original rebuilt at -O3 (and -ffast-math
     if the candidate changes results). A candidate that only matches what the
     compiler does when permitted is not an original optimisation, it is a
     flag -- `compiler_already_can` says so.

Equivalence here is equivalence ON THESE INPUTS. For an application-specific
change that is the point -- it relies on something true of this application's
inputs and not of all inputs -- so the invariant it relies on is carried on the
result, and the result says it was checked on N inputs rather than proved.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from app.services import agent_sandbox_runtime
from app.services.agent_compiler_sandbox import (
    _clean_flags,
    explain_compiler_failure,
    measurement_quality,
    robust_spread,
)
from app.services.agent_implementation_check import compare_output

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_FLAGS = "-O2"
DEFAULT_TRIALS = 7
MAX_TRIALS = 15
MAX_INPUTS = 12
MAX_SOURCE_CHARS = 400_000
MAX_INPUT_CHARS = 2_000_000
#: Seconds one run of one arm on one input may take before it counts as hung.
PER_RUN_TIMEOUT = 30
#: Output past this is compared by hash only. Two outputs that hash equal are
#: identical; two that differ and are this large cannot be compared numerically
#: here, and are reported as diverged rather than guessed at.
OUTPUT_EXCERPT_BYTES = 200_000
#: A relative difference smaller than this is never called a win or a loss,
#: however tight the trials: two builds of the same code differ by about this
#: much from code placement alone.
MIN_RESOLVABLE = 0.03
#: Below this the host's own noise decides. Measured on this machine: ~30 ms
#: workloads kept >100% spread even serialised and trimmed.
NOISE_FLOOR_MS = 100
#: The share of the timeout the trials may spend; the rest is builds and runs.
TRIAL_BUDGET_SHARE = 0.6

DEFAULT_TOLERANCE = 1e-9
VALUE_CHANGING_TOLERANCE = 1e-6

VERDICTS = (
    "baseline_broken",
    "did_not_compile",
    "crashed",
    "diverged",
    "slower",
    "unresolved",
    "compiler_already_can",
    "faster_than_original",
    "faster",
)


@dataclass(frozen=True)
class Arm:
    """One program to build: its binary name, how, and whether it is compared.

    The ceiling arm is timed and never compared -- it is the original under
    different flags, and -ffast-math is allowed to change its output.
    """

    name: str
    build: str
    differential: bool = True


def check_inputs(
    inputs: Sequence[str], bench_input: int
) -> Tuple[Optional[str], List[str]]:
    """Refuse an unusable workload before a container is started for it."""
    cleaned = [str(i if i is not None else "") for i in (inputs or [])]
    if not cleaned:
        return (
            "inputs is required: at least one stdin text for the driver. "
            "Equivalence on no inputs is vacuous, and a candidate that has "
            "never been run cannot be called the same program.",
            [],
        )
    if len(cleaned) > MAX_INPUTS:
        return f"at most {MAX_INPUTS} inputs, got {len(cleaned)}", []
    if sum(len(i) for i in cleaned) > MAX_INPUT_CHARS:
        return f"inputs total more than {MAX_INPUT_CHARS} characters", []
    if not 0 <= bench_input < len(cleaned):
        return (
            f"bench_input {bench_input} is not an index into {len(cleaned)} inputs",
            [],
        )
    return None, cleaned


def sandbox_blocked(image: str) -> Optional[str]:
    if not agent_sandbox_runtime.execution_enabled():
        return (
            "Sandboxed execution is disabled (ENABLE_UNSAFE_CODE_EXECUTION is false)."
        )
    if image not in agent_sandbox_runtime.allowed_images():
        return agent_sandbox_runtime.image_not_allowlisted(image)
    return None


# --------------------------------------------------------------------------- #
# The two scripts. Separate runs over one mounted workdir: binaries built by the
# first are still there for the second, and a candidate that diverges never
# pays for timing.
# --------------------------------------------------------------------------- #


def differential_script(prep: str, arms: Sequence[Arm], n_inputs: int) -> str:
    parts = [
        f"if {{ {prep}; }} >build_prep.log 2>&1; then echo '__prep__ 1'; "
        "else echo '__prep__ 0'; echo '__log_begin__ prep'; "
        "head -c 4000 build_prep.log; echo; echo '__log_end__'; exit 0; fi"
    ]
    for arm in arms:
        parts.append(
            f"if {{ {arm.build}; }} >build_{arm.name}.log 2>&1; then "
            f"echo '__built__ {arm.name} 1'; else echo '__built__ {arm.name} 0'; "
            f"echo '__log_begin__ {arm.name}'; head -c 4000 build_{arm.name}.log; "
            "echo; echo '__log_end__'; fi"
        )
    for arm in arms:
        if not arm.differential:
            continue
        for i in range(n_inputs):
            out = f"out_{arm.name}_{i}.txt"
            parts.append(
                f"if [ -x ./{arm.name} ]; then "
                f"timeout {PER_RUN_TIMEOUT} ./{arm.name} <in_{i}.txt >{out} 2>/dev/null; "
                f'echo "__rc__ {arm.name} {i} $?"; '
                f'echo "__hash__ {arm.name} {i} $(sha256sum {out} | cut -c1-16)"; '
                f"echo '__out_begin__ {arm.name} {i}'; head -c {OUTPUT_EXCERPT_BYTES} {out}; "
                "echo; echo '__out_end__'; fi"
            )
    return "; ".join(parts)


def timing_script(
    arms: Sequence[Arm], bench_input: int, trials: int, budget_ns: int
) -> str:
    # Interleaved: trial t runs every arm once before trial t+1 starts, so a
    # stall on the shared host is paid by all arms rather than by whichever
    # happened to be running.
    names = " ".join(a.name for a in arms)
    return (
        f"__t0=$(date +%s%N); "
        f"for a in {names}; do [ -x ./$a ] && ./$a <in_{bench_input}.txt >/dev/null 2>&1; done; "
        f"for t in $(seq 1 {trials}); do "
        f"  for a in {names}; do "
        f"    [ -x ./$a ] || continue; "
        f"    s=$(date +%s%N); timeout {PER_RUN_TIMEOUT} ./$a <in_{bench_input}.txt >/dev/null 2>&1; "
        f'    e=$(date +%s%N); echo "__t__ $a $(( (e-s)/1000 ))"; '
        f"  done; "
        f"  [ $(( $(date +%s%N) - __t0 )) -gt {budget_ns} ] && break; "
        f"done; "
        "echo \"__loadavg__ $(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo x)\"; "
        'echo "__cpus__ $(nproc 2>/dev/null || echo x)"'
    )


def parse_differential(stdout: str) -> Dict[str, Any]:
    prep_ok: Optional[bool] = None
    built: Dict[str, bool] = {}
    logs: Dict[str, str] = {}
    rc: Dict[Tuple[str, int], int] = {}
    hashes: Dict[Tuple[str, int], str] = {}
    outs: Dict[Tuple[str, int], str] = {}

    lines = (stdout or "").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("__prep__ "):
            prep_ok = line.split()[1] == "1"
        elif line.startswith("__built__ "):
            _, name, ok = line.split()[:3]
            built[name] = ok == "1"
        elif line.startswith("__rc__ "):
            _, name, idx, code = (line.split() + ["", "", "", ""])[:4]
            if idx.isdigit() and code.lstrip("-").isdigit():
                rc[(name, int(idx))] = int(code)
        elif line.startswith("__hash__ "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].isdigit():
                hashes[(parts[1], int(parts[2]))] = parts[3]
        elif line.startswith("__log_begin__ ") or line.startswith("__out_begin__ "):
            head = line.split()
            body: List[str] = []
            i += 1
            while i < len(lines) and lines[i] not in ("__log_end__", "__out_end__"):
                body.append(lines[i])
                i += 1
            text = "\n".join(body)
            if head[0] == "__log_begin__":
                logs[head[1]] = text.strip()
            elif len(head) >= 3 and head[2].isdigit():
                # `echo` after `head -c` added one newline; take it back off.
                outs[(head[1], int(head[2]))] = text
        i += 1
    return {
        "prep_ok": prep_ok,
        "built": built,
        "logs": logs,
        "rc": rc,
        "hashes": hashes,
        "outs": outs,
    }


def parse_timings(
    stdout: str,
) -> Tuple[Dict[str, List[int]], Optional[float], Optional[int]]:
    timings: Dict[str, List[int]] = {}
    load: Optional[float] = None
    cpus: Optional[int] = None
    for line in (stdout or "").splitlines():
        parts = line.split()
        if parts[:1] == ["__t__"] and len(parts) == 3 and parts[2].isdigit():
            timings.setdefault(parts[1], []).append(int(parts[2]))
        elif parts[:1] == ["__loadavg__"] and len(parts) == 2:
            try:
                load = float(parts[1])
            except ValueError:
                pass
        elif parts[:1] == ["__cpus__"] and len(parts) == 2 and parts[1].isdigit():
            cpus = int(parts[1])
    return timings, load, cpus


# --------------------------------------------------------------------------- #
# Judgement. Pure functions of what the sandbox printed, so they are testable
# without Docker -- and the verdict logic is the part worth testing.
# --------------------------------------------------------------------------- #


def compare_runs(
    parsed: Dict[str, Any],
    n_inputs: int,
    *,
    baseline: str,
    candidate: str,
    tolerance: float,
) -> Dict[str, Any]:
    """Did the candidate compute what the original computed, on every input?"""
    rc, hashes, outs = parsed["rc"], parsed["hashes"], parsed["outs"]
    per_input: List[Dict[str, Any]] = []
    first_problem: Optional[Dict[str, Any]] = None
    status = "equivalent"
    exact = True

    for i in range(n_inputs):
        b_rc, c_rc = rc.get((baseline, i)), rc.get((candidate, i))
        entry: Dict[str, Any] = {"input": i}
        if b_rc is None or b_rc != 0:
            # The original failing on its own workload is a harness problem:
            # there is nothing to be equivalent to.
            entry["baseline_exit"] = b_rc
            per_input.append(entry)
            return {
                "status": "baseline_broken",
                "detail": (
                    f"the ORIGINAL program exited {b_rc} on input {i} "
                    f"({'timed out' if b_rc == 124 else 'no output' if b_rc is None else 'non-zero'}). "
                    "The driver or the input is wrong; no candidate can be "
                    "judged against a baseline that does not run."
                ),
                "per_input": per_input,
            }
        if c_rc != 0:
            entry["candidate_exit"] = c_rc
            per_input.append(entry)
            status = "crashed"
            first_problem = first_problem or {
                "input": i,
                "detail": (
                    f"candidate {'timed out' if c_rc == 124 else f'exited {c_rc}'} "
                    "where the original ran cleanly"
                ),
            }
            break
        if hashes.get((baseline, i)) and hashes.get((baseline, i)) == hashes.get(
            (candidate, i)
        ):
            entry["identical"] = True
            per_input.append(entry)
            continue
        exact = False
        b_out, c_out = outs.get((baseline, i), ""), outs.get((candidate, i), "")
        if len(b_out) >= OUTPUT_EXCERPT_BYTES or len(c_out) >= OUTPUT_EXCERPT_BYTES:
            outcome_ok, detail = False, (
                "outputs differ byte-for-byte and are too large to compare "
                "numerically here; treat as diverged"
            )
        else:
            outcome = compare_output(c_out, b_out, tolerance)
            outcome_ok, detail = outcome.passed, outcome.detail
        entry["identical"] = False
        entry["within_tolerance"] = outcome_ok
        per_input.append(entry)
        if not outcome_ok:
            status = "diverged"
            first_problem = first_problem or {
                "input": i,
                "detail": detail or "output differs",
                "expected_excerpt": b_out[:300],
                "actual_excerpt": c_out[:300],
            }
            break

    out: Dict[str, Any] = {"status": status, "per_input": per_input}
    if status == "equivalent":
        out["bit_identical"] = exact
        if not exact:
            out["tolerance"] = tolerance
    if first_problem:
        out["first_problem"] = first_problem
    return out


def _best_ms(values: Sequence[int]) -> Optional[float]:
    return round(min(values) / 1000.0, 3) if values else None


def _median_ms(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    med = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return round(med / 1000.0, 3)


def judge_speed(
    timings: Dict[str, List[int]],
    *,
    baseline: str,
    candidate: str,
    ceiling: Optional[str],
    load: Optional[float] = None,
    cpus: Optional[int] = None,
) -> Dict[str, Any]:
    """Faster, slower, or not resolvable -- and whether the compiler gets there too.

    A win must hold on the fastest trial AND the median, and exceed both arms'
    trimmed spread: the fastest alone is one lucky trial, and on this host a
    single lucky trial is common enough to manufacture a speedup from nothing.
    """
    base, cand = timings.get(baseline, []), timings.get(candidate, [])
    if not base or not cand:
        return {"verdict": "unresolved", "detail": "no timings were collected"}

    b_best, c_best = _best_ms(base), _best_ms(cand)
    b_med, c_med = _median_ms(base), _median_ms(cand)
    spreads = [
        robust_spread(base).get("trial_spread", 0.0),
        robust_spread(cand).get("trial_spread", 0.0),
    ]
    noise = max([MIN_RESOLVABLE] + [s for s in spreads if s is not None])

    speedup = round(b_best / c_best, 3) if c_best else None
    median_speedup = round(b_med / c_med, 3) if c_med else None
    out: Dict[str, Any] = {
        "baseline_ms": {"fastest": b_best, "median": b_med, "trials": len(base)},
        "candidate_ms": {"fastest": c_best, "median": c_med, "trials": len(cand)},
        "speedup": speedup,
        "median_speedup": median_speedup,
        "resolvable_difference": round(noise, 3),
    }
    out.update(measurement_quality(load, cpus))

    warnings: List[str] = []
    if b_best is not None and b_best < NOISE_FLOOR_MS:
        warnings.append(
            f"the original runs in {b_best} ms, under the ~{NOISE_FLOOR_MS} ms this "
            "host can resolve; enlarge the bench input before trusting a verdict"
        )
    if len(base) < 4:
        warnings.append("fewer than four trials; the spread is barely an estimate")

    if (
        speedup
        and median_speedup
        and speedup > 1 + noise
        and median_speedup > 1 + noise
    ):
        verdict = "faster"
    elif (
        speedup
        and median_speedup
        and speedup < 1 / (1 + noise)
        and median_speedup < 1 / (1 + noise)
    ):
        verdict = "slower"
    else:
        verdict = "unresolved"

    if ceiling and verdict == "faster":
        ceil = timings.get(ceiling, [])
        ceil_best = _best_ms(ceil)
        if ceil_best:
            ceiling_ratio = round(ceil_best / c_best, 3)
            ceiling_gain = round(b_best / ceil_best, 3)
            out["ceiling_ms"] = {"fastest": ceil_best, "median": _median_ms(ceil)}
            out["speedup_over_ceiling"] = ceiling_ratio
            out["ceiling_speedup_over_original"] = ceiling_gain
            if ceiling_ratio > 1 + noise:
                pass
            elif ceiling_gain > 1 + noise:
                # The compiler, given the licence, measurably gets the gain.
                verdict = "compiler_already_can"
            else:
                # Neither separable from the ceiling nor shown to be matched
                # by it. Measured: a sincos rewrite 1.31x over -O3 was first
                # labelled compiler_already_can because the host's noise was
                # above 31% -- a claim about the compiler with no evidence.
                verdict = "faster_than_original"
                warnings.append(
                    "beats the original, but whether it beats the compiler at "
                    "the ceiling flags is inside the noise; rerun on a quieter "
                    "host or a longer workload"
                )
    if warnings:
        out["warnings"] = warnings
    out["verdict"] = verdict
    return out


_STATIC_VAR = re.compile(
    r"^\s*static\s+(?!inline\b)(?!const\b)[\w\s\*]+?\b\w+\s*(?:\[[^\]]*\]\s*)*(?:=|;)",
    re.M,
)


def mutable_statics(source: str) -> int:
    """File- or function-scope `static` variables that are not const.

    A candidate that adds state surviving between calls may be memoising the
    driver's repetition rather than making the work cheaper. That can be a
    legitimate application-specific change -- if the application really repeats
    calls the way the driver does -- and it can be a benchmark artefact. Only
    the caller can say which, so it is flagged, not refused.
    """
    return len(_STATIC_VAR.findall(source or ""))


# --------------------------------------------------------------------------- #
# The shared run: prep, build every arm, differential, then timing.
# --------------------------------------------------------------------------- #


async def run_comparison(
    *,
    files: Dict[str, Any],
    prep: str,
    arms: Sequence[Arm],
    inputs: Sequence[str],
    bench_input: int,
    trials: int,
    tolerance: float,
    image: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Build, compare and time. Returns the raw judgement pieces.

    `files` maps names to text or bytes; the caller's build lines refer to
    them. The first arm is the baseline and the second the candidate.
    """
    baseline, candidate = arms[0].name, arms[1].name
    # By name, not position: other timed-only arms (a hand rewrite a pass is
    # compared with) may sit beside the ceiling, and must not be mistaken for it.
    ceiling = next((a.name for a in arms[2:] if a.name == "ceiling"), None)

    with tempfile.TemporaryDirectory(prefix="restructure_") as workdir:
        for name, content in files.items():
            target = Path(workdir, name)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
        for i, text in enumerate(inputs):
            Path(workdir, f"in_{i}.txt").write_text(text, encoding="utf-8")

        try:
            _, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                differential_script(prep, arms, len(inputs)),
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                "verdict": "unresolved",
                "error": f"builds and differential runs timed out after {timeout_seconds}s",
            }
        except FileNotFoundError:
            return {
                "verdict": "unresolved",
                "error": "Docker is not available to this process",
            }
        except Exception as exc:  # pragma: no cover - defensive
            detail = str(exc).strip() or exc.__class__.__name__
            logger.warning(f"restructure comparison failed: {detail}")
            return {"verdict": "unresolved", "error": f"sandbox run failed: {detail}"}

        parsed = parse_differential(stdout)
        if parsed["prep_ok"] is not True:
            return {
                "verdict": "baseline_broken",
                "detail": (
                    "the harness itself did not build, so no candidate was "
                    "judged: "
                    + explain_compiler_failure(parsed["logs"].get("prep") or stderr)
                ),
            }
        if not parsed["built"].get(baseline):
            return {
                "verdict": "baseline_broken",
                "detail": "the ORIGINAL did not build: "
                + explain_compiler_failure(parsed["logs"].get(baseline, "")),
            }
        if not parsed["built"].get(candidate):
            return {
                "verdict": "did_not_compile",
                "compile_errors": explain_compiler_failure(
                    parsed["logs"].get(candidate, "")
                ),
            }

        equivalence = compare_runs(
            parsed,
            len(inputs),
            baseline=baseline,
            candidate=candidate,
            tolerance=tolerance,
        )
        if equivalence["status"] != "equivalent":
            return {"verdict": equivalence["status"], "equivalence": equivalence}

        timed = [a for a in arms if parsed["built"].get(a.name)]
        budget_ns = int(timeout_seconds * TRIAL_BUDGET_SHARE * 1_000_000_000)
        try:
            _, t_out, _ = await agent_sandbox_runtime.run_in_sandbox(
                timing_script(timed, bench_input, trials, budget_ns),
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return {
                "verdict": "unresolved",
                "equivalence": equivalence,
                "error": f"equivalent, but timing did not finish: {detail}",
            }

    timings, load, cpus = parse_timings(t_out)
    speed = judge_speed(
        timings,
        baseline=baseline,
        candidate=candidate,
        ceiling=ceiling if ceiling in timings else None,
        load=load,
        cpus=cpus,
    )
    # Every arm's fastest trial, from the same interleaved run, so a caller
    # comparing more than two programs compares numbers taken together.
    speed["arms_fastest_ms"] = {
        name: _best_ms(values) for name, values in sorted(timings.items())
    }
    return {
        "verdict": speed.pop("verdict"),
        "equivalence": equivalence,
        "timing": speed,
    }


# --------------------------------------------------------------------------- #
# Source mode: a rewritten kernel.c.
# --------------------------------------------------------------------------- #


def _source_preflight(kernel: str, candidate: str, driver: str) -> Optional[str]:
    for label, text in (
        ("kernel", kernel),
        ("candidate", candidate),
        ("driver", driver),
    ):
        if not (text or "").strip():
            return f"{label} is required"
        if len(text) > MAX_SOURCE_CHARS:
            return f"{label} exceeds {MAX_SOURCE_CHARS} characters"
    if "main(" not in driver.replace(" ", ""):
        return (
            "driver must define main: it reads a workload from stdin, calls the "
            "kernel and prints what the kernel computed"
        )
    return None


async def evaluate_restructuring(
    *,
    kernel: str,
    candidate: str,
    driver: str,
    inputs: List[str],
    value_preserving: bool = True,
    invariant: str = "",
    flags: str = DEFAULT_FLAGS,
    bench_input: int = 0,
    trials: int = DEFAULT_TRIALS,
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Is `candidate` the same program as `kernel` on these inputs, and faster?"""
    problem = _source_preflight(kernel, candidate, driver)
    if problem:
        return {"error": problem}
    problem, cleaned = check_inputs(inputs, bench_input)
    if problem:
        return {"error": problem}
    safe = _clean_flags(flags or DEFAULT_FLAGS)
    if safe is None:
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    blocked = sandbox_blocked(image)
    if blocked:
        return {"error": blocked}

    # The ceiling is the compiler given every liberty the candidate takes: a
    # value-changing candidate is compared against -ffast-math, because that is
    # the compiler's own licence to change results.
    ceiling_flags = "-O3" if value_preserving else "-O3 -ffast-math"
    arms = [
        Arm(
            "orig",
            f"clang {safe} -c kernel.c -o kernel.o && clang {safe} -o orig driver.o kernel.o -lm",
        ),
        Arm(
            "cand",
            f"clang {safe} -c candidate.c -o candidate.o && clang {safe} -o cand driver.o candidate.o -lm",
        ),
        Arm(
            "ceiling",
            f"clang {safe} {ceiling_flags} -c kernel.c -o kernel_ceil.o && "
            f"clang {safe} -o ceiling driver.o kernel_ceil.o -lm",
            differential=False,
        ),
    ]
    tolerance = DEFAULT_TOLERANCE if value_preserving else VALUE_CHANGING_TOLERANCE
    result = await run_comparison(
        files={"kernel.c": kernel, "candidate.c": candidate, "driver.c": driver},
        prep=f"clang {safe} -c driver.c -o driver.o",
        arms=arms,
        inputs=cleaned,
        bench_input=bench_input,
        trials=max(3, min(int(trials or DEFAULT_TRIALS), MAX_TRIALS)),
        tolerance=tolerance,
        image=image,
        timeout_seconds=timeout_seconds,
    )
    return package(
        result,
        kind="restructuring_result",
        label=label,
        invariant=invariant,
        value_preserving=value_preserving,
        n_inputs=len(cleaned),
        added_state=max(0, mutable_statics(candidate) - mutable_statics(kernel)),
        ceiling_flags=ceiling_flags,
    )


def package(
    result: Dict[str, Any],
    *,
    kind: str,
    label: str,
    invariant: str,
    value_preserving: bool,
    n_inputs: int,
    added_state: int = 0,
    ceiling_flags: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The tool result: the verdict, what it rests on, and a finding."""
    if (
        "error" in result
        and result.get("verdict") == "unresolved"
        and "equivalence" not in result
    ):
        return {"success": False, "error": result["error"]}

    verdict = result["verdict"]
    data: Dict[str, Any] = {
        "verdict": verdict,
        **{k: v for k, v in result.items() if k != "verdict"},
    }
    data["value_preserving"] = value_preserving
    data["checked_on_inputs"] = n_inputs
    if invariant:
        data["relies_on"] = invariant
    if ceiling_flags:
        data["ceiling_flags"] = ceiling_flags
    if extra:
        data.update(extra)
    notes: List[str] = []
    if verdict in (
        "faster",
        "faster_than_original",
        "compiler_already_can",
        "unresolved",
        "slower",
    ):
        notes.append(
            f"Equivalent on {n_inputs} input(s), not proved equivalent"
            + (f"; it relies on: {invariant}" if invariant else "")
            + ". Add an input that exercises the edge of that assumption before relying on it."
        )
    if added_state:
        notes.append(
            f"The candidate adds {added_state} mutable static variable(s): state that "
            "survives between calls. If it caches work the driver repeats, the "
            "gain is real only where the application repeats calls the same way."
        )
    if verdict == "compiler_already_can":
        notes.append(
            f"The original rebuilt with {ceiling_flags} is as fast: this is a flag, "
            "not an original optimisation."
        )
    if notes:
        data["notes"] = notes

    timing = result.get("timing") or {}
    subject = (label or "").strip() or "candidate"
    title = f"{subject}: {verdict}"
    if timing.get("speedup"):
        title += f" ({timing['speedup']}x over original"
        if timing.get("speedup_over_ceiling"):
            title += (
                f", {timing['speedup_over_ceiling']}x over {ceiling_flags or 'ceiling'}"
            )
        title += ")"
    return {
        "success": verdict not in ("baseline_broken",),
        "data": data,
        "findings": [
            {
                "type": kind,
                "subject": subject,
                "title": title,
                "verdict": verdict,
                "speedup": timing.get("speedup"),
                "median_speedup": timing.get("median_speedup"),
                "speedup_over_ceiling": timing.get("speedup_over_ceiling"),
                "value_preserving": value_preserving,
                "relies_on": invariant or None,
                "checked_on_inputs": n_inputs,
                "measurement_environment": timing.get("measurement_environment"),
            }
        ],
    }
