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
import shutil
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


def differential_script(
    prep: str, arms: Sequence[Arm], n_inputs: int, run_args: str = ""
) -> str:
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
                f"timeout {PER_RUN_TIMEOUT} ./{arm.name} {run_args} <in_{i}.txt >{out} 2>/dev/null; "
                f'echo "__rc__ {arm.name} {i} $?"; '
                f'echo "__hash__ {arm.name} {i} $(sha256sum {out} | cut -c1-16)"; '
                f"echo '__out_begin__ {arm.name} {i}'; head -c {OUTPUT_EXCERPT_BYTES} {out}; "
                "echo; echo '__out_end__'; fi"
            )
    return "; ".join(parts)


CPUTIME_SOURCE = "__cputime.c"
CPUTIME_BIN = "__cputime"
#: Runs one program and writes "wall_us cpu_us" to a file. rusage's user and
#: sys times come from the scheduler's own accounting, so they resolve far
#: below the 10 ms `times` and /usr/bin/time report in this image.
CPUTIME_C = r"""
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
static pid_t child;
static void on_alarm(int s) { (void)s; if (child > 0) kill(child, SIGKILL); }
int main(int argc, char **argv) {
    if (argc < 4) return 125;
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    child = fork();
    if (child == 0) { execv(argv[3], argv + 3); _exit(127); }
    signal(SIGALRM, on_alarm);
    alarm((unsigned)atoi(argv[2]));
    int st; struct rusage ru;
    if (wait4(child, &st, 0, &ru) < 0) return 126;
    clock_gettime(CLOCK_MONOTONIC, &b);
    long long wall = (b.tv_sec - a.tv_sec) * 1000000LL + (b.tv_nsec - a.tv_nsec) / 1000;
    long long cpu = ru.ru_utime.tv_sec * 1000000LL + ru.ru_utime.tv_usec
                  + ru.ru_stime.tv_sec * 1000000LL + ru.ru_stime.tv_usec;
    FILE *f = fopen(argv[1], "w");
    if (f) { fprintf(f, "%lld %lld\n", wall, cpu); fclose(f); }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}
"""


def timing_script(
    arms: Sequence[Arm],
    bench_input: int,
    trials: int,
    budget_ns: int,
    run_args: str = "",
) -> str:
    # Interleaved: trial t runs every arm once before trial t+1 starts, so a
    # stall on the shared host is paid by all arms rather than by whichever
    # happened to be running.
    #
    # And ROTATED: trial t starts at arm t mod k. Always running the baseline
    # first gave every pair the same position effect, and the paired analysis
    # -- which assumes a pair's members are exchangeable -- turned it into a
    # verdict: BOLT's standard recipe on Lua read "slower", CI [0.776, 0.995],
    # and neutral, CI [0.999, 1.054], on the next identical run.
    names = " ".join(a.name for a in arms)
    order_cases = " ".join(
        f"{i}) order='{' '.join(a.name for a in list(arms[i:]) + list(arms[:i]))}';;"
        for i in range(len(arms))
    )
    # The helper times each run itself: wall and user+sys in microseconds,
    # with its own timeout, since `timeout` around it would kill the helper
    # and orphan the program still running. If it fails to build, the shell's
    # clock is the fallback and the verdict is on wall time.
    return (
        f"clang -O2 -o {CPUTIME_BIN} {CPUTIME_SOURCE} 2>/dev/null; "
        f"__t0=$(date +%s%N); "
        f"for a in {names}; do [ -x ./$a ] && ./$a {run_args} <in_{bench_input}.txt >/dev/null 2>&1; done; "
        f"for t in $(seq 1 {trials}); do "
        f"  case $(( t % {max(1, len(arms))} )) in {order_cases} esac; "
        f"  for a in $order; do "
        f"    [ -x ./$a ] || continue; "
        f"    if [ -x ./{CPUTIME_BIN} ]; then "
        # A run that fails is not a time. A prototype timed a binary that
        # was never built at 3 ms, beside the real one at 250 ms.
        f"      ./{CPUTIME_BIN} __t.txt {PER_RUN_TIMEOUT} ./$a {run_args} <in_{bench_input}.txt >/dev/null 2>&1; "
        f"      rc=$?; "
        f'      if [ $rc -eq 0 ]; then echo "__t__ $a $(cat __t.txt)"; else echo "__tfail__ $a $rc"; fi; '
        f"    else "
        f"      s=$(date +%s%N); timeout {PER_RUN_TIMEOUT} ./$a {run_args} <in_{bench_input}.txt >/dev/null 2>&1; "
        f"      rc=$?; e=$(date +%s%N); "
        f'      if [ $rc -eq 0 ]; then echo "__t__ $a $(( (e-s)/1000 ))"; else echo "__tfail__ $a $rc"; fi; '
        f"    fi; "
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
        if parts[:1] == ["__t__"] and len(parts) in (3, 4) and parts[2].isdigit():
            timings.setdefault(parts[1], []).append(int(parts[2]))
        elif parts[:1] == ["__loadavg__"] and len(parts) == 2:
            try:
                load = float(parts[1])
            except ValueError:
                pass
        elif parts[:1] == ["__cpus__"] and len(parts) == 2 and parts[1].isdigit():
            cpus = int(parts[1])
    return timings, load, cpus


def parse_timing_failures(stdout: str) -> Dict[str, List[int]]:
    """Exit codes of timed runs that failed, per arm."""
    failed: Dict[str, List[int]] = {}
    for line in (stdout or "").splitlines():
        parts = line.split()
        if (
            parts[:1] == ["__tfail__"]
            and len(parts) == 3
            and parts[2].lstrip("-").isdigit()
        ):
            failed.setdefault(parts[1], []).append(int(parts[2]))
    return failed


def parse_cpu_timings(stdout: str) -> Dict[str, List[int]]:
    """User+sys microseconds per arm, where the helper reported them."""
    cpu: Dict[str, List[int]] = {}
    for line in (stdout or "").splitlines():
        parts = line.split()
        if parts[:1] == ["__t__"] and len(parts) == 4 and parts[3].isdigit():
            cpu.setdefault(parts[1], []).append(int(parts[3]))
    return cpu


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


#: A program whose CPU time exceeds its wall time by more than this is using
#: several cores, and CPU time would charge its parallelism as a cost.
PARALLEL_CPU_RATIO = 1.2


def judge_speed(
    timings: Dict[str, List[int]],
    *,
    baseline: str,
    candidate: str,
    ceiling: Optional[str],
    load: Optional[float] = None,
    cpus: Optional[int] = None,
    cpu_timings: Optional[Dict[str, List[int]]] = None,
    control: Optional[str] = None,
) -> Dict[str, Any]:
    """Judge on CPU time when it is the honest measure, wall time otherwise.

    On a Linux host, user+sys time does not count the moments a program sat
    descheduled behind someone else's work, so a single-threaded kernel's work
    is measured without the run queue. It does NOT help inside Docker
    Desktop's VM, and this is measured rather than assumed: when the host takes
    a vCPU away the guest cannot tell, so the time is charged as running.
    raylib's blur null control read a 0.159 noise band on CPU time and 0.159
    on wall time in the same run. Both readings are reported so the difference
    is visible where there is one.

    A program using several cores is the exception: its CPU time sums the
    threads and would read as slower exactly when it is faster, so any arm
    whose CPU time exceeds its wall time sends the verdict back to wall time.
    """
    names = [baseline, candidate] + ([ceiling] if ceiling else [])
    usable = bool(cpu_timings) and all(
        cpu_timings.get(n) and timings.get(n) for n in (baseline, candidate)
    )
    parallel = [
        n
        for n in names
        if usable
        and cpu_timings.get(n)
        and timings.get(n)
        and _median_ms(cpu_timings[n]) > PARALLEL_CPU_RATIO * _median_ms(timings[n])
    ]
    if not usable or parallel:
        out = _judge_on(
            timings,
            baseline=baseline,
            candidate=candidate,
            ceiling=ceiling,
            control=control,
            load=load,
            cpus=cpus,
        )
        out["basis"] = "wall"
        if parallel:
            out.setdefault("warnings", []).append(
                f"{', '.join(parallel)} used more than one core (CPU time above "
                "wall time), so this is judged on wall time and carries the "
                "host's scheduling noise"
            )
        return out

    out = _judge_on(
        cpu_timings,
        baseline=baseline,
        candidate=candidate,
        ceiling=ceiling,
        control=control,
        load=load,
        cpus=cpus,
    )
    wall = _judge_on(
        timings,
        baseline=baseline,
        candidate=candidate,
        ceiling=ceiling,
        control=control,
    )
    out["basis"] = "cpu"
    out["wall"] = {
        k: wall.get(k)
        for k in (
            "verdict",
            "speedup",
            "median_speedup",
            "resolvable_difference",
            "baseline_ms",
            "candidate_ms",
        )
    }
    return out


CONTROL_ARM = "control"

#: Paired analysis needs enough pairs for a 95% interval on the median to
#: exist at all: with 5 pairs the tightest order-statistic interval covers
#: only 94%, so below this the unpaired rule decides.
MIN_PAIRS = 6


def paired_ratio(num: Sequence[int], den: Sequence[int]) -> Optional[Dict[str, Any]]:
    """Median of per-trial ratios num[t]/den[t], with a 95% interval.

    The trials are interleaved -- trial t runs every arm back to back -- so a
    slow stretch on the host lands on both members of a pair. Comparing each
    arm's fastest and median separately throws that away and has to clear the
    whole host's spread. Measured on Lua under BOLT: a ~4% layout effect sat
    under a 15-80% spread band on a "quiet" host, and every configuration,
    including the standard recipe, read unresolved.

    The interval is distribution-free: order statistics x(k)..x(n-k+1) of the
    sorted ratios, with k the largest rank for which P(Binomial(n, 1/2) < k)
    stays within 2.5%. It assumes only that pairs are independent, which
    interleaving is for.
    """
    n = min(len(num), len(den))
    if n < MIN_PAIRS or len(num) != len(den):
        return None
    ratios = sorted(num[i] / den[i] for i in range(n) if den[i] > 0)
    if len(ratios) < MIN_PAIRS:
        return None
    n = len(ratios)
    from math import comb

    k, cumulative = 0, 0.0
    while True:
        cumulative += comb(n, k) / 2**n
        if cumulative > 0.025:
            break
        k += 1
    # k is now the count of order statistics trimmed from each end.
    k = max(k, 1)
    mid = n // 2
    median = ratios[mid] if n % 2 else (ratios[mid - 1] + ratios[mid]) / 2
    return {
        "median": round(median, 4),
        "ci95": [round(ratios[k - 1], 4), round(ratios[n - k], 4)],
        "pairs": n,
    }


def _clears(
    pair: Dict[str, Any],
    slower: bool = False,
    null: Optional[Dict[str, Any]] = None,
) -> bool:
    """A paired effect that is certain, larger than code placement, and --
    when the run carried a control -- outside what an identical binary showed.

    The control is the empirical null: a byte-identical copy of the baseline,
    interleaved with everything else. Calibrated on Lua with the host at load
    ~59, one of six A/A runs of identical binaries read "faster" at 1.037x,
    CI [1.015, 1.128] -- the interval alone assumes independent pairs, and a
    host whose load arrives in bursts breaks that. The control is measured
    under the same bursts, in the same run.
    """
    low, high = pair["ci95"]
    if slower:
        ok = high < 1 and pair["median"] < 1 / (1 + MIN_RESOLVABLE)
        return ok and (null is None or high < null["ci95"][0])
    ok = low > 1 and pair["median"] > 1 + MIN_RESOLVABLE
    return ok and (null is None or low > null["ci95"][1])


def _judge_on(
    timings: Dict[str, List[int]],
    *,
    baseline: str,
    candidate: str,
    ceiling: Optional[str],
    load: Optional[float] = None,
    cpus: Optional[int] = None,
    control: Optional[str] = None,
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

    pair = paired_ratio(base, cand)
    null = paired_ratio(base, timings.get(control, [])) if control else None
    if pair:
        out["paired"] = pair
        is_faster, is_slower = _clears(pair, null=null), _clears(
            pair, slower=True, null=null
        )
        if null:
            out["control"] = null
            if _clears(null) or _clears(null, slower=True):
                warnings.append(
                    "the run's control -- a byte-identical copy of the baseline "
                    f"-- itself read {null['median']}x (CI {null['ci95']}); this "
                    "host is manufacturing differences, and only an effect "
                    "outside the control's interval is reported"
                )
    else:
        is_faster = bool(
            speedup
            and median_speedup
            and speedup > 1 + noise
            and median_speedup > 1 + noise
        )
        is_slower = bool(
            speedup
            and median_speedup
            and speedup < 1 / (1 + noise)
            and median_speedup < 1 / (1 + noise)
        )
    verdict = "faster" if is_faster else "slower" if is_slower else "unresolved"

    if ceiling and verdict == "faster":
        ceil = timings.get(ceiling, [])
        ceil_best = _best_ms(ceil)
        if ceil_best:
            ceiling_ratio = round(ceil_best / c_best, 3)
            ceiling_gain = round(b_best / ceil_best, 3)
            out["ceiling_ms"] = {"fastest": ceil_best, "median": _median_ms(ceil)}
            out["speedup_over_ceiling"] = ceiling_ratio
            out["ceiling_speedup_over_original"] = ceiling_gain
            over_ceiling = paired_ratio(ceil, cand)
            ceiling_over_base = paired_ratio(base, ceil)
            if over_ceiling:
                out["paired_over_ceiling"] = over_ceiling
            beats_ceiling = (
                _clears(over_ceiling, null=null)
                if over_ceiling
                else ceiling_ratio > 1 + noise
            )
            ceiling_gains = (
                _clears(ceiling_over_base, null=null)
                if ceiling_over_base
                else ceiling_gain > 1 + noise
            )
            if beats_ceiling:
                pass
            elif ceiling_gains:
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
    run_args: str = "",
    collect: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build, compare and time. Returns the raw judgement pieces.

    `collect` names files the arm builds leave in the workdir -- a BOLT log's
    statistics, say -- to be read back after the run, capped at 64 KB each.

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
        Path(workdir, CPUTIME_SOURCE).write_text(CPUTIME_C, encoding="utf-8")

        try:
            _, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                differential_script(prep, arms, len(inputs), run_args),
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
            out = {"verdict": equivalence["status"], "equivalence": equivalence}
            if equivalence.get("detail"):
                # Every baseline_broken result says why at the top level, where
                # readers look. This one kept its reason inside `equivalence`,
                # and a run was told "Fix the harness: no detail was reported".
                out["detail"] = equivalence["detail"]
            return out

        timed = [a for a in arms if parsed["built"].get(a.name)]
        # The control: a byte-identical copy of the baseline binary, timed with
        # everything else. What it "gains" over the baseline is this host's
        # noise, measured in this run. Copied after the differential stage, so
        # it is the exact binary that was checked.
        if (
            not any(a.name == CONTROL_ARM for a in timed)
            and Path(workdir, baseline).is_file()
        ):
            shutil.copy2(Path(workdir, baseline), Path(workdir, CONTROL_ARM))
            timed.append(Arm(CONTROL_ARM, "true", differential=False))
        budget_ns = int(timeout_seconds * TRIAL_BUDGET_SHARE * 1_000_000_000)
        try:
            _, t_out, _ = await agent_sandbox_runtime.run_in_sandbox(
                timing_script(timed, bench_input, trials, budget_ns, run_args),
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

        collected = {
            name: Path(workdir, name).read_text(encoding="utf-8", errors="replace")[
                :65536
            ]
            for name in collect
            if Path(workdir, name).is_file()
        }

    timings, load, cpus = parse_timings(t_out)
    speed = judge_speed(
        timings,
        baseline=baseline,
        candidate=candidate,
        ceiling=ceiling if ceiling in timings else None,
        load=load,
        cpus=cpus,
        cpu_timings=parse_cpu_timings(t_out),
        control=CONTROL_ARM if CONTROL_ARM in timings else None,
    )
    # Every arm's fastest trial, from the same interleaved run, so a caller
    # comparing more than two programs compares numbers taken together.
    basis = parse_cpu_timings(t_out) if speed.get("basis") == "cpu" else timings
    speed["arms_fastest_ms"] = {
        name: _best_ms(values) for name, values in sorted(basis.items())
    }
    verdict = speed.pop("verdict")
    failures = parse_timing_failures(t_out)
    if failures:
        speed["failed_trials"] = failures
    # Passing every differential run and then failing a timed one is a
    # program that fails sometimes. That outranks any speed it showed.
    if failures.get(candidate):
        verdict = "crashed"
        equivalence = {
            **equivalence,
            "status": "crashed",
            "first_problem": {
                "detail": (
                    f"passed every differential run, then failed "
                    f"{len(failures[candidate])} timed run(s) with exit code(s) "
                    f"{sorted(set(failures[candidate]))}: it fails intermittently"
                )
            },
        }
    elif failures.get(baseline):
        verdict = "baseline_broken"
        detail = (
            f"the ORIGINAL passed every differential run, then failed "
            f"{len(failures[baseline])} timed run(s) with exit code(s) "
            f"{sorted(set(failures[baseline]))} on the bench input"
        )
    out = {"verdict": verdict, "equivalence": equivalence, "timing": speed}
    if verdict == "baseline_broken":
        out["detail"] = detail
    if collected:
        out["collected"] = collected
    return out


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
                # Numeric so a contract can bound it: 1 only for a candidate
                # that beat the original AND the ceiling on identical output.
                "win": 1 if verdict == "faster" else 0,
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
