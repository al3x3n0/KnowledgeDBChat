"""Checks that must pass before any timing on this host means anything.

Two failures cost most of a session before these existed, and neither showed up
as an error -- both produced confident, plausible, wrong numbers.

**The null control.** Time two *identical* dependent add chains, on different
registers, in one program. The ratio must be 1.0. On a loaded machine it is not:
measured at load 173 on 8 cores it read 0.53, and 0.49 with the order reversed,
because whichever block ran first was about twice as slow. A harness that cannot
measure two identical things as equal cannot measure two different things at
all, and every per-instruction latency in this subproject is a ratio of exactly
that kind.

**The value trace.** A dependent floating-point chain changes the value it
carries, and the obvious chains reach infinity almost immediately.
``fadd s0, s0, s0`` from 1.0 doubles every step and overflows in about 128
iterations of a loop that runs 100,000; the held-out mixed kernel gets there in
three. Everything after that times infinity arithmetic, which is not the thing
anyone meant to measure and need not cost the same.

``fmul``, ``fsqrt`` and ``fdiv`` happen to be stable because 1.0 is a fixed
point of all three -- which is luck, not design, and is why this has to be
checked per sequence rather than reasoned about once.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Sequence, Tuple

#: How far the null control may sit from 1.0 and still be believed. This is not
#: a tolerance on results -- it is the point past which the host is disqualified.
NULL_CONTROL_BAND = 0.05

UNROLL = 16
ITERATIONS = 100_000
ROUNDS = 11

_BLOCK = """    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < {iters}; i++) {{
        asm volatile(
{body}
            : [x] "+r"({reg}));
    }}
    clock_gettime(CLOCK_MONOTONIC, &t1);
    {acc} += (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
"""


def _null_control_source() -> str:
    body = "\n".join(['            "add %[x], %[x], #1\\n\\t"'] * UNROLL)
    first = _BLOCK.format(iters=ITERATIONS, body=body, reg="a", acc="first")
    second = _BLOCK.format(iters=ITERATIONS, body=body, reg="b", acc="second")
    return f"""
#include <stdio.h>
#include <time.h>
int main(void) {{
    register long a asm("x9") = 1;
    register long b asm("x10") = 1;
    struct timespec t0, t1;
    double first = 0, second = 0;
    for (int r = 0; r < {ROUNDS}; r++) {{
{first}{second}    }}
    printf("ratio=%.6f\\n", second / first);
    return 0;
}}
"""


def null_control() -> Tuple[float, bool]:
    """Time two identical chains against each other. Returns (ratio, usable).

    A ratio away from 1.0 means the host is measuring position, warm-up or its
    own clock ramp rather than instructions. There is nothing to correct: the
    only fix is a quiet machine.
    """
    with tempfile.TemporaryDirectory() as work:
        source = os.path.join(work, "control.c")
        binary = os.path.join(work, "control")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(_null_control_source())
        subprocess.run(["clang", "-O2", "-o", binary, source], check=True)
        output = subprocess.run(
            [binary], capture_output=True, text=True, check=True
        ).stdout
    ratio = float(output.strip().split("=", 1)[1])
    return ratio, abs(ratio - 1.0) <= NULL_CONTROL_BAND


def host_load() -> Tuple[float, int, float]:
    """One-minute load average, CPU count, and load per CPU."""
    one_minute = os.getloadavg()[0]
    cpus = os.cpu_count() or 1
    return one_minute, cpus, one_minute / cpus


#: Iterations to simulate when asking whether a chain stays finite. It must be
#: of the order of the real loop, not a handful: `fadd s0, s0, s0` doubles, so
#: it survives 8 iterations and overflows at about 128, and checking 8 of them
#: reports "finite" for a chain that spends 99.9% of a 100,000-iteration run at
#: infinity. That mistake was made here first.
FINITE_HORIZON = 300


def simulate_float_chain(
    sequence: Sequence[str], start: float = 1.0, iterations: int = FINITE_HORIZON
) -> List[float]:
    """What value the chain actually carries, iteration by iteration.

    `sequence` is a list of operation names -- "fmul", "fadd", "fsqrt",
    "fmadd", "fdiv" -- applied to the single carried value the way the assembly
    applies them: every operand is the value itself, which is what makes the
    chain dependent.
    """
    import math
    import struct

    def f32(x: float) -> float:
        """Round to single precision after every step.

        The registers are `s` registers. Simulating the chain in Python floats
        is simulating a different machine: `fadd` overflows at 2**128 in float32
        and not until 2**1024 in double, so a double-precision check reports a
        chain stable that spends most of its run at infinity. That mistake was
        also made here first.
        """
        try:
            return struct.unpack("f", struct.pack("f", x))[0]
        except (OverflowError, struct.error):
            return math.inf if x > 0 else -math.inf

    value = f32(start)
    history = []
    for _ in range(iterations):
        for op in sequence:
            if op == "fmul":
                value = f32(value * value)
            elif op == "fadd":
                value = f32(value + value)
            elif op == "fsub":
                value = f32(value - value)
            elif op == "fsqrt":
                value = f32(math.sqrt(value)) if value >= 0 else math.nan
            elif op == "fdiv":
                value = f32(value / value) if value != 0 else math.nan
            elif op == "fmadd":
                value = f32(value * value + value)
            else:
                raise ValueError(f"unknown operation {op!r}")
            if value in (float("inf"), float("-inf")) or value != value:
                history.append(value)
                return history
        history.append(value)
    return history


def stays_finite(sequence: Sequence[str], start: float = 1.0) -> bool:
    """True when the chain never leaves normal range.

    A chain that fails this measures infinity arithmetic for almost all of its
    run, whatever it was intended to measure.
    """
    history = simulate_float_chain(sequence, start=start)
    if len(history) < FINITE_HORIZON:
        # The walk stops early only when it hit infinity or a NaN.
        return False
    return all(abs(v) != float("inf") and v == v for v in history)


def preflight(sequence: Sequence[str] | None = None) -> Dict[str, object]:
    """Everything that must hold before a timing on this host is worth keeping."""
    load, cpus, per_cpu = host_load()
    ratio, usable = null_control()
    report: Dict[str, object] = {
        "load_1min": round(load, 2),
        "cpus": cpus,
        "load_per_cpu": round(per_cpu, 2),
        "null_control_ratio": round(ratio, 4),
        "null_control_passes": usable,
    }
    if sequence is not None:
        finite = stays_finite(sequence)
        report["sequence_stays_finite"] = finite
        walk = simulate_float_chain(sequence)
        report["sequence_values"] = walk[:6]
        report["iterations_before_infinity"] = (
            len(walk) if len(walk) < FINITE_HORIZON else None
        )
    report["usable"] = bool(usable) and bool(report.get("sequence_stays_finite", True))
    if not usable:
        report["refusal"] = (
            f"two identical chains timed {ratio:.3f} apart; at load {per_cpu:.2f} "
            "per CPU this host is measuring position, not instructions"
        )
    elif not report["usable"]:
        report["refusal"] = (
            "the chain reaches infinity, so almost every iteration times "
            "exceptional-value arithmetic rather than the instructions named"
        )
    return report
