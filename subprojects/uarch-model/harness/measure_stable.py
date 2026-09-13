"""Re-take the per-instruction ground truth without the two defects.

The original table in ``results/m3-truth.json`` has two problems that
``GROUND-TRUTH-DEFECTS.md`` records: four of its nine chains reach infinity
within the first few of 100,000 iterations, and every ratio was pooled across
rounds, which on this host attributes a position bias of about two to whichever
chain ran first.

Both are fixed here, and nothing else is changed:

**Value-stable chains.** The dependence is kept and the value is not. Every
instruction still reads and writes the accumulator, so the latency chain is
intact, but the second operand is neutral -- 1.0 for a multiply, 0.0 for an
add -- so the carried value never moves and the chain never leaves normal
range. ``fsqrt`` is stable at 1.0 by itself.

**Median of per-round ratios.** Pooling totals lets one preempted block
dominate. ABBA ordering was tried and made it worse (0.61 to 1.16 on a null
control), so the disturbance is bursty rather than a trend. The median treats a
disturbed round as one sample among many, and takes the null control to 1.000.

Neutral operands are passed as **inputs** with pinned registers. Putting them
in the clobber list says "destroyed", which is the opposite of what is meant,
and silently leaves them holding whatever was there -- measured here as a chain
that carried 0.0 from the first instruction.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Sequence

from control import null_control

UNROLL = 16
ITERATIONS = 100_000
ROUNDS = 31
#: Independent accumulators for the throughput chains. Must comfortably exceed
#: the deepest latency, or the chain is partly dependent and reports latency
#: divided by the number of ways -- the trap that produced four wrong entries
#: in the first table.
WAYS = 16

#: Each entry: the instruction with `{d}` for its accumulator, and which
#: neutral operands it needs. The accumulator is always both read and written.
CLASSES: Dict[str, Dict[str, object]] = {
    "add":      {"asm": "add x{d}, x{d}, #1",                 "kind": "int"},
    "mul":      {"asm": "mul x{d}, x{d}, x{d}",               "kind": "int"},
    "fadd_s":   {"asm": "fadd s{d}, s{d}, s2",                "kind": "float"},
    "fmul_s":   {"asm": "fmul s{d}, s{d}, s1",                "kind": "float"},
    "fmadd_s":  {"asm": "fmadd s{d}, s{d}, s1, s2",           "kind": "float"},
    "fdiv_s":   {"asm": "fdiv s{d}, s{d}, s1",                "kind": "float"},
    "fsqrt_s":  {"asm": "fsqrt s{d}, s{d}",                   "kind": "float"},
    "fadd_v":   {"asm": "fadd v{d}.4s, v{d}.4s, v2.4s",       "kind": "vector"},
    "fmla_v":   {"asm": "fmla v{d}.4s, v{d}.4s, v2.4s",       "kind": "vector"},
}

#: Accumulator registers. s0-s2 and x9 are reserved for constants and anchor.
FLOAT_ACCS = list(range(8, 8 + WAYS))
INT_ACCS = list(range(10, 10 + WAYS))


def _decls(kind: str, regs: Sequence[int]) -> str:
    lines = ['    register long anc asm("x9") = 1;']
    if kind == "int":
        lines += [f'    register long a{r} asm("x{r}") = 1;' for r in regs]
    else:
        lines += ['    register float one asm("s1") = 1.0f;',
                  '    register float zero asm("s2") = 0.0f;']
        lines += [f'    register float a{r} asm("s{r}") = 1.0f;' for r in regs]
    return "\n".join(lines)


def _constraints(kind: str, regs: Sequence[int]) -> str:
    outs = ", ".join(f'"+r"(a{r})' if kind == "int" else f'"+w"(a{r})' for r in regs)
    if kind == "int":
        return f": {outs}"
    return f': {outs} : "w"(one), "w"(zero)'


def _source(name: str, regs: Sequence[int]) -> str:
    spec = CLASSES[name]
    kind = str(spec["kind"])
    template = str(spec["asm"])
    # One pass over the accumulators per unrolled step: with one register the
    # chain is dependent, with WAYS registers it is not.
    body = "\n".join(
        '            "' + template.format(d=r) + r'\n\t"'
        for _ in range(UNROLL)
        for r in regs
    )
    ops = UNROLL * len(regs)
    anchor = "\n".join(['            "add %[x], %[x], #1\\n\\t"'] * UNROLL)
    return f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int cmp(const void *x, const void *y) {{
    double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}}

int main(void) {{
{_decls(kind, regs)}
    struct timespec t0, t1;
    double ratios[{ROUNDS}];
    for (int r = 0; r < {ROUNDS}; r++) {{
        double ta = 0, tb = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long i = 0; i < {ITERATIONS}; i++) {{
            asm volatile(
{anchor}
                : [x] "+r"(anc));
        }}
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ta = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long i = 0; i < {ITERATIONS}; i++) {{
            asm volatile(
{body}
                {_constraints(kind, regs)});
        }}
        clock_gettime(CLOCK_MONOTONIC, &t1);
        tb = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        /* Per op, against per anchor op: the anchor is a one-cycle dependent
           add, so this ratio IS cycles per operation and the frequency, which
           nobody here knows, divides out. */
        ratios[r] = (tb / (double){ops}) / (ta / (double){UNROLL});
    }}
    qsort(ratios, {ROUNDS}, sizeof(double), cmp);
    printf("median=%.6f\\n", ratios[{ROUNDS} / 2]);
    printf("p25=%.6f\\n", ratios[{ROUNDS} / 4]);
    printf("p75=%.6f\\n", ratios[3 * {ROUNDS} / 4]);
    /* Report the accumulators so a chain that silently went to zero, NaN or
       infinity is visible rather than assumed not to have. */
    printf("final=%g\\n", (double)a{regs[0]});
    return 0;
}}
"""


def _run(name: str, regs: Sequence[int]) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as work:
        source = os.path.join(work, "m.c")
        binary = os.path.join(work, "m")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(_source(name, regs))
        subprocess.run(["clang", "-O2", "-o", binary, source], check=True)
        out = subprocess.run([binary], capture_output=True, text=True, check=True).stdout
    return {k: float(v) for k, v in (l.split("=") for l in out.strip().splitlines())}


def measure_all() -> Dict[str, object]:
    ratio, usable = null_control()
    if not usable:
        return {
            "refused": True,
            "null_control_ratio": round(ratio, 4),
            "reason": (
                f"two identical chains timed {ratio:.3f} apart; this host is "
                "measuring position, not instructions"
            ),
        }

    per_instruction: Dict[str, Dict[str, object]] = {}
    for name, spec in CLASSES.items():
        accs = INT_ACCS if spec["kind"] == "int" else FLOAT_ACCS
        dependent = _run(name, accs[:1])
        independent = _run(name, accs[:WAYS])
        spread = lambda r: round((r["p75"] - r["p25"]) / r["median"], 3) if r["median"] else None
        per_instruction[name] = {
            "latency_cycles": round(dependent["median"], 3),
            "latency_spread": spread(dependent),
            "recip_throughput_cycles": round(independent["median"], 3),
            "throughput_spread": spread(independent),
            "chain_final_value": dependent["final"],
            "stayed_finite": dependent["final"] not in (float("inf"), float("-inf"))
            and dependent["final"] == dependent["final"],
        }

    return {
        "source": "apple-m3-host",
        "method": "value-stable dependent/independent chains, median of "
                  f"{ROUNDS} per-round ratios against a one-cycle dependent add",
        "null_control_ratio": round(ratio, 4),
        "ways": WAYS,
        "per_instruction": per_instruction,
    }


if __name__ == "__main__":
    print(json.dumps(measure_all(), indent=1))
