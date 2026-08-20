"""Generate inline-assembly chains that measure one microarchitectural fact.

Two shapes, and the difference between them is the whole measurement:

  dependent    each instruction reads the previous one's result, so the loop
               runs at the instruction's *latency*
  independent  several chains interleaved, so latency is hidden and the loop
               runs at the instruction's *reciprocal throughput*

The instructions are written as inline assembly rather than C. Asked to measure
C expressions the compiler vectorises, unrolls, folds to a closed form, or
hoists the work out of the loop entirely -- all four have already happened in
this project, and each produced a confident wrong number rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class Instruction:
    """One instruction to characterise, and how to build a chain of it."""

    name: str
    # Template with {d} destination and {s} source register indices.
    template: str
    # "x" integer, "s" 32-bit float, "d" 64-bit float, "v" vector.
    regclass: str
    # Registers to initialise so the chain neither faults nor denormalises.
    init: str = ""


INSTRUCTIONS: Sequence[Instruction] = (
    # The anchor: a dependent integer add retires one per cycle on any sane
    # core, which is what converts wall clock into cycles.
    Instruction("add", "add x{d}, x{s}, #1", "x"),
    Instruction("mul", "mul x{d}, x{s}, x{s}", "x"),
    Instruction("fadd_s", "fadd s{d}, s{s}, s{s}", "s"),
    Instruction("fmul_s", "fmul s{d}, s{s}, s{s}", "s"),
    Instruction("fmadd_s", "fmadd s{d}, s{s}, s{s}, s{s}", "s"),
    Instruction("fdiv_s", "fdiv s{d}, s{s}, s{s}", "s"),
    Instruction("fsqrt_s", "fsqrt s{d}, s{s}", "s"),
    Instruction("fadd_v", "fadd v{d}.4s, v{s}.4s, v{s}.4s", "v"),
    Instruction("fmla_v", "fmla v{d}.4s, v{s}.4s, v{s}.4s", "v"),
)

# Registers safe to clobber in AArch64 inline asm, avoiding the frame pointer,
# link register and platform register.
INT_REGS = list(range(9, 18)) + list(range(19, 29)) + list(range(0, 8))
# v16-v31 are caller-saved, then v0-v7 the argument registers. A pool of eight
# was not enough: at eight ways every FP throughput came back as exactly
# latency/8, which measures the harness rather than the core.
FP_REGS = list(range(16, 32)) + list(range(0, 8))


def usable_ways(instruction: Instruction, ways: int) -> int:
    """Clamp a requested chain count to the registers this class actually has.

    Clamping rather than raising, because the register pool is a property of
    the ISA and the caller only needs to know the number it really got --
    which it does need, since the chain-limited check depends on it.
    """
    pool = INT_REGS if instruction.regclass == "x" else FP_REGS
    return max(1, min(ways, len(pool)))


def _registers(instruction: Instruction, ways: int) -> List[int]:
    pool = INT_REGS if instruction.regclass == "x" else FP_REGS
    return pool[: usable_ways(instruction, ways)]


def unroll_for(instruction: "Instruction", ways: int, budget_chars: int = 15000) -> int:
    """Pick an unroll that keeps the generated source under the tool's cap.

    Sized from the instruction's own rendered width rather than a constant: a
    fixed guess produced sources over the 20000-character limit for vector and
    three-operand forms, and those instructions were then dropped from the
    table while the rest of it looked complete.
    """
    width = len(instruction.template.format(d=31, s=31)) + 12
    return max(4, min(64, budget_chars // max(1, ways * width)))


def is_chain_limited(latency_cycles: float, recip_throughput_cycles: float, ways: int) -> bool:
    """True when a throughput result is an artifact of too few chains.

    With fewer independent chains than the machine can keep in flight, the
    measured reciprocal throughput is exactly latency/ways -- a property of
    this harness, not of the core. Reporting one as the other is how a
    convincing table of wrong numbers gets made, so the caller must check.
    """
    if recip_throughput_cycles <= 0:
        return True
    return abs(latency_cycles / ways - recip_throughput_cycles) / recip_throughput_cycles < 0.05


def build_source(
    instruction: Instruction,
    *,
    ways: int,
    unroll: int = 0,
    iterations: int = 2_000_000,
    timed: bool = True,
) -> str:
    """Emit a C program timing `ways` interleaved chains of one instruction.

    ways=1 measures latency; ways>1 hides latency and approaches throughput,
    provided enough independent chains exist to fill the pipeline.
    """
    registers = _registers(instruction, ways)
    unroll = unroll or unroll_for(instruction, ways)
    body_lines: List[str] = []
    for _ in range(unroll):
        for register in registers:
            # Destination and source are the same register: that is what makes
            # the chain dependent within a way, and independent across ways.
            body_lines.append(
                '        "' + instruction.template.format(d=register, s=register) + r'\n\t"'
            )
    body = "\n".join(body_lines)

    if instruction.regclass == "x":
        setup = "\n".join(
            f'    asm volatile("mov x{r}, #1" ::: "x{r}");' for r in registers
        )
        clobbers = ", ".join(f'"x{r}"' for r in registers)
    elif instruction.regclass == "v":
        setup = "\n".join(
            f'    asm volatile("fmov v{r}.4s, #1.0" ::: "v{r}");' for r in registers
        )
        clobbers = ", ".join(f'"v{r}"' for r in registers)
    else:
        setup = "\n".join(
            f'    asm volatile("fmov s{r}, #1.0" ::: "s{r}");' for r in registers
        )
        clobbers = ", ".join(f'"s{r}"' for r in registers)

    total_ops = unroll * ways
    if not timed:
        # The simulator reports cycles itself, so the wall-clock epilogue is
        # not merely redundant there -- it is harmful. `(t1-t0)*1e9 + ns`
        # contracts into a single scalar fmadd, and three of gem5's ARM core
        # models have no functional unit for that instruction and hang. Every
        # benchmark in this suite shared that epilogue, so all of them hung
        # for a reason that had nothing to do with what they measured.
        return f"""
#include <stdio.h>

/* {instruction.name}: {ways} chain(s) x {unroll} steps, {iterations} iterations.
   No timing calls: the simulator counts the cycles. */
int main(void) {{
{setup}
    for (long i = 0; i < {iterations}; i++) {{
        asm volatile(
{body}
            ::: {clobbers}
        );
    }}
    printf("ops=%ld\\n", (long){iterations} * {total_ops});
    return 0;
}}
"""
    return f"""
#include <stdio.h>
#include <time.h>

/* {instruction.name}: {ways} interleaved chain(s), {unroll} unrolled steps.
   {'latency' if ways == 1 else 'reciprocal throughput'} measurement. */
int main(void) {{
{setup}
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < {iterations}; i++) {{
        asm volatile(
{body}
            ::: {clobbers}
        );
    }}
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    double ops = (double){iterations} * {total_ops};
    printf("instruction={instruction.name}\\n");
    printf("ways={ways}\\n");
    printf("ops=%.0f\\n", ops);
    printf("ns_per_op=%.6f\\n", ns / ops);
    return 0;
}}
"""


def cycles_per_op(ns_per_op: float, ghz: float) -> float:
    """Convert a timing to cycles once the anchor has fixed the frequency."""
    return ns_per_op * ghz


def frequency_ghz(anchor_ns_per_op: float) -> float:
    """Derive clock frequency from the anchor's one-cycle dependent add.

    The anchor is the only measurement whose cycle cost is assumed rather than
    measured, and everything else is expressed in terms of it. If the anchor is
    wrong every derived latency is wrong by the same factor -- which is at
    least a single, statable assumption rather than a hidden one.
    """
    if anchor_ns_per_op <= 0:
        raise ValueError("anchor produced no measurable time")
    return 1.0 / anchor_ns_per_op


# A held-out kernel for validating a latency fit. It is inline assembly for the
# same reason the training benchmarks are, plus one specific to validation:
# the native tool compiles with clang and the simulator's with gcc, and on a
# plain C kernel clang vectorised while gcc did not -- so the "same" kernel was
# two different programs and the models missed by 20x for reasons that had
# nothing to do with latency. Assembly removes the compiler from the question.
MIXED_SEQUENCE = (
    "fmul s{r}, s{r}, s{r}",
    "fadd s{r}, s{r}, s{r}",
    "fsqrt s{r}, s{r}",
    "fmul s{r}, s{r}, s{r}",
    "fadd s{r}, s{r}, s{r}",
    "fmadd s{r}, s{r}, s{r}, s{r}",
)


def build_mixed_source(*, iterations: int, unroll: int = 8, timed: bool = True) -> str:
    """A dependent chain cycling through several instruction kinds.

    No single op class dominates, so predicting it right needs the whole
    latency table rather than any one entry -- which is the point of holding
    it out of the fit.
    """
    register = FP_REGS[0]
    body = "\n".join(
        '        "' + step.format(r=register) + r'\n\t"'
        for _ in range(unroll)
        for step in MIXED_SEQUENCE
    )
    total_ops = unroll * len(MIXED_SEQUENCE)
    setup = f'    asm volatile("fmov s{register}, #1.0" ::: "s{register}");'
    clobbers = f'"s{register}"'
    loop = f"""    for (long i = 0; i < {iterations}; i++) {{
        asm volatile(
{body}
            ::: {clobbers}
        );
    }}"""
    if not timed:
        return f"""
#include <stdio.h>
int main(void) {{
{setup}
{loop}
    printf("ops=%ld\\n", (long){iterations} * {total_ops});
    return 0;
}}
"""
    return f"""
#include <stdio.h>
#include <time.h>
int main(void) {{
{setup}
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
{loop}
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    printf("ns_per_op=%.6f\\n", ns / ((double){iterations} * {total_ops}));
    return 0;
}}
"""
