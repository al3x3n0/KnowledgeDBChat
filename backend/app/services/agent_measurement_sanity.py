"""Is the thing measured the thing named?

Controls catch a broken instrument. Replication catches a noisy one. Neither
catches the failure that cost this project the most, because that one is
neither broken nor noisy -- it is precise, stable and about something else.

A dependent floating-point chain changes the value it carries. Starting from
1.0, the chains this project used reach infinity in *single* precision almost
immediately: the held-out mixed kernel at iteration 4, `fmadd` at 8, `fadd` at
128, out of loops that run 100,000 times. So 99.87% to 99.996% of every one of
those runs timed exceptional-value arithmetic rather than the instructions it
named. Four of nine per-instruction latencies and an entire held-out validation
were measurements of infinity.

It reproduced perfectly, run after run, because it was wrong the same way every
time. Every control passed. Every count, bound and uncertainty requirement was
satisfied. Nothing in the numbers said anything was amiss, and the defect was
found only by tracing what the chain actually computes -- which nobody had done
because there was no reason to suspect it.

Two checks, both cheap, both applied to the call rather than left to be
remembered:

* **The chain must stay in normal range.** Simulated in the width the registers
  actually are, over a horizon comparable to the real loop.
* **Reported numbers must be finite.** A harness printing `ns_per_op=inf` has
  told you its measurement failed, and reading that as a number is how it gets
  into a table.

Three classes survive here only by luck: 1.0 is a fixed point of `x*x`,
`sqrt(x)` and `x/x`. Nothing chose them for that, which is exactly why this has
to be checked per sequence instead of reasoned about once.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Iterations to simulate. Must be of the order of the real loop, not a
#: handful: `fadd s0, s0, s0` doubles, so it survives 8 iterations and
#: overflows at about 128. Checking 8 of them reports "finite" for a chain that
#: spends 99.9% of a 100,000-iteration run at infinity -- a mistake made here
#: first, in the original version of this check.
HORIZON = 300

#: Operations whose result feeds the next iteration when every source operand
#: is the destination register.
_GROWING = {
    "fmul": lambda x: x * x,
    "fadd": lambda x: x + x,
    "fsub": lambda x: x - x,
    "fdiv": lambda x: (x / x) if x != 0 else math.nan,
    "fmadd": lambda x: x * x + x,
    "fmla": lambda x: x * x + x,
    "fsqrt": lambda x: math.sqrt(x) if x >= 0 else math.nan,
    "fnmul": lambda x: -(x * x),
}

_ASM_LINE = re.compile(r'"\s*([a-z][a-z0-9.]*)\s+([^"\\]+?)\s*(?:\\n|\\t|")', re.I)
_FMOV_IMM = re.compile(r'"\s*fmov\s+([sdv])(\d+)[^,]*,\s*#([-+0-9.eE]+)', re.I)
_REG = re.compile(r"^([sdqv])(\d+)", re.I)


def f32(value: float) -> float:
    """Round to single precision.

    The registers are `s` registers. Simulating in Python floats is simulating
    a different machine: `fadd` overflows at 2**128 in float32 and not until
    2**1024 in a double, so a double-precision check calls a chain stable that
    spends most of its run at infinity. Also a mistake made here first.
    """
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error, ValueError):
        return math.inf if value > 0 else -math.inf


@dataclass(frozen=True)
class Chain:
    """A self-dependent floating-point chain found in a benchmark's source."""

    register: str
    start: float
    ops: List[str]
    #: Operations on the carried register whose other operands come from
    #: elsewhere. Their values are unknown, so the chain cannot be simulated
    #: past them -- which is reported rather than guessed at.
    opaque: List[str]


def _operands(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def find_chain(code: str) -> Optional[Chain]:
    """The dependent floating-point chain a benchmark's inline assembly carries.

    Returns None when there is nothing of the shape this understands, which is
    the common case and is not a complaint: a C loop with no inline assembly is
    outside what this can analyse, and saying so beats guessing.
    """
    if not isinstance(code, str) or "asm" not in code:
        return None

    seeds = {}
    for kind, number, value in _FMOV_IMM.findall(code):
        try:
            seeds[f"{kind.lower()}{number}"] = float(value)
        except ValueError:
            continue
    if not seeds:
        return None

    # The carried register is the one the ops write. Take the seed with the
    # most operations on it.
    best: Optional[Chain] = None
    for register, start in seeds.items():
        ops: List[str] = []
        opaque: List[str] = []
        for mnemonic, operand_text in _ASM_LINE.findall(code):
            name = mnemonic.lower().split(".")[0]
            if name not in _GROWING:
                continue
            operands = _operands(operand_text)
            if not operands:
                continue
            dest = _REG.match(operands[0])
            if not dest or f"{dest.group(1).lower()}{dest.group(2)}" != register:
                continue
            sources = operands[1:]
            if all(
                (m := _REG.match(s)) and f"{m.group(1).lower()}{m.group(2)}" == register
                for s in sources
            ):
                ops.append(name)
            else:
                # Reads something this cannot see. A value-stable chain looks
                # exactly like this -- `fmul s0, s0, s1` with s1 held at 1.0 --
                # and is the correct way to write one.
                opaque.append(name)
        if ops and (best is None or len(ops) > len(best.ops)):
            best = Chain(register=register, start=start, ops=ops, opaque=opaque)
    return best


def simulate(ops: Sequence[str], start: float, horizon: int = HORIZON) -> List[float]:
    """The value the chain carries, iteration by iteration, in float32."""
    value = f32(start)
    history: List[float] = []
    for _ in range(max(1, horizon)):
        for op in ops:
            handler = _GROWING.get(op)
            if handler is None:
                continue
            try:
                value = f32(handler(value))
            except (ValueError, ZeroDivisionError, OverflowError):
                value = math.nan
            if math.isinf(value) or math.isnan(value):
                history.append(value)
                return history
        history.append(value)
    return history


def chain_leaves_normal_range(code: str) -> Optional[Dict[str, Any]]:
    """Does this benchmark's carried value reach infinity or NaN?

    None when there is no analysable chain. A dict only when there is one *and*
    it leaves normal range -- so a caller can tell "checked and fine" from
    "could not check", which must never read the same way.
    """
    chain = find_chain(code)
    if chain is None or not chain.ops:
        return None

    history = simulate(chain.ops, chain.start)
    if len(history) >= HORIZON:
        return None

    reached = history[-1] if history else math.nan
    return {
        "register": chain.register,
        "start": chain.start,
        "operations": chain.ops,
        "iterations_before_leaving": len(history),
        "reached": "nan" if math.isnan(reached) else ("inf" if reached > 0 else "-inf"),
        "opaque_operations": chain.opaque,
        "reason": (
            f"the chain on {chain.register} reaches "
            f"{'NaN' if math.isnan(reached) else 'infinity'} at iteration "
            f"{len(history)}, so almost every iteration of a long run times "
            "exceptional-value arithmetic rather than the instructions named. "
            "Keep the dependence and not the value: give each operation a "
            "neutral second operand (multiply by 1.0, add 0.0) held in another "
            "register and passed as an input, not named in the clobber list."
        ),
    }


def nonfinite_metrics(result: Any) -> List[str]:
    """Metrics the program reported that are not numbers.

    A harness printing `ns_per_op=inf` has told you its measurement failed, and
    reading that as a number is how it reaches a table.
    """
    data = result.get("data") if isinstance(result, Mapping) else None
    reported = data.get("reported_metrics") if isinstance(data, Mapping) else None
    if not isinstance(reported, Mapping):
        return []

    offenders = []
    for key, value in reported.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, (int, float)) and (
                math.isinf(float(item)) or math.isnan(float(item))
            ):
                offenders.append(str(key))
                break
    return sorted(offenders)


def _has_reported_metrics(result: Any) -> bool:
    data = result.get("data") if isinstance(result, Mapping) else None
    reported = data.get("reported_metrics") if isinstance(data, Mapping) else None
    return bool(isinstance(reported, Mapping) and reported)


def check(params: Any, result: Any) -> Dict[str, Any]:
    """Everything this can say about whether the call measured what it named.

    `checked` is true only when something here was actually analysable. A
    non-empty program is not the same as an analysable one -- a C loop with no
    inline assembly is outside what this understands -- and reporting it as
    checked would make "could not check" and "checked and fine" read the same
    way, which is the rule this module exists to keep.
    """
    code = ""
    if isinstance(params, Mapping):
        code = str(params.get("code") or "")

    chain = chain_leaves_normal_range(code)
    nonfinite = nonfinite_metrics(result)
    analysable = find_chain(code) is not None or _has_reported_metrics(result)

    problems = []
    if chain:
        problems.append(chain["reason"])
    if nonfinite:
        problems.append(
            f"the program reported non-finite values for {', '.join(nonfinite)}, "
            "which is the measurement saying it failed rather than a number"
        )

    return {
        "checked": bool(analysable),
        "sound": not problems,
        "chain": chain,
        "nonfinite_metrics": nonfinite,
        "problems": problems,
    }


def unsound_measurements(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Findings in this run whose measurement was of something else."""
    findings = state.get("findings") if isinstance(state, Mapping) else None
    if not isinstance(findings, Sequence):
        return []

    out = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        record = finding.get("measurement_sanity")
        if not isinstance(record, Mapping) or record.get("sound"):
            continue
        out.append(
            {
                "title": str(finding.get("title") or finding.get("type") or "")[:120],
                "problems": list(record.get("problems") or [])[:3],
            }
        )
    return out


def describe() -> List[str]:
    return [
        "a benchmark's carried floating-point value must stay in normal range "
        "over the length of its loop -- a chain that reaches infinity times "
        "exceptional-value arithmetic, not the instructions it names",
        "a reported metric must be finite; inf or NaN is the measurement "
        "saying it failed",
    ]
