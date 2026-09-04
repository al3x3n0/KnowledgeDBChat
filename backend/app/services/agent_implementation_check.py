"""Checking that an implementation computes the right answer before it is timed.

The fastest implementation of any algorithm is one that returns garbage. A run
that reads a paper, writes C, and hands it to `benchmark_c_snippet` gets back a
real wall-clock number for whatever that code does, and nothing anywhere in the
chain asks whether it does the thing the paper described. The timing is
perfectly accurate and completely worthless, and it looks exactly like a good
result -- better, in fact, because the broken version is usually the fast one.

So this runs the implementation against reference cases: inputs with known
outputs, taken from the paper's own worked examples where it gives them. It
compiles the same source that will later be benchmarked, feeds each case on
stdin, and compares what comes back.

Two rules do most of the work, and both are the same rule the test gate uses:

  A check with no cases is not a check. Zero cases passing out of zero is
  vacuously true and reports as verified under any naive counting, which is the
  worst possible reading -- it lets untested code through *believing it
  verified*. `verified` is false unless at least one case actually ran.

  Running and passing are different facts. A program that fails to compile, or
  crashes on the input, has not failed the algorithm's correctness -- it has
  failed earlier, and the response is different. `ran` and `passed` are
  reported separately, so a contract can require both and a run can tell which
  one it is looking at.

Floating point gets a tolerance because an algorithm reimplemented from a paper
will not reproduce the last bits of a reference output, and demanding that it
does rejects correct implementations. It is a relative tolerance so it means
the same thing at 10^-9 and 10^9.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: How far a numeric output may drift from the reference and still be the same
#: answer. An algorithm rebuilt from a paper's prose reassociates sums and
#: picks its own constants; it will not match bit for bit, and requiring that
#: rejects correct work.
DEFAULT_TOLERANCE = 1e-6

#: A cap on cases per check. Each one is a container round trip, and a run that
#: wants a thousand cases wants a test suite, not a gate.
MAX_CASES = 25

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class CaseResult:
    """What one reference case established."""

    name: str
    passed: bool
    ran: bool = True
    expected: str = ""
    actual: str = ""
    #: Why it failed, in terms someone can act on.
    detail: str = ""


@dataclass
class ImplementationCheck:
    """Whether an implementation may be trusted enough to be worth timing."""

    #: Did the code compile and execute at all?
    ran: bool
    cases: List[CaseResult] = field(default_factory=list)
    compile_error: str = ""
    note: str = ""

    @property
    def cases_run(self) -> int:
        return sum(1 for c in self.cases if c.ran)

    @property
    def cases_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def verified(self) -> bool:
        """The gate: it ran, cases exercised it, and every one of them passed.

        Both counts are against `len(self.cases)` -- the cases that were
        *supplied* -- rather than against each other. Comparing passed to run
        looks equivalent and is not: a case that never executed drops out of
        both sides, so a set where one case crashed and the rest passed reports
        as fully verified. The two ways this can be vacuously true are the same
        bug wearing different clothes, and this closes both.
        """
        if not self.ran or not self.cases:
            return False
        return self.cases_run == len(self.cases) and self.cases_passed == len(
            self.cases
        )

    def as_evidence(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "ran": self.ran,
            "cases_run": self.cases_run,
            "cases_passed": self.cases_passed,
            "failing": [
                {
                    "name": c.name,
                    "expected": c.expected,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                for c in self.cases
                if not c.passed
            ][:10],
            "compile_error": self.compile_error[:2000],
            "note": self.note,
        }


def compare_output(
    actual: str, expected: str, tolerance: float = DEFAULT_TOLERANCE
) -> CaseResult:
    """Compare one program's output against a reference output.

    Numeric first: if both sides are sequences of numbers of the same length,
    they are compared with a relative tolerance, because that is what almost
    every algorithm from a paper produces. Otherwise falls back to text with
    whitespace normalised, which is the only textual difference that is never
    meaningful.
    """
    actual_text = (actual or "").strip()
    expected_text = (expected or "").strip()

    actual_nums = _NUMBER.findall(actual_text)
    expected_nums = _NUMBER.findall(expected_text)

    # Only take the numeric path when the reference is numeric and the output
    # matches it in shape. A count mismatch is a real failure and must not be
    # papered over by falling through to a text comparison that also fails but
    # says something less useful.
    if expected_nums and len(actual_nums) != len(expected_nums):
        return CaseResult(
            name="",
            passed=False,
            expected=expected_text[:400],
            actual=actual_text[:400],
            detail=(
                f"Expected {len(expected_nums)} numbers, the program printed "
                f"{len(actual_nums)}"
            ),
        )

    if expected_nums and actual_nums:
        for index, (got, want) in enumerate(zip(actual_nums, expected_nums)):
            try:
                got_value = float(got)
                want_value = float(want)
            except ValueError:  # pragma: no cover - regex guarantees floats
                continue
            scale = max(abs(want_value), 1.0)
            if abs(got_value - want_value) / scale > tolerance:
                return CaseResult(
                    name="",
                    passed=False,
                    expected=expected_text[:400],
                    actual=actual_text[:400],
                    detail=(
                        f"Value {index + 1} differs: expected {want_value:g}, "
                        f"got {got_value:g}"
                    ),
                )
        return CaseResult(
            name="",
            passed=True,
            expected=expected_text[:400],
            actual=actual_text[:400],
        )

    normalized_actual = " ".join(actual_text.split())
    normalized_expected = " ".join(expected_text.split())
    if normalized_actual == normalized_expected:
        return CaseResult(
            name="",
            passed=True,
            expected=expected_text[:400],
            actual=actual_text[:400],
        )
    return CaseResult(
        name="",
        passed=False,
        expected=expected_text[:400],
        actual=actual_text[:400],
        detail="Output does not match the reference",
    )


def normalize_cases(cases: Any) -> List[Dict[str, str]]:
    """Read the cases a caller supplied, dropping any that cannot be a case.

    A case needs an expected output. One without it cannot pass or fail, and
    silently keeping it would inflate `cases_run` with checks that check
    nothing -- the same vacuous-truth hole `verified` closes.
    """
    if not isinstance(cases, (list, tuple)):
        return []
    out: List[Dict[str, str]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        expected = case.get("expected_output")
        if expected is None or not str(expected).strip():
            continue
        out.append(
            {
                "name": str(case.get("name") or f"case {index + 1}")[:80],
                "input": str(case.get("input") or ""),
                "expected_output": str(expected),
            }
        )
        if len(out) >= MAX_CASES:
            break
    return out


async def check_c_implementation(
    *,
    code: str,
    cases: Sequence[Dict[str, str]],
    flags: str = "-O2",
    tolerance: float = DEFAULT_TOLERANCE,
    image: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> ImplementationCheck:
    """Compile a C program once and run it against every reference case.

    The source checked here is the source that gets benchmarked: verifying one
    program and timing a different one proves nothing about the number, and
    keeping them the same string is the only way to know they are the same
    program.
    """
    from app.services import agent_compiler_sandbox as sandbox

    resolved_image = image or sandbox.DEFAULT_IMAGE
    resolved_timeout = timeout_seconds or sandbox.DEFAULT_TIMEOUT_SECONDS

    normalized = normalize_cases(cases)
    if not normalized:
        return ImplementationCheck(
            ran=False,
            note=(
                "No usable reference cases were supplied. A correctness check "
                "with no cases passes vacuously, so this reports as unverified "
                "rather than letting untested code through. Give at least one "
                "case with an expected_output, ideally a worked example from "
                "the paper itself."
            ),
        )

    blocked = sandbox._preflight(code, resolved_image)
    if blocked:
        return ImplementationCheck(
            ran=False, note=str(blocked.get("error") or "sandbox unavailable")
        )

    safe_flags = sandbox._clean_flags(flags)
    if safe_flags is None:
        return ImplementationCheck(
            ran=False, note=f"flags contain unsupported characters: {flags!r}"
        )

    with tempfile.TemporaryDirectory(prefix="impl_check_") as workdir:
        Path(workdir, "impl.c").write_text(code, encoding="utf-8")
        script_parts = [
            f"clang {safe_flags} -o impl impl.c 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }"
        ]
        for index, case in enumerate(normalized):
            Path(workdir, f"case_{index}.in").write_text(
                case["input"], encoding="utf-8"
            )
            # Each case is delimited in the stream rather than run in its own
            # container: one compile and one round trip for the whole set,
            # while still attributing every line to the case that produced it.
            script_parts.append(
                f'echo "__case_begin__ {index}"; '
                f"./impl <case_{index}.in; "
                f'echo "__case_exit__ {index} $?"'
            )
        script = "; ".join(script_parts)

        try:
            returncode, stdout, stderr = await sandbox._run(
                script, workdir, image=resolved_image, timeout_seconds=resolved_timeout
            )
        except Exception as exc:
            logger.warning(f"check_c_implementation failed: {exc}")
            return ImplementationCheck(ran=False, note=f"Check failed: {exc}")

    if returncode == 90:
        return ImplementationCheck(
            ran=False,
            compile_error=sandbox.explain_compiler_failure(stderr),
            note=(
                "The implementation did not compile, so its correctness is "
                "unknown. This is a different failure from a wrong answer: fix "
                "the build before reading anything into the algorithm."
            ),
        )

    outputs = _split_case_output(stdout, len(normalized))
    results: List[CaseResult] = []
    for index, case in enumerate(normalized):
        captured, exit_code = outputs.get(index, ("", None))
        if exit_code is None:
            results.append(
                CaseResult(
                    name=case["name"],
                    passed=False,
                    ran=False,
                    expected=case["expected_output"][:400],
                    detail="The program produced no output for this case",
                )
            )
            continue
        if exit_code != 0:
            results.append(
                CaseResult(
                    name=case["name"],
                    passed=False,
                    ran=True,
                    expected=case["expected_output"][:400],
                    actual=captured[:400],
                    detail=f"The program exited {exit_code} on this input",
                )
            )
            continue
        outcome = compare_output(captured, case["expected_output"], tolerance)
        outcome.name = case["name"]
        results.append(outcome)

    return ImplementationCheck(ran=True, cases=results)


def _split_case_output(stdout: str, expected_cases: int):
    """Attribute each line of the run to the case that produced it."""
    outputs: Dict[int, Any] = {}
    current: Optional[int] = None
    buffer: List[str] = []
    for line in (stdout or "").splitlines():
        if line.startswith("__case_begin__ "):
            current = _int_or_none(line.split()[1:2])
            buffer = []
            continue
        if line.startswith("__case_exit__ "):
            parts = line.split()
            index = _int_or_none(parts[1:2])
            code = _int_or_none(parts[2:3])
            if index is not None:
                outputs[index] = ("\n".join(buffer), code)
            current = None
            buffer = []
            continue
        if current is not None:
            buffer.append(line)
    return outputs


def _int_or_none(parts: Sequence[str]) -> Optional[int]:
    try:
        return int(parts[0])
    except (IndexError, ValueError):
        return None


def describe() -> List[str]:
    """What makes a correctness check worth having, for the run that does one.

    These reach the thinking prompt through `agent_evidence_map.method_notes`,
    keyed to the derived tool chain. A run whose contract asks for a verified
    implementation is about to make every one of these mistakes otherwise.
    """
    return [
        "Verify the implementation BEFORE timing it. The fastest "
        "implementation of any algorithm is one that returns garbage, and a "
        "benchmark of unchecked code is a perfectly accurate number for work "
        "nobody looked at -- it is usually also the best-looking number in the "
        "run.",
        "Reference cases come from the paper's own worked examples. A case "
        "written by reading your own implementation checks that the code does "
        "what it does, which is true of every program ever written.",
        "Check the same source you benchmark. Verifying one program and timing "
        "another establishes nothing about the timing, and the two drift apart "
        "the moment the implementation is edited for speed.",
        "A check with no cases is not a check: zero passing out of zero is "
        "vacuously true, so it is reported as unverified. If the paper gives "
        "no worked example, construct one whose answer is known independently "
        "and say where it came from.",
        "Code that fails to compile has not produced a wrong answer -- it has "
        "failed earlier, and the fix is to the build, not to the algorithm. "
        "Do not go looking for a bug in the method until it runs.",
    ]
