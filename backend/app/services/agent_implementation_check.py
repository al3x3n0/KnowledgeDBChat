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

A handful of cases is a weak check, and a run left to write its own check
writes one that passes: a live stage's harness printed PASS whatever its
input. So a check may also name an **oracle** -- a reference program,
obviously correct and as slow as it likes (plain ``%`` for a fast remainder,
a Python loop for a SIMD kernel) -- and a **generator** that prints one input
per index. Every generated input goes to both programs and the outputs must
agree. This is chia's rule for its RISC-V flow (ucb-bar/chia: Spike runs
inside the simulator and "the golden model *is* the check"), adapted: no
expected output is written by hand, so none can be written to match the
code. Three ways it could still check nothing are refused: a reference that
is the candidate itself, a generator that prints one input whatever the
index, and a reference that fails on every input.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.services import agent_toolchains

logger = logging.getLogger(__name__)

#: How far a numeric output may drift from the reference and still be the same
#: answer. An algorithm rebuilt from a paper's prose reassociates sums and
#: picks its own constants; it will not match bit for bit, and requiring that
#: rejects correct work.
DEFAULT_TOLERANCE = 1e-6

#: A cap on cases per check. Each one is a container round trip, and a run that
#: wants a thousand cases wants a test suite, not a gate.
MAX_CASES = 25

#: Generated inputs when the caller names no count, and the most it may ask
#: for. They all run in the one container the cases use, so the cost is
#: process starts, not round trips.
DEFAULT_GENERATED_CASES = 100
MAX_GENERATED_CASES = 1000

#: What each generated input and each output is cut to before it leaves the
#: sandbox: a thousand of them have to fit in one captured stdout.
_ORACLE_OUTPUT_BYTES = 4096

#: Disagreements reported with their inputs; the rest are counted.
_DISAGREEMENTS_SHOWN = 3

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
class OracleProgram:
    """A program the check builds beside the candidate: a reference or a generator."""

    code: str
    language: str = agent_toolchains.DEFAULT_LANGUAGE


@dataclass
class OracleResult:
    """What comparing the candidate with a reference on generated inputs found."""

    requested: int
    reference_language: str = agent_toolchains.DEFAULT_LANGUAGE
    #: Inputs the generator printed before it stopped or reached `requested`.
    generated: int = 0
    distinct_inputs: int = 0
    #: Inputs the reference itself failed on. They test nothing about the
    #: candidate, so they are counted apart rather than against it.
    reference_failed: int = 0
    agreed: int = 0
    disagreed: int = 0
    disagreements: List[Dict[str, str]] = field(default_factory=list)
    generator_error: str = ""

    @property
    def compared(self) -> int:
        return self.agreed + self.disagreed

    @property
    def passed(self) -> bool:
        """Agreement on every input, over inputs that actually vary.

        One distinct input is one case, however many times it was run; a
        generator that ignores its index would otherwise turn a single
        example into "100 agreed".
        """
        return self.compared > 0 and self.disagreed == 0 and self.distinct_inputs >= 2

    def as_evidence(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "reference_language": self.reference_language,
            "requested": self.requested,
            "generated": self.generated,
            "distinct_inputs": self.distinct_inputs,
            "compared": self.compared,
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "reference_failed": self.reference_failed,
            "disagreements": self.disagreements[:_DISAGREEMENTS_SHOWN],
            "generator_error": self.generator_error,
        }


@dataclass
class ImplementationCheck:
    """Whether an implementation may be trusted enough to be worth timing."""

    #: Did the code compile and execute at all?
    ran: bool
    cases: List[CaseResult] = field(default_factory=list)
    #: The comparison with a reference program, when the caller named one.
    oracle: Optional[OracleResult] = None
    compile_error: str = ""
    note: str = ""
    #: Which kind of not-run this was, so a caller can tell a mistake it made
    #: from a fact about the code. "no_cases" is the caller supplying nothing
    #: to check against; "compile_error" and "sandbox" are not its fault, and
    #: the first is a real finding about the implementation.
    reason: str = ""
    #: Which toolchain built it. Carried into the evidence because a later
    #: stage reading "verified" needs to know what was verified: a Rust
    #: implementation and a C one are different programs with different
    #: performance, and a benchmark stage that assumes the wrong one compiles
    #: the source as the wrong language.
    language: str = agent_toolchains.DEFAULT_LANGUAGE

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

        With an oracle the directed cases may be empty, and both halves must
        hold when both are given: agreeing with the reference does not excuse
        a worked example from the paper that fails.
        """
        if not self.ran or not (self.cases or self.oracle):
            return False
        if self.oracle is not None and not self.oracle.passed:
            return False
        return self.cases_run == len(self.cases) and self.cases_passed == len(
            self.cases
        )

    def as_evidence(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "ran": self.ran,
            "language": self.language,
            "cases_run": self.cases_run,
            "cases_passed": self.cases_passed,
            # At the top level so a contract can bound them
            # (`validity.bounds`: generated_cases >= 100).
            "oracle_checked": self.oracle is not None,
            "generated_cases": self.oracle.compared if self.oracle else 0,
            "oracle": self.oracle.as_evidence() if self.oracle else None,
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
            "reason": self.reason,
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


#: What a caller may call the expected output. `expected_output` is the name,
#: and the rest are what callers actually write -- measured: a run supplied two
#: cases keyed `expected`, had both dropped as unusable, and repeated the same
#: call three times. The same vocabulary allowance `EMIT_ALIASES` makes for
#: "assembly", for the same reason: rejecting a synonym costs an iteration to
#: learn a naming difference, and teaches nothing about the code.
EXPECTED_OUTPUT_KEYS = ("expected_output", "expected", "output", "expects")


def _expected_of(case: Dict[str, Any]) -> Any:
    for key in EXPECTED_OUTPUT_KEYS:
        value = case.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def describe_unusable_cases(cases: Any) -> str:
    """Why the supplied cases could not be used, in terms of what was sent.

    "No usable reference cases were supplied" is true and unhelpful when two
    were: it names the key the tool wanted without naming the keys it got, so
    the caller cannot see which half of the disagreement to change.
    """
    if not isinstance(cases, (list, tuple)) or not cases:
        return ""
    keys: List[str] = []
    for case in cases:
        if isinstance(case, dict):
            keys.extend(str(k) for k in case.keys())
    if not keys:
        return f"{len(cases)} case(s) were supplied but none was an object."
    seen = ", ".join(sorted(set(keys)))
    return (
        f"{len(cases)} case(s) were supplied and none carried a non-empty "
        f"expected output. They had: {seen}. Name it one of: "
        f"{', '.join(EXPECTED_OUTPUT_KEYS)}."
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
        expected = _expected_of(case)
        if expected is None:
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


async def check_implementation(
    *,
    code: str,
    cases: Sequence[Dict[str, str]],
    language: str = agent_toolchains.DEFAULT_LANGUAGE,
    flags: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
    image: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    reference: Optional[OracleProgram] = None,
    generator: Optional[OracleProgram] = None,
    generated_cases: int = DEFAULT_GENERATED_CASES,
) -> ImplementationCheck:
    """Compile a program once and run it against every reference case.

    The source checked here is the source that gets benchmarked: verifying one
    program and timing a different one proves nothing about the number, and
    keeping them the same string is the only way to know they are the same
    program. The build recipe comes from `agent_toolchains` for the same
    reason -- the checker and the benchmark have to compile it identically, or
    the binary that was verified is not the binary that was timed.

    With `reference` and `generator`, every generated input is also run
    through both programs and their outputs compared (see the module
    docstring). The oracle needs both: a reference with nothing to run on
    checks nothing, and generated inputs with no reference have no answer.
    """
    from app.services import agent_compiler_sandbox as sandbox

    chain = agent_toolchains.resolve(language)
    if chain is None:
        # Not a fallback to C. Compiling Rust as C yields a wall of syntax
        # errors that read as the model having written bad code, and the run
        # then spends its iterations rewriting source that was already correct.
        return ImplementationCheck(
            ran=False,
            reason="bad_language",
            note=agent_toolchains.unsupported_language(language),
        )

    resolved_image = image or sandbox.DEFAULT_IMAGE
    resolved_timeout = timeout_seconds or sandbox.DEFAULT_TIMEOUT_SECONDS

    oracle_chains = None
    if reference is not None or generator is not None:
        refused = _refuse_oracle(code, reference, generator)
        if refused is not None:
            refused.language = chain.language
            return refused
        oracle_chains = (
            agent_toolchains.resolve(reference.language),
            agent_toolchains.resolve(generator.language),
        )
        for program, oracle_chain in zip((reference, generator), oracle_chains):
            if oracle_chain is None:
                return ImplementationCheck(
                    ran=False,
                    language=chain.language,
                    reason="bad_language",
                    note=agent_toolchains.unsupported_language(program.language),
                )

    normalized = normalize_cases(cases)
    if not normalized and oracle_chains is None:
        return ImplementationCheck(
            ran=False,
            language=chain.language,
            reason="no_cases",
            note=(
                "No usable reference cases were supplied. A correctness check "
                "with no cases passes vacuously, so this reports as unverified "
                "rather than letting untested code through. Give at least one "
                "case with an expected_output, ideally a worked example from "
                "the paper itself -- or a reference implementation and a "
                "generator, which check it on many inputs with no expected "
                "output written by hand."
                + (f" {describe_unusable_cases(cases)}" if cases else "")
            ),
        )

    for source in [code] + (
        [reference.code, generator.code] if oracle_chains is not None else []
    ):
        blocked = sandbox._preflight(source, resolved_image)
        if blocked:
            return ImplementationCheck(
                ran=False,
                language=chain.language,
                reason="sandbox",
                note=str(blocked.get("error") or "sandbox unavailable"),
            )

    safe_flags = sandbox._clean_flags(flags or chain.default_flags)
    if safe_flags is None:
        return ImplementationCheck(
            ran=False,
            language=chain.language,
            reason="bad_flags",
            note=f"flags contain unsupported characters: {flags!r}",
        )
    # After sanitising, never before: what is added here is ours and known
    # safe, and running it through the caller's filter could only reject it.
    safe_flags = agent_toolchains.enforce(chain, safe_flags)

    with tempfile.TemporaryDirectory(prefix="impl_check_") as workdir:
        Path(workdir, chain.source_file).write_text(code, encoding="utf-8")
        build = agent_toolchains.build_script(chain, safe_flags)
        script_parts = [
            f"{build} 2>compile_err.txt || " "{ cat compile_err.txt >&2; exit 90; }"
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
                f"./prog <case_{index}.in; "
                f'echo "__case_exit__ {index} $?"'
            )
        count = max(1, min(int(generated_cases or 0), MAX_GENERATED_CASES))
        if oracle_chains is not None:
            for directory, program, oracle_chain in zip(
                (_REFERENCE_DIR, _GENERATOR_DIR), (reference, generator), oracle_chains
            ):
                Path(workdir, directory).mkdir()
                Path(workdir, directory, oracle_chain.source_file).write_text(
                    program.code, encoding="utf-8"
                )
            script_parts.append(_oracle_script(oracle_chains, count))
        script = "; ".join(script_parts)

        try:
            returncode, stdout, stderr = await sandbox._run(
                script, workdir, image=resolved_image, timeout_seconds=resolved_timeout
            )
        except Exception as exc:
            # `str(exc)` is empty for several exceptions worth distinguishing --
            # asyncio.TimeoutError most of all, which is what a sandbox run
            # produces on a loaded machine. Measured live: this path reported
            # `note="Check failed: "` with `reason=""` and `compile_error=""`,
            # so a run had no way to tell a timeout from a broken toolchain
            # from its own bad code. The class name is there even when the
            # message is not.
            detail = str(exc).strip() or exc.__class__.__name__
            logger.warning(f"check_implementation failed: {detail}")
            return ImplementationCheck(
                ran=False,
                language=chain.language,
                reason="error",
                note=(
                    f"Check failed before any case ran ({detail}). That says "
                    "nothing about the implementation -- the sandbox did not "
                    "finish. Retry; if it fails the same way, the toolchain is "
                    "the suspect rather than the code."
                ),
            )

    if returncode == 90:
        return ImplementationCheck(
            ran=False,
            language=chain.language,
            reason="compile_error",
            compile_error=sandbox.explain_compiler_failure(stderr),
            note=(
                "The implementation did not compile, so its correctness is "
                "unknown. This is a different failure from a wrong answer: fix "
                "the build before reading anything into the algorithm."
            ),
        )
    for exit_code, which in ((91, "reference"), (92, "generator")):
        if returncode == exit_code:
            return ImplementationCheck(
                ran=False,
                language=chain.language,
                reason=f"{which}_compile_error",
                compile_error=sandbox.explain_compiler_failure(stderr),
                note=(
                    f"The {which} did not compile, so nothing was compared. "
                    "This is about the check, not the implementation."
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

    oracle = None
    if oracle_chains is not None:
        oracle = _read_oracle(stdout, count, oracle_chains[0].language, tolerance)
    return ImplementationCheck(
        ran=True, cases=results, oracle=oracle, language=chain.language
    )


#: Where the reference and the generator are built: each in its own
#: directory, because every toolchain builds `./prog` from a fixed file name,
#: and run from there, because the Python shim finds its source relative to
#: the directory it runs in.
_REFERENCE_DIR = "oracle_reference"
_GENERATOR_DIR = "oracle_generator"
_ORACLE_MARK = "__oracle__"


def _same_program(a: str, b: str) -> bool:
    return " ".join((a or "").split()) == " ".join((b or "").split())


def _refuse_oracle(
    code: str,
    reference: Optional[OracleProgram],
    generator: Optional[OracleProgram],
) -> Optional[ImplementationCheck]:
    """The oracle calls that could only check nothing, refused before running."""
    if reference is None or generator is None:
        missing = "generator" if reference is not None else "reference"
        return ImplementationCheck(
            ran=False,
            reason="oracle_incomplete",
            note=(
                f"An oracle needs a reference and a generator; the {missing} is "
                "missing. A reference with no inputs checks nothing, and "
                "generated inputs with no reference have no answer to compare."
            ),
        )
    if not str(reference.code or "").strip() or not str(generator.code or "").strip():
        return ImplementationCheck(
            ran=False,
            reason="oracle_incomplete",
            note="The reference and the generator each need source code.",
        )
    if _same_program(code, reference.code):
        return ImplementationCheck(
            ran=False,
            reason="reference_is_candidate",
            note=(
                "The reference is the implementation itself, which agrees with "
                "itself on every input. Write the reference the obvious way "
                "(plain `%`, a scalar loop, a direct transcription of the "
                "definition), however slow."
            ),
        )
    return None


def _oracle_script(chains, count: int) -> str:
    """Build the reference and the generator, then compare on `count` inputs.

    Exit 91 and 92 are the two builds failing, as 90 is the candidate's.
    Each section of the output is marked with what it is and the index and
    exit status it belongs to, and cut to `_ORACLE_OUTPUT_BYTES`.
    """
    reference_chain, generator_chain = chains
    parts = []
    for directory, oracle_chain, failed in (
        (_REFERENCE_DIR, reference_chain, 91),
        (_GENERATOR_DIR, generator_chain, 92),
    ):
        build = agent_toolchains.build_script(
            oracle_chain,
            agent_toolchains.enforce(oracle_chain, oracle_chain.default_flags),
        )
        parts.append(
            f"( cd {directory} && {{ {build}; }} ) 2>oracle_err.txt || "
            f"{{ cat oracle_err.txt >&2; exit {failed}; }}"
        )
    cut = _ORACLE_OUTPUT_BYTES
    mark = _ORACLE_MARK
    parts.append(
        f"i=0; while [ $i -lt {count} ]; do "
        f"echo $i | ( cd {_GENERATOR_DIR} && ./prog ) >oracle.in 2>/dev/null; g=$?; "
        f'if [ $g -ne 0 ]; then echo "{mark} gen $i $g"; break; fi; '
        f"( cd {_REFERENCE_DIR} && ./prog ) <oracle.in >oracle.ref 2>/dev/null; r=$?; "
        "./prog <oracle.in >oracle.out 2>/dev/null; p=$?; "
        f'echo "{mark} in $i 0"; head -c {cut} oracle.in; echo; '
        f'echo "{mark} ref $i $r"; head -c {cut} oracle.ref; echo; '
        f'echo "{mark} out $i $p"; head -c {cut} oracle.out; echo; '
        "i=$((i+1)); done"
    )
    return "; ".join(parts)


def _read_oracle(
    stdout: str, requested: int, reference_language: str, tolerance: float
) -> OracleResult:
    """Read the marked sections back and compare each input's two outputs."""
    sections: Dict[tuple, tuple] = {}
    current: Optional[tuple] = None
    buffer: List[str] = []
    result = OracleResult(requested=requested, reference_language=reference_language)

    def close() -> None:
        if current is not None:
            kind, index, code = current
            sections[(kind, index)] = ("\n".join(buffer), code)

    for line in (stdout or "").splitlines():
        if line.startswith(_ORACLE_MARK + " "):
            close()
            parts = line.split()
            index, code = _int_or_none(parts[2:3]), _int_or_none(parts[3:4])
            kind = parts[1] if len(parts) > 1 else ""
            current = None
            buffer = []
            if kind == "gen" and index is not None:
                result.generator_error = (
                    f"The generator exited {code} on index {index}, so "
                    f"{index} of {requested} inputs were generated."
                )
                continue
            if index is not None:
                current = (kind, index, code)
            continue
        if current is not None:
            buffer.append(line)
    close()

    inputs = set()
    for index in range(requested):
        if ("in", index) not in sections:
            break
        given, _ = sections[("in", index)]
        expected, reference_exit = sections.get(("ref", index), ("", None))
        actual, candidate_exit = sections.get(("out", index), ("", None))
        result.generated += 1
        inputs.add(given)
        if reference_exit != 0:
            result.reference_failed += 1
            continue
        if candidate_exit != 0:
            outcome = CaseResult(
                name="",
                passed=False,
                expected=expected[:400],
                actual=actual[:400],
                detail=f"The program exited {candidate_exit} on this input",
            )
        else:
            outcome = compare_output(actual, expected, tolerance)
        if outcome.passed:
            result.agreed += 1
            continue
        result.disagreed += 1
        if len(result.disagreements) < _DISAGREEMENTS_SHOWN:
            result.disagreements.append(
                {
                    "index": str(index),
                    "input": given[:400],
                    "expected": outcome.expected,
                    "actual": outcome.actual,
                    "detail": outcome.detail,
                }
            )
    result.distinct_inputs = len(inputs)
    return result


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
        "Prefer an oracle to hand-written cases: a plainly correct reference "
        "(the definition, however slow) and a generator of many inputs, edge "
        "cases first. A handful of cases chosen by the author of the code "
        "tends to be the handful it passes.",
        "Code that fails to compile has not produced a wrong answer -- it has "
        "failed earlier, and the fix is to the build, not to the algorithm. "
        "Do not go looking for a bug in the method until it runs.",
    ]
