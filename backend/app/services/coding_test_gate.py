"""Running a repository's tests as a gate, and reading the result.

`run_command` already runs anything in a workspace, and its evidence is a
`command_result`: an exit code and some output. That is enough to *run* tests
and not enough to *gate* on them. A stage that must not proceed unless the
tests pass needs to say so in its contract, and a contract can only require
evidence that carries the answer — how many ran, how many failed, which ones.

An exit code alone also lies in a specific way that matters here. A test
command that fails to start — no pytest installed, wrong directory, syntax
error in the harness — exits non-zero exactly like a test failure, and a gate
that reads only the code cannot tell "your change broke three tests" from "the
tests never ran". Those call for opposite responses, so this distinguishes
them: `ran` is false when no test was collected, whatever the exit code said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestOutcome:
    """What a test run actually established."""

    #: Did the harness run tests at all? False for a command that never
    #: collected one, which is not the same as tests failing.
    ran: bool
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    #: Names where the harness reported them, so a gate can say which.
    failing: List[str] = field(default_factory=list)
    framework: str = "unknown"
    exit_code: Optional[int] = None
    #: Why this could not be read as a test run, when `ran` is false.
    note: str = ""

    @property
    def green(self) -> bool:
        """The gate condition: tests ran, and none of them failed."""
        return self.ran and self.failed == 0 and self.errors == 0

    def as_evidence(self) -> Dict:
        return {
            "ran": self.ran,
            "green": self.green,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "failing": self.failing[:20],
            "framework": self.framework,
            "exit_code": self.exit_code,
            "note": self.note,
        }


#: pytest's summary line, e.g. "3 failed, 20 passed, 1 skipped in 4.2s".
_PYTEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed)")
_PYTEST_FAIL_LINE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)
_PYTEST_COLLECTED = re.compile(r"collected\s+(\d+)\s+item")

#: jest / vitest, e.g. "Tests:       2 failed, 15 passed, 17 total".
_JEST_LINE = re.compile(r"^Tests:\s+(.+)$", re.MULTILINE)
_JEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|skipped|todo)")
_JEST_FAIL_LINE = re.compile(r"^\s*●\s+(.+?)$", re.MULTILINE)

#: go test, which prints one line per failing test.
_GO_FAIL = re.compile(r"^---\s+FAIL:\s+(\S+)", re.MULTILINE)
_GO_OK = re.compile(r"^(ok|PASS)\b", re.MULTILINE)


def read_test_output(stdout: str, stderr: str, exit_code: Optional[int]) -> TestOutcome:
    """Read a test run's output into something a gate can decide on.

    Deliberately several small parsers rather than one clever one: the
    frameworks disagree about everything except that they print a summary, and
    a regex general enough to match all three matches nonsense as well.
    """
    text = f"{stdout}\n{stderr}"

    jest = _JEST_LINE.search(text)
    if jest:
        counts = {kind: int(n) for n, kind in _JEST_COUNT.findall(jest.group(1))}
        return TestOutcome(
            ran=True,
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            skipped=counts.get("skipped", 0) + counts.get("todo", 0),
            failing=[m.strip() for m in _JEST_FAIL_LINE.findall(text)][:20],
            framework="jest",
            exit_code=exit_code,
        )

    if _PYTEST_COUNT.search(text) or _PYTEST_COLLECTED.search(text):
        counts: Dict[str, int] = {}
        for n, kind in _PYTEST_COUNT.findall(text):
            key = "errors" if kind.startswith("error") else kind
            counts[key] = counts.get(key, 0) + int(n)
        collected = _PYTEST_COLLECTED.search(text)
        total = int(collected.group(1)) if collected else 0
        ran = bool(counts) or total > 0
        return TestOutcome(
            ran=ran,
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            errors=counts.get("errors", 0),
            skipped=counts.get("skipped", 0) + counts.get("xfailed", 0),
            failing=[m for m in _PYTEST_FAIL_LINE.findall(text)][:20],
            framework="pytest",
            exit_code=exit_code,
            note="" if ran else "pytest produced no summary",
        )

    go_failures = _GO_FAIL.findall(text)
    if go_failures or _GO_OK.search(text):
        return TestOutcome(
            ran=True,
            failed=len(go_failures),
            failing=go_failures[:20],
            framework="go",
            exit_code=exit_code,
        )

    # Nothing recognised. An exit code on its own cannot tell a broken harness
    # from a failing test, so this refuses to guess: `ran` is false and the
    # gate treats it as "the tests did not run", which is the safe reading.
    return TestOutcome(
        ran=False,
        exit_code=exit_code,
        note=(
            "No test summary recognised in the output. An exit code alone "
            "cannot distinguish a failing test from a harness that never ran, "
            "so this is reported as not-run rather than guessed at."
        ),
    )


#: The command each ecosystem uses, tried in order of the marker file found.
#: Only used when the caller does not name a command themselves.
DEFAULT_TEST_COMMANDS = (
    ("pytest.ini", "pytest -q"),
    ("pyproject.toml", "pytest -q"),
    ("setup.cfg", "pytest -q"),
    ("package.json", "npm test --silent"),
    ("go.mod", "go test ./..."),
    ("Cargo.toml", "cargo test"),
)
