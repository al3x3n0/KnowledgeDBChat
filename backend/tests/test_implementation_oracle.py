"""An implementation checked against a reference program on generated inputs.

The stand-in for the sandbox runs the real script the check writes, with
``/bin/sh`` in the same working directory, so the build steps, the loop and
the section markers are the ones a container runs. The programs are Python
because ``python3`` is on every machine the suite runs on; the toolchain is
the one the sandbox uses for Python.
"""

import asyncio

import pytest

from app.services import agent_compiler_sandbox as sandbox
from app.services import agent_implementation_check as impl

pytestmark = pytest.mark.unit

CANDIDATE = """
import sys
n, d = map(int, sys.stdin.read().split())
print(n % d)
"""

REFERENCE = """
import sys
a, b = sys.stdin.read().split()
a, b = int(a), int(b)
print(a - (a // b) * b)
"""

GENERATOR = """
import random, sys
i = int(sys.stdin.read())
r = random.Random(i)
print(r.randrange(0, 10**9), r.randrange(1, 10**6))
"""


def program(code):
    return impl.OracleProgram(code=code, language="python")


@pytest.fixture
def host_sandbox(monkeypatch):
    calls = []

    async def run(script, workdir, *, image, timeout_seconds):
        calls.append(script)
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            script,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(process.communicate(), timeout_seconds)
        return process.returncode, out.decode(), err.decode()

    monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
    monkeypatch.setattr(sandbox, "_allowed_images", lambda: {sandbox.DEFAULT_IMAGE})
    monkeypatch.setattr(sandbox, "_run", run)
    return calls


async def check(code=CANDIDATE, reference=REFERENCE, generator=GENERATOR, **kw):
    kw.setdefault("generated_cases", 30)
    return await impl.check_implementation(
        code=code,
        cases=kw.pop("cases", []),
        language="python",
        reference=program(reference) if reference is not None else None,
        generator=program(generator) if generator is not None else None,
        timeout_seconds=300,
        **kw,
    )


class TestAgreement:
    async def test_a_correct_implementation_agrees_on_every_input(self, host_sandbox):
        result = await check()

        assert result.verified, result.as_evidence()
        evidence = result.as_evidence()
        assert evidence["oracle_checked"] and evidence["generated_cases"] == 30
        assert evidence["oracle"]["distinct_inputs"] == 30

    async def test_one_wrong_input_in_thirty_is_found_with_its_input(
        self, host_sandbox
    ):
        wrong_on_odd_divisors = CANDIDATE.replace(
            "print(n % d)", "print(n % d if d % 2 == 0 else (n % d) + 1)"
        )

        result = await check(code=wrong_on_odd_divisors)

        assert not result.verified
        oracle = result.oracle
        assert oracle.disagreed > 0 and oracle.agreed > 0
        shown = oracle.disagreements[0]
        assert int(shown["input"].split()[1]) % 2 == 1
        assert "differs" in shown["detail"]

    async def test_a_harness_that_prints_pass_whatever_its_input_fails(
        self, host_sandbox
    ):
        """The shape a live stage wrote: a check that always passes."""
        result = await check(code='print("PASS")')

        assert not result.verified
        assert result.oracle.disagreed == 30

    async def test_a_hand_written_case_still_counts(self, host_sandbox):
        result = await check(
            cases=[{"name": "paper", "input": "10 3", "expected_output": "2"}]
        )

        assert not result.verified
        assert result.oracle.passed


class TestChecksThatCheckNothing:
    async def test_a_generator_ignoring_its_index_is_one_case(self, host_sandbox):
        result = await check(generator="print(1000, 7)")

        assert not result.verified
        assert result.oracle.distinct_inputs == 1 and result.oracle.disagreed == 0

    async def test_a_reference_failing_everywhere_compares_nothing(self, host_sandbox):
        result = await check(reference="raise SystemExit(3)")

        assert not result.verified
        assert result.oracle.reference_failed == 30 and result.oracle.compared == 0

    async def test_the_candidate_as_its_own_reference_is_refused_unrun(
        self, host_sandbox
    ):
        result = await check(reference="  " + CANDIDATE.replace("\n", "\n  ").strip())

        assert result.reason == "reference_is_candidate"
        assert host_sandbox == []

    async def test_a_reference_without_a_generator_is_refused_unrun(self, host_sandbox):
        result = await check(generator=None)

        assert result.reason == "oracle_incomplete"
        assert host_sandbox == []


class TestWhatTheyReport:
    async def test_a_generator_that_stops_early_says_how_far_it_got(self, host_sandbox):
        stops = (
            "import random, sys\n"
            "i = int(sys.stdin.read())\n"
            "if i >= 5:\n"
            "    raise SystemExit(4)\n"
            "r = random.Random(i)\n"
            "print(r.randrange(0, 10**9), r.randrange(1, 10**6))\n"
        )

        result = await check(generator=stops)

        assert result.oracle.generated == 5
        assert "index 5" in result.oracle.generator_error

    async def test_a_reference_that_does_not_build_is_the_check_s_fault(
        self, host_sandbox
    ):
        result = await check(reference="def broken(:\n")

        assert result.reason == "reference_compile_error"
        assert not result.verified

    async def test_the_count_is_capped(self, host_sandbox):
        # Stops after three, so the test does not run a thousand of them.
        stops_at_three = (
            "import sys\n"
            "i = int(sys.stdin.read())\n"
            "if i >= 3:\n"
            "    raise SystemExit(1)\n"
            "print(i + 1, 7)\n"
        )

        result = await check(generated_cases=10**6, generator=stops_at_three)

        assert result.oracle.requested == impl.MAX_GENERATED_CASES


class TestTheTool:
    def _handler(self):
        from app.services.agent_tool_providers import workspace_mutation

        executor = type("E", (), {})()
        provider = workspace_mutation.build_autonomous_workspace_mutation_provider(
            executor
        )
        return provider._handlers["check_implementation"]

    async def test_a_self_referential_oracle_is_an_error_so_repeats_escalate(
        self, host_sandbox
    ):
        result = await self._handler()(
            {
                "code": CANDIDATE,
                "language": "python",
                "reference": {"code": CANDIDATE, "language": "python"},
                "generator": {"code": GENERATOR, "language": "python"},
            },
            None,
        )

        assert "error" in result

    async def test_an_oracle_alone_is_enough_to_produce_the_finding(self, host_sandbox):
        result = await self._handler()(
            {
                "code": CANDIDATE,
                "language": "python",
                "reference": {"code": REFERENCE, "language": "python"},
                "generator": {"code": GENERATOR, "language": "python", "count": 12},
            },
            None,
        )

        (finding,) = result["findings"]
        assert finding["type"] == "implementation_verified"
        assert finding["verified"] is True and finding["generated_cases"] == 12
