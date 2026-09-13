"""Python as a third language, and the floor a Python timing sits on.

A paper's reference implementation is usually Python, and the reproduce-a-paper
chain could not run one at all: `agent_toolchains` carried C and Rust. Adding it
is a table entry, but an interpreted language brings a measurement problem the
compiled ones do not have -- `./prog` is timed as a whole process, so every
Python result includes CPython starting up.
"""

import pytest

from app.services import agent_compiler_sandbox as sandbox
from app.services import agent_toolchains


class TestPythonIsBuildable:
    def test_it_resolves_by_name_and_by_the_names_a_model_might_use(self):
        for name in ("python", "py", ".py", "python3", "PYTHON"):
            chain = agent_toolchains.resolve(name)
            assert chain is not None, name
            assert chain.language == "python"

    def test_the_build_script_checks_syntax_and_produces_prog(self):
        chain = agent_toolchains.resolve("python")
        script = agent_toolchains.build_script(chain, chain.default_flags)

        # The syntax check is the analogue of compiling: without it a syntax
        # error surfaces as a failed reference case, and check_implementation
        # exists precisely to separate "wrong" from "does not build".
        assert "py_compile" in script
        # Everything downstream runs ./prog -- both the timing loop and the
        # correctness check -- so a language that launched differently would
        # have to be special-cased in both.
        assert "> prog" in script and "chmod +x prog" in script

    def test_the_group_braces_survive_formatting(self):
        """`{{` is not shell grouping, and `str.format` is what makes it `{`.

        Without the group, the caller's `2>compile_err.txt` binds to the last
        command of the && chain -- the chmod -- so a syntax error would be
        reported from an empty file.
        """
        chain = agent_toolchains.resolve("python")
        script = agent_toolchains.build_script(chain, "")

        assert script.startswith("{ ") and script.rstrip().endswith("; }")
        assert "{{" not in script and "}}" not in script

    def test_it_defaults_to_no_flags(self):
        """`-O` is the flag that looks like optimisation and is not: it strips
        assert statements, so a self-checking reference implementation would
        quietly stop checking itself."""
        chain = agent_toolchains.resolve("python")

        assert chain.default_flags == ""
        assert "assert" in chain.flags_hint

    def test_a_compiled_language_declares_no_startup_probe(self):
        for name in ("c", "rust"):
            assert agent_toolchains.resolve(name).startup_probe == ""
        assert agent_toolchains.resolve("python").startup_probe


class TestAPythonTimingCarriesItsFloor:
    """Measured in the sandbox: a 15 ms Python run was 8 ms of CPython
    starting up. Reported as the algorithm's cost, that number would compare
    two implementations on their shared startup."""

    def test_the_floor_is_reported_with_its_share(self):
        quality = sandbox.measurement_quality(
            load_average=0.5, cpu_count=8, timings=[100, 105, 110], startup_ms=10
        )

        assert quality["interpreter_startup_ms"] == 10
        assert quality["startup_share"] == pytest.approx(0.1, abs=0.001)

    def test_a_timing_that_is_mostly_startup_is_called_out(self):
        quality = sandbox.measurement_quality(
            load_average=0.1, cpu_count=8, timings=[15, 16, 17], startup_ms=8
        )

        warning = quality.get("measurement_warning") or ""
        assert "interpreter starting up" in warning
        # The numbers, not just the verdict: a reader has to be able to see how
        # much room was left for the algorithm.
        assert "8 ms of 15 ms" in warning

    def test_real_work_is_not_warned_about(self):
        """The control. A floor that flagged every Python run would be noise,
        and noise gets ignored exactly when it matters."""
        quality = sandbox.measurement_quality(
            load_average=0.1, cpu_count=8, timings=[1000, 1010, 1005], startup_ms=9
        )

        assert quality["startup_share"] < 0.05
        assert "interpreter" not in (quality.get("measurement_warning") or "")

    def test_a_compiled_language_reports_no_floor_at_all(self):
        """Not zero -- absent. A reported floor of 0 ms is a claim that the
        measurement was taken and came back free; there was no measurement."""
        quality = sandbox.measurement_quality(
            load_average=0.1, cpu_count=8, timings=[100, 101], startup_ms=None
        )

        assert "interpreter_startup_ms" not in quality
        assert "startup_share" not in quality


class TestTheSlimImageIsRegisteredEverywhereItMustBe:
    """A sandbox image is unusable unless three separate places know it.

    Registration is spread across the allowlist, the provenance map and the
    build target, and a miss in any one of them fails at a different layer:
    the runner refuses the image, an evidence bundle pins an id nobody can
    rebuild, or `make sandbox-images` quietly stops building it.
    """

    IMAGE = "ghcr.io/al3x3n0/kdbc-polyglot-slim:latest"

    def test_the_runner_will_accept_it(self):
        from app.core.config import settings

        allowed = settings.SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES
        assert self.IMAGE in allowed

    def test_a_bundle_can_say_where_it_came_from(self):
        from app.services.agent_evidence_bundle import IMAGE_ORIGINS

        origin = IMAGE_ORIGINS.get("kdbc-polyglot-slim")
        assert origin, "a pinned image id nobody can rebuild is an opaque hash"
        assert origin["dockerfile"].endswith("polyglot-slim/Dockerfile")

    def test_the_default_image_is_still_the_one_with_the_crates(self):
        """Switching the default here would silently remove rand and rayon
        from every Rust run -- the compile line degrades to no crates rather
        than failing, so it would look like the model misusing an import."""
        from app.services.agent_compiler_sandbox import DEFAULT_IMAGE

        assert DEFAULT_IMAGE.endswith("kdbc-compiler-research:latest")


class TestTheSchemaOffersWhatTheTableBuilds:
    def test_every_language_a_tool_offers_can_actually_be_built(self):
        """The enum used to be written out by hand as ["c", "rust"].

        A language the table builds but the schema does not offer is
        unreachable: the model is refused for naming it. One the schema offers
        and the table cannot build fails at compile time with an error blaming
        the code.
        """
        from app.agent_core.tool_specs.measurement import SPECS

        checked = 0
        for spec in SPECS:
            prop = (spec.parameters.get("properties") or {}).get("language")
            if not isinstance(prop, dict) or "enum" not in prop:
                continue
            checked += 1
            assert list(prop["enum"]) == list(agent_toolchains.SUPPORTED), spec.name
        assert checked >= 2, "expected the benchmark and the correctness check"


class TestTheComparisonRefusesAStartupTiming:
    """The benchmark reports the floor; this is the half that reads it back.

    Scoring a mostly-startup timing against a paper compares that paper's
    algorithm with this machine's process launch -- the same mistake the busy-
    host and unstable-spread concerns already refuse on their own axes.
    """

    def test_a_startup_dominated_timing_cannot_settle_a_claim(self):
        from app.services import agent_claim_comparison as comparison

        concerns = comparison.measurement_concerns(
            {"startup_share": 0.53, "interpreter_startup_ms": 8, "fastest_ms": 15}
        )

        assert concerns
        assert "interpreter starting up" in concerns[0]
        assert "8 ms of 15 ms" in concerns[0]

    def test_a_timing_with_room_to_measure_is_accepted(self):
        from app.services import agent_claim_comparison as comparison

        assert (
            comparison.measurement_concerns(
                {"startup_share": 0.04, "interpreter_startup_ms": 9, "fastest_ms": 240}
            )
            == []
        )

    def test_a_compiled_language_is_unaffected(self):
        from app.services import agent_claim_comparison as comparison

        assert comparison.measurement_concerns({"fastest_ms": 8}) == []

    def test_the_two_thresholds_are_one_number(self):
        """Two copies of a judgement call drift, and the drift shows up as a
        benchmark that warns while the comparison accepts it anyway."""
        from app.services import agent_claim_comparison as comparison
        from app.services import agent_compiler_sandbox as sandbox

        assert comparison.STARTUP_DOMINATES == sandbox.STARTUP_DOMINATES
