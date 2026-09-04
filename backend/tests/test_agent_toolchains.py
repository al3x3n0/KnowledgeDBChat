"""The build recipes the checker and the benchmark both use.

This table exists because two tools have to compile the same source the same
way. If they disagreed -- a different optimisation level, a different linker --
the binary that was verified would not be the binary that was timed, and the
correctness check would be certifying a program nobody measured.
"""

import pytest

from app.services import agent_toolchains as tc

pytestmark = pytest.mark.unit


class TestResolving:
    def test_the_languages_that_exist(self):
        assert tc.resolve("c").language == "c"
        assert tc.resolve("rust").language == "rust"

    def test_spellings_a_model_will_actually_use(self):
        for spelling in ("rust", "Rust", " RUST ", "rs", ".rs"):
            assert tc.resolve(spelling).language == "rust", spelling
        for spelling in ("c", "C", "c99", ".c"):
            assert tc.resolve(spelling).language == "c", spelling

    def test_nothing_named_falls_back_to_c(self):
        # Back-compatible: every existing caller passes no language at all.
        assert tc.resolve(None).language == "c"
        assert tc.resolve("").language == "c"

    def test_an_unknown_language_is_refused_not_guessed(self):
        # The important one. Compiling Rust as C yields a wall of syntax errors
        # that read as the model having written bad code, and the run then
        # burns its iterations rewriting source that was already correct.
        assert tc.resolve("go") is None
        assert tc.resolve("python") is None

    def test_the_refusal_says_what_is_available(self):
        message = tc.unsupported_language("go")
        assert "'go'" in message
        assert "c" in message and "rust" in message
        # And why a language cannot simply be added at run time.
        assert "no network" in message


class TestTheBuildLines:
    def test_c_builds_with_clang(self):
        chain = tc.resolve("c")
        assert tc.compile_command(chain, "-O2") == "clang -O2 -o prog prog.c"

    def test_rust_builds_with_rustc(self):
        chain = tc.resolve("rust")
        command = tc.compile_command(chain, "-O")
        assert command.startswith("rustc -O")
        assert "prog.rs" in command

    def test_rust_pins_its_linker(self):
        # The sandbox base carries clang but no gcc, and rustc invokes `cc` by
        # default. Without this every Rust build dies at link time with an
        # error that reads as a broken toolchain.
        command = tc.compile_command(tc.resolve("rust"), "-O")
        assert "-C linker=clang" in command

    def test_the_linker_survives_a_caller_overriding_flags(self):
        # It lives in the template rather than the default flags precisely so
        # a caller cannot drop it by passing their own.
        command = tc.compile_command(tc.resolve("rust"), "-C opt-level=3")
        assert "-C linker=clang" in command

    def test_both_languages_produce_the_same_executable_name(self):
        # The run script says `./prog`; a language whose build wrote something
        # else would compile fine and then appear to produce no output.
        for language in tc.SUPPORTED:
            chain = tc.resolve(language)
            assert "-o prog " in tc.compile_command(chain, chain.default_flags)

    def test_only_flags_are_substituted(self):
        # The template is not a general format string: a stray {} in a
        # caller's flags must not reach into it.
        chain = tc.resolve("c")
        assert tc.compile_command(chain, "-O2") == "clang -O2 -o prog prog.c"


class TestDefaultsThatChangeTheNumber:
    def test_rust_defaults_to_optimised(self):
        # rustc's own default is a debug build that runs several times slower.
        # A benchmark taking that default measures the absence of optimisation
        # and reports it as the algorithm's speed.
        assert tc.resolve("rust").default_flags == "-O"

    def test_c_keeps_the_default_it_always_had(self):
        assert tc.resolve("c").default_flags == "-O2"

    def test_rust_flags_warn_about_the_flag_that_looks_right(self):
        # -O2 is the obvious guess from the C side and rustc rejects it.
        assert "-O2" in tc.resolve("rust").flags_hint
