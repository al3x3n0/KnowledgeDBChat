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


class TestRustCrates:
    """Crates without a network: built into the image, linked as rlibs."""

    def test_the_list_is_pinned_exactly(self):
        # A caret range makes the dependency graph a function of the build
        # date, which is the opposite of what a reproduction study needs.
        for crate in tc.RUST_CRATES:
            parts = crate.version.split(".")
            assert len(parts) == 3, f"{crate.name} is not pinned to a patch version"
            assert all(p.isdigit() for p in parts), crate.version

    def test_cargo_names_become_use_names(self):
        # `num-traits` on crates.io is `num_traits` in a use statement, and an
        # --extern under the wrong name silently makes the crate invisible.
        by_name = {c.name: c for c in tc.RUST_CRATES}
        assert by_name["num-traits"].extern_name == "num_traits"
        assert by_name["rand"].extern_name == "rand"

    def test_a_seeded_generator_is_available(self):
        # Reproducibility needs a *named* PRNG seeded explicitly; the default
        # thread RNG gives different inputs on every run and on every machine.
        assert any(c.name == "rand_chacha" for c in tc.RUST_CRATES)

    def test_the_model_is_told_what_it_has(self):
        described = tc.describe_rust_crates()
        for crate in tc.RUST_CRATES:
            assert crate.name in described
            assert crate.version in described

    def test_the_compile_line_reads_the_externs_from_the_image(self):
        # Not a hard-coded list of --extern flags: that couples this code to
        # one image build and fails every Rust compile on any other, with an
        # error about a missing rlib that says nothing about the real cause.
        chain = tc.resolve("rust")
        script = tc.build_script(chain, "-O")
        assert tc.RUST_EXTERNS_FILE in script
        assert "$RUSTC_EXTERNS" in script

    def test_no_crates_still_compiles(self):
        # The fallback matters: an older image has no externs file, and Rust
        # without crates must still work rather than failing to build at all.
        chain = tc.resolve("rust")
        script = tc.build_script(chain, "-O")
        assert 'RUSTC_EXTERNS=""' in script

    def test_the_registry_loads_without_the_application(self):
        """The image builds its Cargo manifest by importing this module.

        That is what makes drift impossible -- one list, two consumers -- and
        it only works while the module imports nothing but the standard
        library. An `from app.core...` added here would break the image build
        with a traceback about a missing package, long after the change.

        Loaded the way the image loads it, including the sys.modules
        registration @dataclass needs, so this fails the same way the build
        would.
        """
        import importlib.util
        import sys
        from pathlib import Path

        source = Path(tc.__file__)
        spec = importlib.util.spec_from_file_location("standalone_toolchains", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules["standalone_toolchains"] = module
        try:
            spec.loader.exec_module(module)
            assert [c.name for c in module.RUST_CRATES] == [
                c.name for c in tc.RUST_CRATES
            ]
        finally:
            sys.modules.pop("standalone_toolchains", None)

    def test_the_generator_emits_a_manifest_for_every_crate(self):
        """The script the image runs, exercised here rather than at build time.

        A mistake in it surfaces as a failed image build otherwise, which is a
        slow and confusing place to find out.
        """
        import subprocess
        import sys
        from pathlib import Path

        generator = (
            Path(__file__).resolve().parents[2]
            / "deploy"
            / "sandbox-images"
            / "compiler-research"
            / "gen_crate_manifest.py"
        )
        if not generator.exists():  # pragma: no cover - backend-only checkout
            pytest.skip("sandbox image generator not present")

        manifest = subprocess.run(
            [sys.executable, str(generator), tc.__file__],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for crate in tc.RUST_CRATES:
            # Exactly pinned, in the form cargo wants.
            assert f'{crate.name} = "={crate.version}"' in manifest
        assert 'edition = "2021"' in manifest

        names = subprocess.run(
            [sys.executable, str(generator), tc.__file__, "--emit", "extern-names"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        # Every declared crate gets an --extern, under its `use` name. One
        # built but never named is invisible to a run.
        assert names == [c.extern_name for c in tc.RUST_CRATES]


class TestTheEditionTrap:
    """rustc defaults to edition 2015; nothing modern compiles there."""

    def test_the_edition_is_added_when_the_caller_says_nothing(self):
        chain = tc.resolve("rust")
        assert "--edition 2021" in tc.enforce(chain, "-O")

    def test_the_edition_survives_a_caller_overriding_flags(self):
        # A default would be dropped the moment a caller passes their own
        # flags, silently landing back on edition 2015.
        chain = tc.resolve("rust")
        assert "--edition 2021" in tc.enforce(chain, "-C opt-level=3")

    def test_a_caller_naming_an_edition_is_not_overruled(self):
        # rustc rejects a repeated --edition outright, so adding it blindly
        # would break exactly the callers who got it right.
        chain = tc.resolve("rust")
        result = tc.enforce(chain, "-O --edition 2018")
        assert result.count("--edition") == 1
        assert "2018" in result

    def test_c_gains_nothing(self):
        chain = tc.resolve("c")
        assert tc.enforce(chain, "-O2") == "-O2"
