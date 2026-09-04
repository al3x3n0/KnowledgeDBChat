"""How to build a self-contained program, per language.

Two tools need this and they must agree: the correctness check compiles a
program and runs it against reference cases, and the benchmark compiles the
same program and times it. If they built it differently -- different compiler,
different optimisation level, different source file name -- then the thing that
was verified and the thing that was measured would be two different binaries,
and the verification would say nothing about the timing. That is the failure
`check_implementation` exists to prevent, so the build recipe lives in one
place rather than being written out twice.

Adding a language is a table entry plus a sandbox image that carries its
compiler. The entry is deliberately small: a source filename, a command that
turns it into an executable called `prog`, and the default flags.

Rust reaches third-party crates, and the way it does so is worth stating,
because the obvious route is closed. The sandbox runs with `--network none`,
which is a security property rather than an oversight, so `cargo build` cannot
fetch anything and never will. Instead a pinned set of crates is compiled into
the image at BUILD time, where there is a network, and the resulting rlibs are
linked directly with `--extern`. A run gets real dependencies and still cannot
reach the network.

Which crates exist is decided by the image, not by this file. The compile line
reads the extern flags out of `/opt/rust-deps/externs.txt`, written when those
crates were built, and falls back to no crates when the file is absent. The
alternative -- naming each `--extern` here -- couples the code to one image
version and fails every Rust compile on an image built before or after it,
with an error about a missing rlib that says nothing about the real cause.
The list below is therefore documentation for the model and a drift check for
the tests, never the authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Toolchain:
    """Everything needed to turn one source string into one executable."""

    language: str
    #: What the source is written to inside the sandbox working directory.
    source_file: str
    #: Builds `source_file` into `./prog`. `{flags}` is substituted; nothing
    #: else is, so a caller cannot smuggle a command in through this template.
    compile_template: str
    #: Used when the caller names none.
    default_flags: str
    #: What the flags mean, for the tool description a model reads. The two
    #: languages spell optimisation differently and a model that assumes -O2
    #: works everywhere gets an unoptimised Rust binary and a timing that
    #: reflects the debug build rather than the algorithm.
    flags_hint: str
    #: Extensions a caller might name for this language, so a mislabelled
    #: request can be recognised rather than compiled as the wrong language.
    aliases: Tuple[str, ...] = ()
    #: Shell run before the compile, for a language that needs to discover
    #: something about the image it is running in. Ours, never the caller's:
    #: user flags are sanitised to exclude every shell metacharacter, so
    #: nothing from a tool call can reach this.
    prelude: str = ""
    #: (token, flag) pairs added only when `token` is absent from the flags in
    #: force. Not the template, and not the defaults, because neither works:
    #: the template would emit the flag twice when a caller passes their own
    #: and rustc rejects a repeated --edition outright, while a default is
    #: silently dropped the moment a caller overrides flags at all -- landing
    #: back on the edition nobody wanted.
    enforced_flags: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Crate:
    """A third-party crate baked into the sandbox image."""

    name: str
    #: Pinned exactly. A reproduction study whose numbers depend on which day
    #: the image was built is not a reproduction study, and a caret range
    #: makes the dependency graph a function of the build date.
    version: str
    #: Why it is here. Reaches the model in the tool description, because a
    #: crate it does not know is available is a crate it will not use.
    purpose: str

    @property
    def extern_name(self) -> str:
        """What `use` calls it: cargo hyphens become underscores."""
        return self.name.replace("-", "_")


#: Where the image puts the prebuilt rlibs and the extern flags naming them.
RUST_DEPS_DIR = "/opt/rust-deps"
RUST_EXTERNS_FILE = f"{RUST_DEPS_DIR}/externs.txt"

#: The crates the compiler-research image builds. Kept deliberately small:
#: each one costs image size and build time, and a sandbox with no network is
#: the wrong place to grow a general-purpose dependency tree. These are the
#: ones an algorithm from a paper actually reaches for.
RUST_CRATES: Tuple[Crate, ...] = (
    Crate("rand", "0.8.5", "random inputs"),
    Crate(
        "rand_chacha",
        "0.3.1",
        "a named, portable PRNG -- seed it so the inputs reproduce on another "
        "machine, which the default thread RNG does not",
    ),
    Crate("rayon", "1.10.0", "data parallelism, for algorithms measured across cores"),
    Crate("ndarray", "0.15.6", "n-dimensional numeric arrays"),
    Crate("num-traits", "0.2.19", "generic numeric bounds"),
    Crate("num-complex", "0.4.6", "complex arithmetic"),
    Crate("itertools", "0.13.0", "iterator combinators"),
)


def describe_rust_crates() -> str:
    """One line per crate, for the tool description a model reads."""
    return "; ".join(f"{c.name} {c.version} ({c.purpose})" for c in RUST_CRATES)


#: A debug-build Rust binary runs several times slower than a release one, so
#: the default here is not cosmetic: a run that benchmarks rustc's default
#: output measures the absence of optimisation and reports it as the
#: algorithm's speed.
_RUST_DEFAULT_FLAGS = "-O"

TOOLCHAINS: Tuple[Toolchain, ...] = (
    Toolchain(
        language="c",
        source_file="prog.c",
        compile_template="clang {flags} -o prog prog.c",
        default_flags="-O2",
        flags_hint=(
            "clang flags. The sandbox targets aarch64: use '-mcpu=native', "
            "not '-march=native'."
        ),
        aliases=("c99", "c11", ".c"),
    ),
    Toolchain(
        language="rust",
        source_file="prog.rs",
        # rustc, not cargo: there is no network in the sandbox, so a manifest
        # with dependencies could not be resolved anyway, and a single file is
        # what both tools accept.
        # -C linker=clang is in the template, not the flags, so a caller who
        # overrides flags cannot drop it. The sandbox base carries clang but
        # no gcc, and rustc invokes `cc` by default -- without this every Rust
        # build fails at link time with an error about a missing linker, which
        # reads as a broken toolchain rather than a missing alias.
        # $RUSTC_EXTERNS, not a literal list of --extern flags: see the module
        # docstring. The image decides which crates exist; this degrades to no
        # crates rather than failing to compile at all.
        compile_template=(
            "rustc {flags} -C linker=clang $RUSTC_EXTERNS -o prog prog.rs"
        ),
        prelude=(
            f'RUSTC_EXTERNS=""; [ -f {RUST_EXTERNS_FILE} ] && '
            f"RUSTC_EXTERNS=$(cat {RUST_EXTERNS_FILE})"
        ),
        default_flags=_RUST_DEFAULT_FLAGS,
        # rustc's own default edition is 2015, where `use some_crate::Thing`
        # does not resolve without an `extern crate` line -- so every modern
        # snippet fails, and every crate is invisible, with an error that
        # blames the import rather than the edition. Cargo hides this by
        # writing an edition into Cargo.toml; bare rustc has no manifest to
        # read, so it has to be said here.
        enforced_flags=(("--edition", "--edition 2021"),),
        flags_hint=(
            "rustc flags. Optimisation is '-O' or '-C opt-level=3', NOT '-O2' "
            "-- rustc rejects '-O2', and its default build is unoptimised, "
            "which times the debug binary rather than the algorithm. Target "
            "the host with '-C target-cpu=native'."
        ),
        aliases=("rs", ".rs"),
    ),
)

_BY_NAME: Dict[str, Toolchain] = {}
for _chain in TOOLCHAINS:
    _BY_NAME[_chain.language] = _chain
    for _alias in _chain.aliases:
        _BY_NAME[_alias] = _chain

#: The languages a caller may name, for error messages and tool schemas.
SUPPORTED = tuple(chain.language for chain in TOOLCHAINS)

DEFAULT_LANGUAGE = "c"


def resolve(language: Optional[str]) -> Optional[Toolchain]:
    """The toolchain for a language name, or None if it is not one we build.

    None rather than a fallback to C. Silently compiling Rust as C produces a
    wall of syntax errors that look like the model wrote bad code, when what
    actually happened is that the language was misnamed -- and the run then
    spends its iterations rewriting correct source.
    """
    key = str(language or DEFAULT_LANGUAGE).strip().lower().lstrip(".")
    if not key:
        key = DEFAULT_LANGUAGE
    return _BY_NAME.get(key)


def unsupported_language(language: Optional[str]) -> str:
    """Say what was asked for and what exists, rather than just refusing."""
    return (
        f"Unsupported language {str(language or '').strip()!r}. This sandbox "
        f"builds: {', '.join(SUPPORTED)}. A language is available only if its "
        "compiler is baked into the sandbox image, because there is no network "
        "at run time to fetch one."
    )


def compile_command(chain: Toolchain, flags: str) -> str:
    """The build line for this toolchain with these flags."""
    return chain.compile_template.format(flags=flags)


def enforce(chain: Toolchain, flags: str) -> str:
    """Add the flags this toolchain cannot correctly go without.

    Applied after the caller's flags are sanitised, and only when the caller
    did not already say something about that option -- rustc rejects a
    repeated --edition, so adding it blindly would break exactly the callers
    who got it right.
    """
    result = flags
    for token, addition in chain.enforced_flags:
        if token not in result:
            result = f"{result} {addition}".strip()
    return result


def build_script(chain: Toolchain, flags: str) -> str:
    """Everything that has to run to turn the source into `./prog`.

    Both the correctness check and the benchmark call this rather than
    assembling their own, because the two must build the program identically.
    A prelude that ran in one and not the other would mean the verified binary
    linked different crates from the timed one.
    """
    command = compile_command(chain, flags)
    return f"{chain.prelude}; {command}" if chain.prelude else command
