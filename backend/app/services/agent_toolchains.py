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
turns it into an executable called `prog`, and the default flags. Anything a
language needs beyond that -- a manifest, a package fetch, a build system --
does not belong in a sandbox with no network, and a language that cannot
compile one self-contained file with one command does not fit this tool.
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
        compile_template="rustc {flags} -C linker=clang -o prog prog.rs",
        default_flags=_RUST_DEFAULT_FLAGS,
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
