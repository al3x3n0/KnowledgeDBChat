# The three languages an algorithm from a paper is actually written in, and
# nothing else.
#
# The runner invokes this image as:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
#
# Why this exists next to compiler-research, which can already do all of this:
# that image is 1.78 GB because it carries LLVM, cmake, ninja, the
# candidate-coster binary and a set of prebuilt Rust crates. A run that
# implements a paper's algorithm in one file and times it needs none of that.
# This image is what `agent_toolchains` requires and no more, so a deployment
# that only reproduces algorithms can pull it instead.
#
# "Slim" is relative, and the numbers are worth stating so nobody tries the
# obvious reductions again. Measured on arm64: this image is 1.42 GB against
# compiler-research's 1.78 GB. The floor is two independent copies of LLVM --
# rustc bundles its own (153 MB) and clang links another (98 MB) -- plus 535 MB
# of Rust toolchain that the minimal rustup profile has already trimmed.
# Two reductions were measured and rejected:
#   - Stripping debug symbols from every .so and .rlib saves 25 MB. The
#     shipped binaries are already stripped.
#   - Swapping clang for gcc saves 270 MB (1.15 GB total), but the toolchain
#     table names `clang` for C and pins it as Rust's linker. Making the
#     compiler a function of which image ran the build is the coupling
#     agent_toolchains refuses for crates, and it breaks the one guarantee
#     check_implementation exists to provide: that the binary which was
#     verified is the binary that was timed.
#
# What it deliberately does NOT carry, and the consequences, because each one
# is a silent degradation rather than an error:
#   - No prebuilt Rust crates. The compile line reads
#     /opt/rust-deps/externs.txt and degrades to no crates when it is absent
#     (see agent_toolchains), so `use rand::...` fails to RESOLVE here while
#     compiling fine on compiler-research. Rust work that needs the crate set
#     belongs on that image; this is why this one is not the default.
#   - No llvm-mca, no candidate-coster: compile_c_snippet's codegen counting
#     works, but instruction-level cost modelling does not.
#   - No cmake, make or ninja: a single self-contained file is the unit here.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# clang rather than gcc, because the toolchain table names `clang` for C and
# pins `-C linker=clang` for Rust. Swapping the compiler per image would make
# the build recipe a function of which image ran it -- the exact coupling
# agent_toolchains refuses for crates, and for the same reason: the binary that
# was verified must be the binary that was timed.
#
# binutils is explicit rather than left to recommends: with
# --no-install-recommends, clang arrives without a linker, and every link fails
# with an error that reads as broken code rather than a missing package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        binutils \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# rustup rather than Debian's rustc: measured on the compiler-research image,
# apt's is 577 MB and pins rustc 1.63 (2022) while the minimal rustup profile
# is 508 MB and current.
#
# Installed to a system path with RUSTUP_HOME and CARGO_HOME set, because the
# runner executes as --user 65534:65534, which has no writable home and would
# never find a toolchain under /root.
#
# The ENV PATH below is NOT sufficient on its own: the runner invokes
# `/bin/sh -lc`, and a LOGIN shell sources /etc/profile, which overwrites PATH
# and discards the ENV -- producing `rustc: not found` in an image where rustc
# is plainly installed. Hence both the symlink into /usr/local/bin (already on
# /etc/profile's PATH) and the profile.d fragment.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
    && chmod -R a+rX /usr/local/rustup /usr/local/cargo \
    && ln -s /usr/local/cargo/bin/rustc /usr/local/bin/rustc \
    && printf 'export RUSTUP_HOME=/usr/local/rustup\nexport CARGO_HOME=/usr/local/cargo\nexport PATH=/usr/local/cargo/bin:$PATH\n' \
        > /etc/profile.d/rust.sh \
    && apt-get purge -y curl ca-certificates && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Probed exactly as the runner invokes it: `/bin/sh -lc` as uid 65534. Both
# halves matter. An earlier probe elsewhere in this repo used `su`, which
# inherits root's environment, so it passed while every real run failed with
# `rustc: not found` -- the login shell had reset PATH. A probe that does not
# reproduce the caller's posture verifies nothing.
#
# One probe per language, each building and RUNNING a program rather than
# asking for --version: a toolchain that reports its version and cannot link
# is the failure this catches. The Rust probe uses no crates on purpose --
# there are none here, and a probe that needed them would fail correctly but
# for a reason this image is not claiming to support.
RUN printf 'int main(void){return 42;}\n' > /probe.c \
    && printf 'fn main(){ std::process::exit(42); }\n' > /probe.rs \
    && printf 'import sys; sys.exit(42)\n' > /probe.py \
    && chmod 644 /probe.c /probe.rs /probe.py \
    && setpriv --reuid 65534 --regid 65534 --clear-groups /bin/sh -lc '\
        cd /tmp \
        && clang -O2 -o pc /probe.c; [ $? -eq 0 ] || exit 1; ./pc; [ $? -eq 42 ] || exit 1; \
        rustc -O --edition 2021 -C linker=clang -o pr /probe.rs || exit 1; ./pr; [ $? -eq 42 ] || exit 1; \
        python3 -m py_compile /probe.py || exit 1; python3 /probe.py; [ $? -eq 42 ] || exit 1; \
        echo ALL_THREE_OK' \
    | grep -q ALL_THREE_OK \
    && rm -f /probe.c /probe.rs /probe.py

WORKDIR /work
