# Compiler research sandbox for scientific validation runs.
#
# The runner invokes this image as:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
#
# Two consequences shape this file: there is no network at run time, so every
# toolchain the profile advertises must be baked in; and the command runs as
# nobody, so nothing may need to write outside /work.
# Built on the shared base (deploy/sandbox-images/base), which carries clang,
# lld, binutils, make, python3 and time. Only what is specific to compiler
# research is installed here, so a change to it does not rebuild the toolchain
# every other image shares.
#
# NOTE: this image builds from the repository root, not from this
# directory, because the Rust crate lives at tools/candidate-coster:
#   docker build -f deploy/sandbox-images/compiler-research/Dockerfile .
#
# candidate-coster is built in a stage of its own: the Rust toolchain is 1.5 GB
# and none of it is needed to run a 2 MB binary. It is here so an agent can
# generate an llvm-mca snippet from a mined pattern instead of hand-writing
# assembly -- the step that produced a 40% error when done by hand.
FROM rust:1-slim-bookworm AS coster
WORKDIR /src
COPY tools/candidate-coster/ ./
RUN cargo build --release --offline || cargo build --release

FROM ghcr.io/al3x3n0/kdbc-sandbox-base:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        llvm \
        cmake \
        ninja-build \
        python3-pytest \
    && rm -rf /var/lib/apt/lists/*

# A Rust toolchain, so an agent can implement an algorithm in Rust and have the
# same image both check it against reference cases and time it. Verifying in
# one language and benchmarking in another would defeat the check entirely, so
# the compiler has to live where the benchmark runs.
#
# rustup rather than Debian's rustc, measured: apt's is 577 MB and pins
# rustc 1.63 (2022); the minimal rustup profile is 508 MB and current. Smaller
# and three years newer, so there is no trade to weigh.
#
# Installed to a system path, not a home directory: the runner executes as
# --user 65534:65534, which has no writable home and would not find a
# toolchain under /root. RUSTUP_HOME and CARGO_HOME are set for the same
# reason -- rustup's defaults are under $HOME.
#
# The ENV PATH below is NOT enough on its own, and this cost a debugging round:
# the runner invokes `/bin/sh -lc`, and a LOGIN shell sources /etc/profile,
# which overwrites PATH and throws the ENV away. `rustc: not found`, in an
# image where rustc is plainly installed. Hence both of the belt-and-braces
# below -- a symlink into /usr/local/bin, which /etc/profile's PATH already
# contains, and a profile.d fragment that puts the toolchain back for anything
# that reads it.
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
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=coster /src/target/release/candidate-coster /usr/local/bin/candidate-coster

RUN llvm-mca --version > /dev/null && candidate-coster --help > /dev/null

# Probed exactly as the runner invokes it: `/bin/sh -lc` as uid 65534. Both
# halves of that matter and neither is incidental. An earlier version of this
# probe used `su`, which inherits root's environment, so it passed while every
# real run failed with `rustc: not found` -- the login shell had reset PATH.
# A probe that does not reproduce the caller's posture verifies nothing.
RUN printf 'fn main(){println!("{}", 6*7);}' > /probe.rs && chmod 644 /probe.rs \
    && setpriv --reuid 65534 --regid 65534 --clear-groups \
        /bin/sh -lc 'cd /tmp && rustc -O -C linker=clang -o p /probe.rs && ./p' \
    | grep -qx 42 \
    && rm -f /probe.rs
