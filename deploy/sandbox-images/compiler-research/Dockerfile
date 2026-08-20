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

COPY --from=coster /src/target/release/candidate-coster /usr/local/bin/candidate-coster

RUN llvm-mca --version > /dev/null && candidate-coster --help > /dev/null
