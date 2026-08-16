# AXIS architecture-description sandbox.
#
# AXIS (the user's separate project) turns one .axisl description of an
# instruction-set extension into the collateral a proposal needs: encoder,
# decoder, executable semantics, compiler patterns, a golden reference model,
# and SMT-LIB bit-vector semantics. Having the agent author AXIS rather than
# hand-write those per candidate is what makes a proposal machine-checkable
# and a bundle regenerable from a single source of truth.
#
# z3 is here because the semantics are only worth emitting if the equivalence
# obligation they support can be discharged in the same sandbox.
#
# BUILD CONTEXT IS THE AXIS REPOSITORY, not this one:
#   docker build -f deploy/sandbox-images/axis-research/Dockerfile \
#     -t ghcr.io/al3x3n0/kdbc-axis-research:latest /path/to/KevinAI/axis
#
# The runner invokes this image the same way as the other sandboxes:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
# so there is no network at run time and nothing may write outside /work.

FROM rust:1-slim-bookworm AS builder
WORKDIR /src
COPY . .
# Offline would be better, but the crate cache is not vendored in the source
# tree; the build is the only step that reaches the network, never a run.
RUN cargo build --release --locked || cargo build --release

FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        z3 \
        python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/release/axis /usr/local/bin/axis

# Fail at build time rather than inside a sandboxed run that cannot explain why.
RUN set -eux; \
    axis --help >/dev/null 2>&1 || axis 2>&1 | head -1; \
    z3 --version; \
    python3 --version

ENV HOME=/work
WORKDIR /work
