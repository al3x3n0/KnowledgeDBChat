# Microarchitecture research sandbox for scientific validation runs.
#
# Invoked like the compiler image: --network none, --cap-drop ALL, uid 65534,
# with the experiment tree bind-mounted at /work.
#
# Note on perf counters: the profile advertises a "perf" toolchain, but PMU
# access needs privileges this sandbox deliberately drops, and is unavailable
# entirely when the daemon runs in a VM (Docker Desktop). perf is installed so
# a run can report what it found, but experiments here are expected to measure
# microarchitectural behaviour the way lmbench and STREAM do: with a clock.
# Built on the shared base (deploy/sandbox-images/base), which carries clang,
# lld, binutils, make, python3 and time.
FROM ghcr.io/al3x3n0/kdbc-sandbox-base:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        linux-perf \
        python3-pytest \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    clang --version; \
    python3 --version; \
    python3 -m pytest --version

ENV HOME=/work
WORKDIR /work
