# gem5 research sandbox: simulate_c_workload and sample_hardware_counters.
#
# This image had no Dockerfile. It was assembled by hand from a source build
# and a manual apt, which is why nobody could say what was in it -- and why the
# C++ corpus was blocked for weeks by a missing g++ that the documented
# dependency list (build-essential) would have installed. An image whose
# contents are a memory cannot be audited, and a study that runs in it inherits
# that.
#
# The runner invokes it as:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
# so there is no network at run time and nothing may write outside /work.
#
# ARCHITECTURE. gem5 is built for the ARM ISA and the workload is compiled by
# the plain `gcc` in this image, with no cross-compiler in the command. Those
# only agree when the image is arm64: gcc emits aarch64, gem5/ARM runs it.
# Building this for x86_64 produces an image where every simulation fails on a
# binary gem5 cannot execute, and the error will be about the ELF rather than
# about the platform. Build it as:
# Nothing is COPYed from the build context -- gem5 comes from a clone inside
# the build stage -- so the context is this directory, not the repository root.
# Pointing it at the root would ship 2.3 GB of bundles and traces to the daemon
# to be ignored.
#   docker build --platform linux/arm64 \
#     -t ghcr.io/al3x3n0/kdbc-gem5-research:latest \
#     deploy/sandbox-images/gem5-research
#
# NOT built on kdbc-sandbox-base. That base carries clang and lld, which this
# image never uses, and gem5's own runtime closure has almost nothing in common
# with it. Sharing a base is worth it when the shared layer is the bulk; here it
# would be dead weight.

# Defaults to the stable branch rather than a pinned tag: a tag guessed wrong
# fails at `git clone`, and the real pin for a result is the image id a bundle
# records, not this string. Pass --build-arg GEM5_REF=v24.0.0.1 (or whichever
# tag that release actually carries) when a rebuild must match an old one.
ARG GEM5_REF=stable
ARG BUILD_JOBS=2

# ---------------------------------------------------------------------------
# Build gem5 from source.
#
# Not from ghcr.io/gem5/devcontainer: that is 2.19 GB and, on a link that
# intercepts TLS, fails part-way through with `tls: bad record MAC`. A shallow
# source clone is 155 MB and transfers fine on the same link.
#
# Expect HOURS, not minutes -- the ARM target is a few thousand translation
# units. BUILD_JOBS defaults to 2 deliberately: building at -j4 alongside the
# application stack wedged an 8-core / 16 GB machine badly enough that the
# Docker daemon stopped answering its own socket. Raise it only on a machine
# doing nothing else.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS build

ARG GEM5_REF
ARG BUILD_JOBS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        scons \
        python3-dev \
        protobuf-compiler \
        libprotobuf-dev \
        libgoogle-perftools-dev \
        libboost-dev \
        zlib1g-dev \
        m4 \
        libpng-dev \
        libelf-dev \
        pkg-config \
        libcapstone-dev \
        libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${GEM5_REF}" https://github.com/gem5/gem5.git /opt/gem5

WORKDIR /opt/gem5
RUN scons build/ARM/gem5.opt -j"${BUILD_JOBS}" --ignore-style

# The runtime closure, derived rather than listed. Naming the runtime packages
# by hand means writing libprotobuf32, libhdf5-103, libcapstone4 -- soname
# versions that are release-specific and go stale silently, leaving an image
# that builds and then cannot start gem5. Asking the linker which libraries the
# binary actually loads, and dpkg which packages own them, cannot drift.
#
# Each path is canonicalised before the lookup: bookworm is usr-merged, so ldd
# reports /lib/aarch64-linux-gnu/... while dpkg's database records the
# /usr/lib/... it resolves to, and the literal path can miss. The list is
# printed and asserted non-empty, because an empty one would install nothing
# and produce an image that builds cleanly and cannot start gem5 -- the exact
# failure this README already records from a silently failed apt.
RUN set -eux; \
    ldd build/ARM/gem5.opt | awk '/=> \//{print $3}' | while read -r lib; do \
        real="$(readlink -f "$lib")"; \
        dpkg -S "$real" 2>/dev/null || dpkg -S "$lib" 2>/dev/null || true; \
    done | cut -d: -f1 | sort -u > /tmp/runtime-packages.txt; \
    test -s /tmp/runtime-packages.txt; \
    cat /tmp/runtime-packages.txt

# ---------------------------------------------------------------------------
# Runtime: gem5 and a toolchain, without the 20 GB of object files that built
# it. The build tree is not copied -- only the binary and configs.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=build /tmp/runtime-packages.txt /tmp/runtime-packages.txt

# gcc compiles the workload; g++ is here because sample_hardware_counters
# accepts a C++ corpus and the previous image could not build one, which
# blocked every result from being about real programs rather than about
# workloads written for the study. On bookworm g++ pulls libstdc++-12-dev,
# which carries the static libstdc++.a that -static needs -- the compiler alone
# is not enough, and the two fail differently.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        python3 \
        $(tr '\n' ' ' < /tmp/runtime-packages.txt) \
    && rm -rf /var/lib/apt/lists/* /tmp/runtime-packages.txt

COPY --from=build /opt/gem5/build/ARM/gem5.opt /opt/gem5/build/ARM/gem5.opt
COPY --from=build /opt/gem5/configs /opt/gem5/configs

# gem5.opt is NOT stripped. Its symbols are what turn a panic into a diagnosis:
# running two threads on O3CPU panics with "Not enough physical registers", one
# register class at a time, and reading that took the backtrace. Stripping
# would save a few hundred megabytes and cost the only signal those runs give.

# Everything below is asserted at build time rather than discovered in an
# agent's run three layers up. This file's own README records an image that was
# committed empty because an apt failure was hidden by a redirect, and every
# tool in it reported "not found" only when someone tried to use it.
# Not `gem5.opt --version`: this build has no such option -- checked against
# the real binary, which rejected it -- and a binary that starts is not the
# claim worth asserting anyway. The SE-mode run below is.
RUN test -f /opt/gem5/configs/deprecated/example/se.py

# The two compilers, as the tool actually invokes them: statically, because
# syscall-emulation mode has no dynamic loader. The C++ probe is the same one
# agent_gem5_sandbox.cpp_support() runs against the finished image, so an image
# that passes here cannot refuse C++ there.
RUN set -eux; \
    cd /tmp; \
    printf 'int main(void){return 0;}\n' > probe.c; \
    gcc -O2 -static -o probe_c probe.c; \
    printf '#include <string>\nint main(){ std::string s="x"; return (int)s.size()-1; }\n' > probe.cc; \
    g++ -O2 -static -std=c++17 -o probe_cc probe.cc; \
    ./probe_c; \
    ./probe_cc

# End to end: a real simulation of a real binary in the configuration the tool
# uses. A gem5 that starts and a gem5 that can run a workload are different
# claims, and only the second one is the product.
RUN set -eux; \
    /opt/gem5/build/ARM/gem5.opt --outdir=/tmp/m5out \
        /opt/gem5/configs/deprecated/example/se.py \
        --cmd=/tmp/probe_c --cpu-type=O3CPU --caches --l2cache; \
    grep -q 'simInsts' /tmp/m5out/stats.txt; \
    rm -rf /tmp/m5out /tmp/probe*

ENV HOME=/work
WORKDIR /work
