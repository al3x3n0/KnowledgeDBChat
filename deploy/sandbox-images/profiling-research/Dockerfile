# Dynamic profiling sandbox.
#
# Instruction-set extension work needs to know how often a sequence *executes*,
# not how often it appears in source. Callgrind counts every instruction
# exactly and needs no performance counters, so it works here where a PMU does
# not: this sandbox drops the privileges perf would require, and on Docker
# Desktop the counters are unavailable to the VM at all.
#
# clang and binutils are here so a workload can be built with debug info and
# disassembled in the same run that profiles it. A profile whose addresses
# cannot be mapped back to instructions is not evidence of anything.
#
# The runner invokes this image as:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
# so there is no network at run time, and nothing may write outside /work.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        lld \
        binutils \
        valgrind \
        make \
        python3 \
        time \
    && rm -rf /var/lib/apt/lists/*

# Assert the toolchain at build time rather than inside a sandboxed run that
# cannot report why it is missing.
RUN set -eux; \
    clang --version; \
    objdump --version | head -1; \
    valgrind --version; \
    python3 --version

ENV HOME=/work
WORKDIR /work
