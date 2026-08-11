# Compiler research sandbox for scientific validation runs.
#
# The runner invokes this image as:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
#
# Two consequences shape this file: there is no network at run time, so every
# toolchain the profile advertises must be baked in; and the command runs as
# nobody, so nothing may need to write outside /work.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        lld \
        llvm \
        cmake \
        ninja-build \
        make \
        python3 \
        python3-pytest \
        time \
    && rm -rf /var/lib/apt/lists/*

# The profile advertises an "llvm-opt" toolchain; Debian ships it unversioned
# already, but assert the binaries exist so a broken image fails at build time
# rather than inside a sandboxed run that cannot report why.
RUN set -eux; \
    clang --version; \
    opt --version; \
    llc --version; \
    cmake --version; \
    ninja --version; \
    python3 -m pytest --version

# HOME must be writable for tools that expect one; /work is the only writable
# mount at run time and it is bound per-run.
ENV HOME=/work
WORKDIR /work
