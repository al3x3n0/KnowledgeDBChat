# BOLT: post-link layout optimisation of whole executables.
#
# Object-file binary mode can swap one function because the linker is still
# there to prefer a strong symbol. A linked executable has no linker left, and
# rewriting one needs a tool that understands its layout: BOLT, which reorders
# blocks and functions from a profile, splits hot from cold code and folds
# identical functions. Debian bookworm ships it as bolt-19; LLVM 14 in the
# other images predates usable AArch64 support.
#
# Its own image for the reason pass-dev has one: every compile and benchmark
# call pulls compiler-research, and almost none of them rewrite a binary.
#
# Profiles come from BOLT's own instrumentation, not perf: the sandbox runs
# with --cap-drop ALL, so perf_event_open is unavailable, and an instrumented
# binary needs nothing but a writable /work.
#
# Two facts the tool depends on, both measured on Lua 5.4.6:
#   - the executable must be linked with --emit-relocs, or BOLT cannot move
#     functions at all;
#   - it must be non-PIE. Debian's clang defaults to PIE, and instrumenting a
#     PIE build failed on luaV_execute -- the hottest function -- with
#     "unable to get new address corresponding to input address".
FROM ghcr.io/al3x3n0/kdbc-sandbox-base:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends bolt-19 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/lib/llvm-19/bin:$PATH
# A login shell rebuilds PATH from /etc/profile and drops what ENV set; the
# runner starts every sandbox with `sh -lc`. Same fix as pass-dev.
RUN printf 'export PATH=/usr/lib/llvm-19/bin:$PATH\n' > /etc/profile.d/bolt.sh \
    && chmod a+r /etc/profile.d/bolt.sh

# Verified through a login shell, because that is how the runner starts one.
RUN set -eux; \
    /bin/sh -lc 'llvm-bolt --version | head -3'; \
    /bin/sh -lc 'command -v merge-fdata'; \
    /bin/sh -lc 'clang --version | head -1'

ENV HOME=/work
WORKDIR /work
