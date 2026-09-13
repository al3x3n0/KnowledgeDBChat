# Shared base for the sandbox images that compile code.
#
# Three images independently installed clang, make, python3 and time. Docker
# shares layers only when they are literally the same layer, so those were
# three separate copies of the same ~400 MB toolchain: nothing was reused on
# disk, and pulling the set downloaded clang three times. Building them on one
# base makes it one copy, and makes adding a tool to a derived image a layer of
# a few megabytes rather than a new image of a few hundred.
#
# What belongs here is what more than one image needs and what rarely changes.
# A package used by a single image belongs in that image, where it can be
# changed without invalidating everyone else's cache.
#
# Deliberately NOT the base for axis-research: that image needs only z3 and
# python3, and is 235 MB standing alone. Rebasing it onto a clang-bearing
# image would nearly triple it, which is the opposite of the point.
#
# Every derived image is run as:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
# so there is no network at run time and nothing may write outside /work.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        clang \
        lld \
        binutils \
        make \
        python3 \
        time \
    && rm -rf /var/lib/apt/lists/*

# Sanity, at build time rather than in someone's run: an image that cannot
# compile is easier to diagnose here than as a failed tool call three layers up.
RUN printf 'int main(void){return 0;}\n' > /tmp/probe.c \
    && clang -O2 -o /tmp/probe /tmp/probe.c \
    && /tmp/probe \
    && rm -f /tmp/probe /tmp/probe.c

WORKDIR /work
