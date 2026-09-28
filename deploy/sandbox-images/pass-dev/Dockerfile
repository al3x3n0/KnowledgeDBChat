# Building LLVM pass plugins, which is the one thing the compiler-research
# sandbox cannot do.
#
# That image ships the four plugins this repo has written but deliberately not
# the headers that built them: llvm-14-dev and g++ are 500 MB and nothing is
# needed to *load* a 300 KB plugin. The same reasoning that keeps them out of
# the common image is why they get one of their own -- every compile, benchmark
# and scan call pulls compiler-research, and almost none of them build a pass.
#
# The runner invokes this the same way as the other sandboxes:
#   docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
#     --user 65534:65534 -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
# so there is no network at build or run time beyond this file, and the command
# runs as nobody.
#
# llvm-config is NOT on PATH here: llvm-14-dev installs it under
# /usr/lib/llvm-14/bin, and a bare `llvm-config` resolves in the
# compiler-research image but not in this base. That difference already cost
# one build, which failed reporting a missing llvm/IR/Function.h rather than a
# missing tool, so the path is fixed in the environment below and anything
# building here inherits it.
FROM ghcr.io/al3x3n0/kdbc-sandbox-base:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        llvm-14-dev \
        g++ \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_CONFIG=/usr/lib/llvm-14/bin/llvm-config
ENV PATH=/usr/lib/llvm-14/bin:$PATH

# ENV PATH is not enough. The runner invokes `/bin/sh -lc`, and a login shell
# rebuilds PATH from /etc/profile, discarding what Docker set -- which is how
# opt and llvm-config went missing from a sandbox that demonstrably had them.
# The same fix the Rust toolchain needed in compiler-research, for the same
# reason.
RUN printf 'export PATH=/usr/lib/llvm-14/bin:$PATH\nexport LLVM_CONFIG=/usr/lib/llvm-14/bin/llvm-config\n' \
        > /etc/profile.d/llvm.sh \
    && chmod a+r /etc/profile.d/llvm.sh

# Verified through a LOGIN shell, because that is how the runner starts one. An
# earlier version of this check used plain `sh -c`, passed, and shipped an
# image where opt was unreachable at run time.
RUN set -eux; \
    /bin/sh -lc 'llvm-config --version'; \
    /bin/sh -lc 'opt --version | head -2'; \
    /bin/sh -lc 'g++ --version | head -1'; \
    /bin/sh -lc 'clang --version | head -1'; \
    /bin/sh -lc 'test -d "$(llvm-config --includedir)/llvm"' 

ENV HOME=/work
WORKDIR /work
