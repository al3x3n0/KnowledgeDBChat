# Sandbox images

Each directory here builds one image that agent tools run submitted code in.
Every one is invoked under the same posture, defined once in
`app/services/agent_sandbox_runtime.py`:

```
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 256 --memory 2048m --cpus 2 --user 65534:65534 \
  -v <tmp>:/work -w /work <image> /bin/sh -lc "<command>"
```

Two consequences shape every Dockerfile: there is **no network at run time**, so
each toolchain must be baked in; and the command runs as `nobody`, so nothing
may write outside `/work`.

An image must also be listed in `SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES`
(see `core/config.py`) or the tools refuse to run it.

| Image | Tools | Contains |
|---|---|---|
| `sandbox-base` | (not run directly) | clang, lld, binutils, make, python3, time |
| `compiler-research` | `compile_c_snippet`, `benchmark_c_snippet`, `analyze_snippet_cycles` | base + llvm (incl. `llvm-mca`), cmake, ninja, `candidate-coster` |
| `microarch-research` | scientific-validation runs | base + linux-perf, pytest |
| `profiling-research` | `profile_c_workload` | base + valgrind (callgrind) |
| `axis-research` | `axis_check`, `axis_emit`, `axis_prove` | the AXIS binary, z3, python3 |
| `gem5-research` | `simulate_c_workload`, `sample_hardware_counters` | gem5 `build/ARM/gem5.opt` and `configs/`, gcc, g++, python3 |

## One base, several images

Three of these compile code, and each used to install clang, make, python3 and
time for itself. Docker shares layers only when they are literally the same
layer, so that was three separate copies of the same toolchain: nothing reused
on disk, and pulling the set downloaded clang three times. They now build on
`base`, which holds exactly what more than one of them needs.

Measured on this machine, for the three together:

| | disk |
|---|---|
| standalone | 2245.8 MB |
| on the shared base | 1358.1 MB |

so 888 MB, about 40%, and adding a tool to a derived image now costs a layer of
megabytes rather than a new image of hundreds. A package only one image needs
stays in that image, where changing it does not invalidate everyone else's
cache.

`axis-research` is deliberately **not** rebased. It needs only z3 and python3
and is 235 MB standing alone; putting it on a clang-bearing base would nearly
triple it. Sharing a base is worth it exactly when the shared part is most of
the image, and blanket-applying it is how a layering scheme makes things worse.

## Building

The base comes first, since the others are `FROM` it:

```bash
docker build -t ghcr.io/al3x3n0/kdbc-sandbox-base:latest \
  deploy/sandbox-images/base

docker build -t ghcr.io/al3x3n0/kdbc-profiling-research:latest \
  deploy/sandbox-images/profiling-research
```

`compiler-research` builds from the **repository root**, not from its own
directory, because it compiles `tools/candidate-coster` in a stage that never
reaches the final image:

```bash
docker build -f deploy/sandbox-images/compiler-research/Dockerfile \
  -t ghcr.io/al3x3n0/kdbc-compiler-research:latest .
```

Two do not, and the reasons are worth knowing before you spend an afternoon on
them.

### axis-research: the build context is the AXIS repository

AXIS lives in its own repository, so the Dockerfile here is used against *that*
tree:

```bash
docker build -f deploy/sandbox-images/axis-research/Dockerfile \
  -t ghcr.io/al3x3n0/kdbc-axis-research:latest /path/to/KevinAI/axis
```

### gem5-research: built from source, in a Dockerfile

This image had no Dockerfile until recently. It was assembled by hand, which is
why nobody could say what was in it — and why a C++ corpus was blocked for
weeks by a missing `g++` that the dependency list below would have installed.
An image whose contents are a memory cannot be audited, and a study that runs
in it inherits that.

```bash
docker build --platform linux/arm64 \
  -f deploy/sandbox-images/gem5-research/Dockerfile \
  -t ghcr.io/al3x3n0/kdbc-gem5-research:latest .
```

**`--platform linux/arm64` is not optional.** gem5 is built for the ARM ISA and
the workload is compiled by the plain `gcc` inside the image, with no
cross-compiler in the command. Those agree only on arm64. Build it for x86_64
and every simulation fails on a binary gem5 cannot execute, with an error about
the ELF rather than about the platform.

Three things the Dockerfile encodes, all learned the hard way:

**Build from source, not from `ghcr.io/gem5/devcontainer`.** That image is
2.19 GB and, on a link that intercepts TLS, fails part-way through with `tls:
bad record MAC` or `record with version 300 when expecting 303`. A shallow
source clone is 155 MB and transfers fine on the same link.

**`BUILD_JOBS` defaults to 2, and the stack should be stopped while it runs.**
Building at `-j4` alongside the compose stack wedged an 8-core / 16 GB-VM
machine badly enough that the Docker daemon stopped answering its own socket
and had to be restarted. Expect hours, not minutes: the ARM target is a few
thousand translation units. The build stage's object tree is never copied into
the final image.

**The runtime library list is derived, not written down.** Naming the runtime
packages by hand means writing `libprotobuf32`, `libhdf5-103`, `libcapstone4` —
soname versions that are release-specific and go stale silently, leaving an
image that builds and then cannot start gem5. The build stage asks `ldd` which
libraries the binary loads and `dpkg` which packages own them.

`gem5.opt` is deliberately **not stripped**: its symbols are what turn a panic
into a diagnosis. Running two threads on O3CPU panics with "Not enough physical
registers", one register class at a time, and reading that took the backtrace.

Everything is asserted at build time — `gem5.opt --version`, a static C
compile, a static C++ compile, and a real SE-mode simulation whose `stats.txt`
must contain `simInsts`. That last one matters because a gem5 that starts and a
gem5 that can run a workload are different claims. The C++ probe is the same
one `agent_gem5_sandbox.cpp_support()` runs against the finished image, so an
image that passes here cannot refuse C++ there.

## When apt fails inside `docker build`

On a proxied link, BuildKit's network can fail (`Failed to fetch
http://deb.debian.org/...`) while `docker run` on the same host installs
packages fine. When that happens, assemble the image by hand rather than
fighting it:

```bash
docker run -d --name stage debian:bookworm-slim sleep 900
docker exec stage apt-get update
docker exec stage apt-get install -y --no-install-recommends <packages>
docker commit -c 'ENV HOME=/work' -c 'WORKDIR /work' stage <image>
docker rm -f stage
```

Check the toolchain **inside the committed image** afterwards. Suppressing apt
output with `>/dev/null` once hid a failed install here, and the image was
committed empty — every tool in it reported "not found" only when an agent
tried to use it.
