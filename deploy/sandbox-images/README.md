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
| `compiler-research` | `compile_c_snippet`, `benchmark_c_snippet`, `analyze_snippet_cycles` | clang, lld, llvm (incl. `llvm-mca`), cmake, ninja, python3 |
| `microarch-research` | scientific-validation runs | clang, linux-perf, python3 |
| `profiling-research` | `profile_c_workload` | clang, binutils, valgrind (callgrind), python3 |
| `axis-research` | `axis_check`, `axis_emit`, `axis_prove` | the AXIS binary, z3, python3 |
| `gem5-research` | `simulate_c_workload` | gem5 `build/ARM/gem5.opt` and `configs/`, gcc |

## Building

Most build normally:

```bash
docker build -t ghcr.io/al3x3n0/kdbc-profiling-research:latest \
  deploy/sandbox-images/profiling-research
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

### gem5-research: build gem5 from source, not from the published image

The obvious route — pulling `ghcr.io/gem5/devcontainer` — is a 2.19 GB image,
and on a link that intercepts TLS it fails repeatedly part-way through with
`tls: bad record MAC` or `record with version 300 when expecting 303`. A
shallow **source clone is 155 MB** and transfers fine on the same link, so
build it:

```bash
git clone --depth 1 https://github.com/gem5/gem5.git
# deps: build-essential scons python3-dev protobuf-compiler libprotobuf-dev
#       libgoogle-perftools-dev libboost-dev zlib1g-dev m4 libpng-dev
#       libelf-dev pkg-config libcapstone-dev libhdf5-serial-dev
scons build/ARM/gem5.opt -j2
```

**Use `-j2`, and stop the application stack while it runs.** Building at `-j4`
alongside the compose stack wedged an 8-core / 16 GB-VM machine badly enough
that the Docker daemon stopped answering its own socket and had to be
restarted. The build itself survives that: the container's filesystem keeps its
object files and scons resumes incrementally. Expect hours, not minutes — the
ARM target is a few thousand translation units.

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
