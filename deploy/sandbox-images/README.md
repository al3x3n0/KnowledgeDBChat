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
| `compiler-research` | `compile_c_snippet`, `benchmark_c_snippet`, `analyze_snippet_cycles` | base + llvm (incl. `llvm-mca`), cmake, ninja, `candidate-coster`, prebuilt Rust crates |
| `polyglot-slim` | `benchmark_c_snippet`, `check_implementation` | clang, rustc, python3 and nothing else |
| `microarch-research` | scientific-validation runs | base + linux-perf, pytest |

The images a research program's tools run in (gem5, BOLT, pass-dev,
profiling, AXIS) are built from that program's repository, not from here;
installing the program allows them (`app/agent_core/research_programs.py`).
`make sandbox-load EXTRA_SANDBOX_IMAGES="..."` copies them into the stack's
daemon, and the program's Makefile calls it that way.

## polyglot-slim: the three languages and nothing else

`compiler-research` can already build all three languages -- python3 comes from
the base -- so this image is not about capability. It is what
`agent_toolchains` requires and no more, for a deployment that reproduces
algorithms and does no microarchitectural work: 1.42 GB against 1.78 GB.

That saving is smaller than it looks and the floor is worth knowing. clang and
rustc each ship their own copy of LLVM (98 MB and 153 MB), and the minimal
rustup profile is another 535 MB, so **no C+Rust image gets much below 1.4 GB**.
Two reductions were measured and rejected, and are recorded in the Dockerfile
so nobody spends the afternoon again: stripping every `.so` and `.rlib` saves
25 MB because they ship stripped, and swapping clang for gcc saves 270 MB but
makes the C compiler a function of which image ran the build.

**It is deliberately not the default.** It carries no prebuilt Rust crates, and
the compile line degrades to no crates rather than failing, so `use rand::Rng;`
resolves on `compiler-research` and does not resolve here -- verified both
ways. It has no `llvm-mca` either, so `analyze_snippet_cycles` cannot run on it.
Point a job at it explicitly when the work is single-file and stdlib-only.

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

`make sandbox-images` builds every image this repository can build, in order.
`make sandbox-check` reports which ones exist locally and what the compiler
image actually carries -- the toolchain versions and the crate count -- which
is the fastest way to tell whether a rebuild is what a failing agent tool
needs. The two that cannot be built from here have their own targets:
`make sandbox-gem5` and `make sandbox-axis AXIS_PATH=...`.

Keeping a rebuild reachable matters more than it looks. These images were
built by hand and pushed, so nothing recorded which code needed which image:
the Rust toolchain and the pinned crate set only work against an image built
after they were added, and an older one gives `rustc: not found` or an
unresolved-import error that reads like the model's mistake.

The commands the targets run, and why the contexts differ:

The base comes first, since the others are `FROM` it:

```bash
docker build -t ghcr.io/al3x3n0/kdbc-sandbox-base:latest \
  deploy/sandbox-images/base

docker build -t ghcr.io/al3x3n0/kdbc-profiling-research:latest \
  deploy/sandbox-images/profiling-research
```

`polyglot-slim` is the exception that needs no base: it installs its own
toolchains from `debian:bookworm-slim`, because basing it on an image carrying
lld, make and time would defeat the only reason it exists.

```bash
docker build -t ghcr.io/al3x3n0/kdbc-polyglot-slim:latest \
  deploy/sandbox-images/polyglot-slim
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
