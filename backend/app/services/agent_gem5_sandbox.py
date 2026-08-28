"""Measure a workload in a simulated core, as the referee for a prediction.

llvm-mca estimates how a sequence issues on a scheduling model with a warm
front end and no cache misses. gem5 executes it in a modelled core with caches,
a branch predictor and a memory hierarchy, and reports what that took. The
difference between those two numbers is the thing worth knowing: it is how a
prediction gets scored rather than merely defended.

Simulation is slow -- an out-of-order core runs on the order of 100k
instructions a second -- so what belongs here is a kernel with a bounded input,
after llvm-mca has already thrown out the candidates that were never going to
pay. The tool says that rather than letting it be discovered as a timeout.
"""

from __future__ import annotations

import posixpath
import re

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from app.services import agent_sandbox_runtime, gem5_stats

DEFAULT_IMAGE = "ghcr.io/al3x3n0/kdbc-gem5-research:latest"
GEM5_BINARY = "/opt/gem5/build/ARM/gem5.opt"
GEM5_SE_CONFIG = "/opt/gem5/configs/deprecated/example/se.py"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_FLAGS = "-O3 -static"

#: Intervals each side of a candidate split needs before its median means
#: anything. Below this every trace has a "regime change" at its second sample.
MIN_PER_SIDE = 15
DEFAULT_CPU = "O3CPU"
MAX_CODE_CHARS = 40_000
MAX_OUTPUT_CHARS = 8_000

# Simulated core models. Restricted because the value is interpolated into a
# shell command, and named so the catalog documents the real choice: AtomicCPU
# counts instructions quickly and tells you nothing about timing, O3CPU models
# an out-of-order pipeline and is the one a performance claim needs.
CPU_TYPES = {
    "O3CPU": "a generic out-of-order pipeline; timing-capable but models no real core",
    "MinorCPU": "an in-order pipeline",
    "TimingSimpleCPU": "simple timing, memory latencies but no pipeline",
    "AtomicSimpleCPU": "fastest, no timing model at all",
    # Named ARM cores. The generic O3CPU carries gem5's default latencies,
    # which correspond to no shipped silicon -- comparing its cycles against a
    # real machine measures the gap between two unrelated cores as much as
    # anything else. Naming a core is the first requirement of a cycle claim.
    "NeoverseV2": "Arm Neoverse V2 (server); a modelled real core",
    "O3_ARM_v7a_3": "Arm Cortex-A15 class, 3-wide",
    "HPI": "Arm's High Performance In-order model",
    "ex5_big": "ARM big.LITTLE big core (Cortex-A15 class)",
    "ex5_LITTLE": "ARM big.LITTLE little core (Cortex-A7 class)",
}

#: Names gem5 itself answers to that this list spells differently. A live run
#: asked for DerivO3CPU -- the class O3CPU actually resolves to, and a name
#: gem5's own ObjectList offers -- and was refused as unknown. Being stricter
#: than the simulator costs an iteration and teaches nothing.
CPU_TYPE_ALIASES = {
    # Short forms a caller reaches for when the full name is a mouthful. A
    # live run asked for "O3" and was refused with a list that contains
    # "O3CPU" three characters away.
    "O3": "O3CPU",
    "o3": "O3CPU",
    "Minor": "MinorCPU",
    "Atomic": "AtomicSimpleCPU",
    "Timing": "TimingSimpleCPU",
    "DerivO3CPU": "O3CPU",
    "BaseO3CPU": "O3CPU",
    "ArmO3CPU": "O3CPU",
    "BaseMinorCPU": "MinorCPU",
    "ArmMinorCPU": "MinorCPU",
    "BaseTimingSimpleCPU": "TimingSimpleCPU",
    "ArmTimingSimpleCPU": "TimingSimpleCPU",
    "BaseAtomicSimpleCPU": "AtomicSimpleCPU",
    "ArmAtomicSimpleCPU": "AtomicSimpleCPU",
}


def resolve_cpu_type(name: str) -> str:
    """The model this name means, or the name unchanged if it means nothing."""
    cleaned = str(name or "").strip()
    return CPU_TYPE_ALIASES.get(cleaned, cleaned)


# Parameter overrides are how a core model gets tuned: each entry is a full
# `system....=<value>` assignment applied after the config is built, so a
# candidate model is a list of these rather than a forked config file.
# The path must be complete. An earlier version prepended `system.cpu[0].`
# for the caller, which silently doubled the prefix when the caller passed a
# full path and produced `KeyError: system` from deep inside gem5.
# Vector members need indices: the flattened names config.ini prints
# (`FUList03.opList4`) are not addressable, `FUList[3].opList[4]` is.
SAFE_PARAM = re.compile(r"^system[A-Za-z0-9_.\[\]]*=[A-Za-z0-9_.\-]+$")
MAX_PARAM_OVERRIDES = 40

# Core models whose functional-unit pool declares SimdFloatMultAcc but not
# FloatMultAcc. A scalar `fmadd` then has no unit that can ever accept it, and
# the out-of-order model does not report this -- it waits forever. Measured:
# a program whose only unusual instruction is one `fmadd` finishes in 2s on
# O3CPU and never finishes on NeoverseV2. Since the compiler contracts `a*b+c`
# into fmadd by default, this hits ordinary floating-point code, which is
# exactly the code an instruction-set proposal is about.
MODELS_WITHOUT_SCALAR_FMA = frozenset({"NeoverseV2", "ex5_big", "ex5_LITTLE"})
SCALAR_FMA_MNEMONICS = "fmadd|fmsub|fnmadd|fnmsub"

# Core parameters worth reporting as tunable. Widths and queue depths, which
# no single-instruction benchmark can constrain -- only whole-kernel behaviour
# pins them down, which is why they are listed apart from the op latencies.
TUNABLE_CPU_PARAMS = frozenset(
    {
        "numROBEntries",
        "numIQEntries",
        "numPhysIntRegs",
        "numPhysFloatRegs",
        "numPhysVecRegs",
        "LQEntries",
        "SQEntries",
        "fetchWidth",
        "decodeWidth",
        "renameWidth",
        "dispatchWidth",
        "issueWidth",
        "wbWidth",
        "commitWidth",
        "squashWidth",
        "fetchBufferSize",
    }
)

SAFE_RUN_ARGS = re.compile(r"^[-A-Za-z0-9_=+.,/ ]*$")
SAFE_FLAGS = re.compile(r"^[-A-Za-z0-9_=+., /]*$")


#: A level shift this large between the two sides of a split is a regime
#: change rather than the workload's own variation.
REGIME_SHIFT_RATIO = 1.5


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank quantile of an already sorted sequence."""
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _absolute_deviation(ordered: Sequence[float]) -> float:
    """Total absolute deviation from the median -- the L1 change-point cost.

    Deliberately unnormalised and a total rather than a mean. Scaling each side
    by its own median lets the cheaper regime dominate the objective, which
    drags the reported break towards the start of the trace; taking means
    instead of totals makes a two-interval side look perfectly homogeneous and
    drags it towards the ends. Both were tried on a real trace, and both put
    the break in the wrong place.
    """
    middle = ordered[len(ordered) // 2]
    return sum(abs(v - middle) for v in ordered)


def find_regime_change(cycles: Sequence[float]) -> Optional[Dict[str, Any]]:
    """The interval where the trace stops being one experiment and becomes two.

    A co-runner that has not finished initialising, a cache that is still cold,
    a phase the workload enters once and never leaves: the counters before such
    a point describe a machine that does not recur. Measuring across it reports
    persistence that is mostly the detection of the break -- on the SMT trace
    this was written for, persistence read 0.843 across the break and 0.405
    after it, so more than half of what looked like predictable structure was
    the trace announcing that solo intervals stay solo.

    **A level shift is not a phase.** A workload alternating between two costs
    is doing what it was written to do, and flagging that as a regime change
    would condemn every interesting trace. Comparing medians is not enough to
    tell them apart -- on a period-two alternation the median of each side
    lands on whichever level happens to hold the majority there, and a one-
    element parity difference flips it, which reported a clean 9x regime change
    on an alternating control. So the test is separation, not distance: the
    upper quartile of the lower side must sit below the lower quartile of the
    upper side. Two regimes barely overlap; two phases overlap completely.

    Reported, never trimmed. Silently dropping the front of a caller's trace
    would change what was measured without saying so.
    """
    n = len(cycles)
    if n < 4 * MIN_PER_SIDE:
        return None

    # Two objectives, and conflating them puts the break in the wrong place.
    # Separation decides WHICH splits are candidates, and the ratio decides
    # whether the shift is worth reporting at all -- but on a clean step every
    # split from the floor onwards ties on both, so neither locates anything.
    # The break is where the two sides are most internally homogeneous, which
    # is the change point rather than the first split that qualifies.
    best = None
    for split in range(MIN_PER_SIDE, n - MIN_PER_SIDE):
        before = sorted(cycles[:split])
        after = sorted(cycles[split:])
        median_before = before[len(before) // 2]
        median_after = after[len(after) // 2]
        if median_before <= 0 or median_after <= 0:
            continue

        low, high = (
            (before, after) if median_before <= median_after else (after, before)
        )
        if _quantile(low, 0.75) >= _quantile(high, 0.25):
            continue

        # Not the interquartile range: when a side is mostly one level with a
        # block of another at one end, both quartiles land inside the majority
        # block and the IQR reads zero, so a split straddling two regimes
        # scores as homogeneous as one that separates them. Absolute deviation
        # counts every interval that disagrees with its own side.
        spread = sum(_absolute_deviation(side) for side in (before, after))
        if best is None or spread < best["spread"]:
            best = {
                "at": split,
                "spread": spread,
                "ratio": max(
                    median_before / median_after, median_after / median_before
                ),
                "before": median_before,
                "after": median_after,
            }

    if not best or best["ratio"] < REGIME_SHIFT_RATIO:
        return None
    return {
        "at_interval": best["at"],
        "ratio": round(best["ratio"], 2),
        "median_cycles_before": round(best["before"], 1),
        "median_cycles_after": round(best["after"], 1),
    }


#: Whether an image can build a static C++ binary. Asked of the image once and
#: remembered, because the answer changes only when the image is rebuilt.
_CPP_SUPPORT: Dict[str, Dict[str, Any]] = {}

#: A trivial compile has no reason to take longer than this, and a probe that
#: hangs must not spend a simulation-sized timeout finding out.
CPP_PROBE_TIMEOUT_SECONDS = 120


async def cpp_support(image: str) -> Dict[str, Any]:
    """Whether this image can build C++, asked rather than assumed.

    This used to be a constant: C++ was refused with a message asserting that
    the gem5 image has no C++ compiler, which was true, and which would have
    stayed true in the code long after it stopped being true of the image. The
    message even named the fix -- add g++ and libstdc++-static -- and would
    then have gone on refusing a caller who applied it. Two halves built apart.

    So the question goes to the image. The probe is a real static C++ compile
    rather than `command -v g++`, because the requirement is a compiler AND a
    static libstdc++, those fail differently, and they need different fixes.

    A definite answer is cached; a probe that could not run is not, so a docker
    hiccup does not disable C++ for the life of the process.
    """
    cached = _CPP_SUPPORT.get(image)
    if cached is not None:
        return cached

    script = (
        "printf 'int main(){return 0;}\n' > probe.cc && "
        "g++ -O0 -static -o probe probe.cc 2> probe_err.txt "
        "&& echo CPP_STATIC_OK || { tail -5 probe_err.txt >&2; exit 1; }"
    )
    with tempfile.TemporaryDirectory(prefix="gem5_cpp_probe_") as workdir:
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=CPP_PROBE_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # timeout, missing daemon, image not pulled
            return {
                "supported": False,
                "probed": False,
                "detail": f"the C++ probe could not run: {exc}",
            }

    result = {
        "supported": returncode == 0 and "CPP_STATIC_OK" in str(stdout),
        "probed": True,
        "detail": str(stderr or "").strip()[:400],
    }
    _CPP_SUPPORT[image] = result
    return result


#: What this image's gem5 can actually run, keyed by (image, cpu_type).
_MODEL_SUPPORT: Dict[str, Dict[str, Any]] = {}

#: Op classes without which any glibc binary deadlocks. gem5 issues an
#: instruction only when a functional unit for its class exists; when none
#: does, the instruction waits forever and the simulation neither finishes nor
#: fails.
#:
#: FloatMisc, established by elimination rather than assumed. The first guess
#: was FloatMemRead/FloatMemWrite, on the strength of three models: the two
#: that ran declared them and the one that hung did not. NeoverseV2 refutes
#: that -- it declares neither and runs glibc binaries perfectly, sixteen of
#: them in one afternoon. Diffing its pool against ex5_big's leaves exactly one
#: class, and it holds across every model in this image: ex5_LITTLE,
#: NeoverseV2, O3CPU and MinorCPU all declare FloatMisc and all run; ex5_big
#: does not, and is the only one that hangs.
REQUIRED_OP_CLASSES = ("FloatMisc",)

MODEL_PROBE_TIMEOUT_SECONDS = 120


async def model_support(image: str, cpu_type: str) -> Dict[str, Any]:
    """Whether this image's gem5 can run this core model, asked not assumed.

    CPU_TYPES lists five named ARM cores. In the image this project uses, two
    of them are not compiled in and one deadlocks on every glibc binary -- and
    the deadlock is the expensive kind, because a run that hangs consumes its
    whole timeout and then reports a timeout, which reads as "the workload was
    too big" and sends the caller to shrink a workload that was never the
    problem.

    The probe runs a thousand instructions and reads the configuration gem5
    dumps at startup. That is enough: an unavailable model says so immediately,
    and a model whose functional-unit pool omits a class glibc needs can be
    identified from the pool rather than by waiting for the hang.
    """
    key = f"{image}::{cpu_type}"
    cached = _MODEL_SUPPORT.get(key)
    if cached is not None:
        return cached

    script = (
        "printf 'int main(void){return 0;}\n' > probe.c && "
        "gcc -O0 -static -o probe probe.c && "
        f"{GEM5_BINARY} --outdir=probe_out {GEM5_SE_CONFIG} "
        f"--cmd=./probe --cpu-type={cpu_type} --caches -I 1000 > probe.log 2>&1; "
        "grep -h '^opClass=' probe_out/config.ini 2>/dev/null | sort -u; "
        "echo ---; tail -3 probe.log"
    )
    with tempfile.TemporaryDirectory(prefix="gem5_model_probe_") as workdir:
        try:
            _rc, stdout, _stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=MODEL_PROBE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # Not cached: a docker hiccup must not disable a model for the life
            # of the process.
            return {"usable": True, "probed": False, "detail": str(exc)[:200]}

    text = str(stdout or "")
    pool, _, log = text.partition("---")
    op_classes = {
        line.split("=", 1)[1].strip()
        for line in pool.splitlines()
        if line.startswith("opClass=")
    }

    if "is unavailable" in log or (not op_classes and "unavailable" in log.lower()):
        result = {
            "usable": False,
            "probed": True,
            "reason": (
                f"{cpu_type} is not compiled into the gem5 build in {image}. "
                "It is named in this tool's model list but this image cannot "
                "run it; choose another model."
            ),
        }
    elif not op_classes:
        # The probe ran and told us nothing: no opClass lines, and the log did
        # not say the model is unavailable either. `op_classes and ...` below
        # was written to stop an empty set from looking like "every required
        # class is missing", and it does -- but it also turns an unreadable
        # config.ini into `usable: True, probed: True`, a check that could not
        # tell recorded as a check that passed. Say what happened instead, and
        # do not cache it: the next call may read the file fine.
        return {
            "usable": True,
            "probed": False,
            "detail": (
                f"{cpu_type} could not be probed: the run produced no "
                "opClass lines and no 'unavailable' message, so nothing was "
                "learned about its functional units. Proceeding, but a model "
                "missing a class glibc needs would hang rather than fail."
            ),
        }
    else:
        missing = [c for c in REQUIRED_OP_CLASSES if c not in op_classes]
        if missing:
            result = {
                "usable": False,
                "probed": True,
                "reason": (
                    f"{cpu_type} in this gem5 build has no functional unit for "
                    f"{', '.join(missing)}, so any program linked against glibc "
                    "deadlocks before main: the instruction waits for a unit "
                    "that does not exist and the simulation neither finishes "
                    "nor fails. This is a property of the model in this image, "
                    "not of the workload -- a smaller workload hangs too. "
                    "Choose another model."
                ),
            }
        else:
            result = {"usable": True, "probed": True, "op_classes": sorted(op_classes)}

    _MODEL_SUPPORT[key] = result
    return result


def forget_model_support(image: str = "") -> None:
    """Drop what was learned about an image's models, for a rebuild or a test."""
    if image:
        for key in [k for k in _MODEL_SUPPORT if k.startswith(f"{image}::")]:
            _MODEL_SUPPORT.pop(key, None)
    else:
        _MODEL_SUPPORT.clear()


def forget_cpp_support(image: str = "") -> None:
    """Drop what was learned about an image, for a rebuild or for a test."""
    if image:
        _CPP_SUPPORT.pop(image, None)
    else:
        _CPP_SUPPORT.clear()


def _check_arguments(code: str) -> Optional[Dict[str, Any]]:
    """What is wrong with the call, decidable without a sandbox."""
    if not (code or "").strip():
        return {"success": False, "error": "code is required"}
    if len(code) > MAX_CODE_CHARS:
        return {"success": False, "error": f"code exceeds {MAX_CODE_CHARS} characters"}
    return None


def _check_staged_paths(
    extra_files: Optional[Dict[str, str]],
    include_dirs: Optional[Sequence[str]],
) -> Optional[Dict[str, Any]]:
    """Whether the staged names stay inside the workspace, decided lexically.

    The same property is checked again during staging, against the real
    directory, and that check is the one that guards the write -- this one
    exists so the refusal does not need a sandbox to be reached. A caller who
    passes ``../../etc/passwd`` is told so on a server with execution disabled,
    instead of being told about ENABLE_UNSAFE_CODE_EXECUTION and left to
    discover the path problem later.
    """
    root = "/workspace"
    for name in extra_files or {}:
        candidate = posixpath.normpath(posixpath.join(root, str(name)))
        if not candidate.startswith(root + "/"):
            return {
                "success": False,
                "error": f"extra_files path escapes the workspace: {name!r}",
            }
    for directory in include_dirs or []:
        resolved = posixpath.normpath(posixpath.join(root, str(directory)))
        if not resolved.startswith(root):
            return {
                "success": False,
                "error": f"include_dirs path escapes the workspace: {directory!r}",
            }
    return None


def _check_environment(image: str) -> Optional[Dict[str, Any]]:
    """Whether this server can run a sandbox at all.

    Checked after every argument, never before one. These two were a single
    preflight that ran first, so a caller who mistyped a language or a cpu_type
    was told the sandbox was disabled -- an accurate statement about the server
    and the wrong thing to fix, which sends a run off configuring a host when
    the defect is in its own call. The module said as much further down ("after
    argument validation, never before it") while opening with the violation.
    """
    if not agent_sandbox_runtime.execution_enabled():
        return {
            "success": False,
            "error": (
                "Sandboxed execution is disabled on this server "
                "(ENABLE_UNSAFE_CODE_EXECUTION is false)."
            ),
        }
    if image not in agent_sandbox_runtime.allowed_images():
        return {
            "success": False,
            "error": agent_sandbox_runtime.image_not_allowlisted(image),
        }
    return None


def explain_gem5_failure(stderr: str, overrides: Sequence[str]) -> str:
    """Say what gem5's config errors mean for the overrides that caused them.

    A wrong parameter path surfaces from deep inside gem5's Python as
    `KeyError` or `AttributeError` with no mention of the flag that produced
    it. The two failures are diagnostic once named: `SimObjectVector has no
    attribute` means a vector was addressed without an index, and `KeyError`
    means the path names something that is not a child of its parent.
    """
    text = str(stderr or "")
    if overrides:
        if "SimObjectVector" in text:
            return (
                "A parameter override addressed a vector without an index. "
                "`system.cpu` and `instQueues` are vectors even when there is "
                "one of each, so the path must read "
                "`system.cpu[0].instQueues[0]....`. Overrides passed: "
                + "; ".join(overrides[:4])
            )
        if "KeyError" in text:
            return (
                "A parameter override named something that is not a child of "
                "its parent. Note that the flattened names config.ini prints "
                "(`FUList03.opList4`) are not addressable -- index the vectors "
                "instead (`FUList[3].opList[4]`). Call "
                "describe_model_parameters for the paths this model actually "
                "has. Overrides passed: " + "; ".join(overrides[:4])
            )
    return "gem5 exited non-zero; see stderr for its last output"


# Path segments that are SimObjectVectors, so an index is required even when
# config.ini prints them bare. NeoverseV2 prints `FUList` with no suffix -- one
# functional unit per issue queue -- and the resulting unindexed path was
# rejected, while O3CPU's `FUList03` worked. Whether a name carries a numeric
# suffix says how many members there are, not whether it is a vector.
VECTOR_SEGMENTS = frozenset({"cpu", "instQueues", "FUList", "opList"})


def _addressable(config_path: str) -> str:
    """Turn a config.ini path into one gem5's -P flag accepts.

    config.ini flattens vector members into names (`FUList03.opList4`) that
    cannot be assigned to; the flag needs indices (`FUList[3].opList[4]`).
    Bare vector names need `[0]`. Finding this out took an hour of KeyErrors,
    which is why a wrong path now explains itself -- see explain_gem5_failure.
    """
    segments = []
    for segment in config_path.split("."):
        indexed = re.match(r"^([A-Za-z_]+?)(\d+)$", segment)
        if indexed and indexed.group(1) in VECTOR_SEGMENTS:
            segments.append(f"{indexed.group(1)}[{int(indexed.group(2))}]")
        elif segment in VECTOR_SEGMENTS:
            segments.append(f"{segment}[0]")
        else:
            segments.append(segment)
    return ".".join(segments)


def parse_model_parameters(config_ini: str) -> Dict[str, Any]:
    """Read a gem5 config.ini into the knobs a calibration run can turn.

    Two groups, because they are tuned for different reasons: per-op-class
    latencies, which is what a measurement of one instruction constrains, and
    the core's widths and queue depths, which only whole-kernel behaviour can
    pin down.
    """
    op_by_class: Dict[str, Dict[str, Any]] = {}
    cpu_parameters: List[Dict[str, Any]] = []
    section = ""
    fields: Dict[str, str] = {}

    def flush() -> None:
        if not section:
            return
        if ".opList" in section and "opClass" in fields:
            op_class = fields.get("opClass", "").strip()
            try:
                op_lat = int(fields.get("opLat", "") or 0)
            except ValueError:
                op_lat = 0
            if op_class:
                entry = op_by_class.setdefault(
                    op_class,
                    {
                        "op_class": op_class,
                        "op_lat": op_lat,
                        "pipelined": str(fields.get("pipelined", "")).strip().lower()
                        in {"true", "1"},
                        "parameters": [],
                    },
                )
                entry["parameters"].append(f"{_addressable(section)}.opLat")
                # Same op class at different latencies in different queues is
                # possible; report the range rather than silently keeping one.
                if op_lat != entry["op_lat"]:
                    entry["op_lat_varies"] = True
                    entry["op_lat"] = max(entry["op_lat"], op_lat)

    for raw_line in config_ini.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            flush()
            section = line[1:-1].strip()
            fields = {}
            continue
        if "=" in line and section:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
            if (
                section in {"system.cpu", "system.cpu0"}
                and key.strip() in TUNABLE_CPU_PARAMS
            ):
                try:
                    numeric: Any = int(value.strip())
                except ValueError:
                    continue
                if not any(row["name"] == key.strip() for row in cpu_parameters):
                    cpu_parameters.append(
                        {
                            "name": key.strip(),
                            "value": numeric,
                            "parameter": f"{_addressable(section)}.{key.strip()}",
                        }
                    )
    flush()

    op_latencies = sorted(op_by_class.values(), key=lambda row: row["op_class"])
    cpu_parameters.sort(key=lambda row: row["name"])
    return {"op_latencies": op_latencies, "cpu_parameters": cpu_parameters}


async def describe_model_parameters(
    *,
    cpu_type: str = DEFAULT_CPU,
    op_classes: Optional[Sequence[str]] = None,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = 600,
) -> Dict[str, Any]:
    """List a core model's tunable parameters and the paths that set them.

    Tuning a model is impossible without knowing what it exposes and how to
    address it, and neither is guessable: the latencies live in a functional
    unit pool whose layout differs per model, and the paths config.ini prints
    are not the paths that can be assigned to. This runs the model once on a
    trivial program and reports what it found.
    """
    model = resolve_cpu_type(cpu_type or DEFAULT_CPU)
    if model not in CPU_TYPES:
        return {
            "success": False,
            "error": (
                f"Unknown cpu_type {cpu_type!r}. Available: "
                + ", ".join(f"{name} ({why})" for name, why in CPU_TYPES.items())
            ),
        }

    blocked = _check_environment(image)
    if blocked:
        return blocked

    with tempfile.TemporaryDirectory(prefix="gem5_params_") as workdir:
        Path(workdir, "workload.c").write_text(
            "int main(void){return 0;}\n", encoding="utf-8"
        )
        script = (
            "gcc -O0 -static -o workload workload.c || exit 90; "
            f"{GEM5_BINARY} --outdir=m5out {GEM5_SE_CONFIG} "
            f"--cmd=./workload --cpu-type={model} --caches --l2cache "
            "> gem5.log 2>&1 || { tail -20 gem5.log >&2; exit 91; }; "
            "echo done"
        )
        try:
            returncode, _stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
                memory="4096m",
                cpus="2",
            )
        except TimeoutError:
            return {
                "success": False,
                "error": f"Reading the model's configuration timed out after {timeout_seconds}s",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Docker is not available to this process",
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"describe_model_parameters failed: {exc}")
            return {
                "success": False,
                "error": f"Could not read the model configuration: {exc}",
            }

        if returncode != 0:
            return {
                "success": False,
                "error": explain_gem5_failure(stderr, []),
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }

        config_path = Path(workdir, "m5out", "config.ini")
        if not config_path.exists():
            return {"success": False, "error": "gem5 produced no config.ini"}
        parsed = parse_model_parameters(config_path.read_text(encoding="utf-8"))

    wanted = {str(x).strip().lower() for x in (op_classes or []) if str(x).strip()}
    op_latencies = parsed["op_latencies"]
    if wanted:
        op_latencies = [
            row for row in op_latencies if row["op_class"].lower() in wanted
        ]

    missing_fma = model in MODELS_WITHOUT_SCALAR_FMA
    return {
        "success": True,
        "data": {
            "cpu_type": model,
            "image": image,
            "op_latencies": op_latencies[:120],
            "cpu_parameters": parsed["cpu_parameters"][:60],
            "op_class_count": len(parsed["op_latencies"]),
            "scalar_fma_supported": not missing_fma,
            "note": (
                "Set any of these with the param_overrides argument of "
                "simulate_c_workload, appending =<value> to each string in "
                "`parameters`. An op class may appear in several issue queues; "
                "set every path it lists or the model stays partly untuned. "
                "opLat is an integer number of cycles, so a measured latency "
                "of 4.29 can only ever be modelled as 4."
                + (
                    " This model has no functional unit for scalar fused "
                    "multiply-add, so a workload containing fmadd would hang "
                    "it; simulate_c_workload refuses those rather than hanging."
                    if missing_fma
                    else ""
                )
            ),
        },
    }


async def simulate_c_workload(
    *,
    code: str,
    flags: str = DEFAULT_FLAGS,
    cpu_type: str = DEFAULT_CPU,
    param_overrides: Optional[Sequence[str]] = None,
    run_args: str = "",
    label: str = "",
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Compile a self-contained C program and run it in a simulated core."""
    blocked = _check_arguments(code)
    if blocked:
        return blocked

    model = resolve_cpu_type(cpu_type or DEFAULT_CPU)
    if model not in CPU_TYPES:
        return {
            "success": False,
            "error": (
                f"Unknown cpu_type {cpu_type!r}. Available: "
                + ", ".join(f"{name} ({why})" for name, why in CPU_TYPES.items())
            ),
        }
    overrides = [str(p).strip() for p in (param_overrides or []) if str(p).strip()]
    if len(overrides) > MAX_PARAM_OVERRIDES:
        return {
            "success": False,
            "error": (
                f"{len(overrides)} parameter overrides exceeds the limit of "
                f"{MAX_PARAM_OVERRIDES}"
            ),
        }
    for override in overrides:
        if not SAFE_PARAM.match(override):
            return {
                "success": False,
                "error": (
                    f"parameter override {override!r} is not of the form "
                    "system.<path>=<value>. The path must start at `system`, "
                    "index vector members as FUList[3].opList[4], and contain "
                    "no shell metacharacters."
                ),
            }
    safe_flags = (flags or DEFAULT_FLAGS).strip()
    if not SAFE_FLAGS.match(safe_flags):
        return {
            "success": False,
            "error": f"flags contain unsupported characters: {flags!r}",
        }
    if "-static" not in safe_flags.split():
        # Syscall-emulation mode has no dynamic loader; a dynamically linked
        # binary fails inside the simulator with an error about the interpreter
        # that says nothing about how to fix it.
        safe_flags = f"{safe_flags} -static"
    arguments = (run_args or "").strip()
    if not SAFE_RUN_ARGS.match(arguments):
        return {
            "success": False,
            "error": f"run_args contain unsupported characters: {run_args!r}",
        }

    blocked = _check_environment(image)
    if blocked:
        return blocked

    support = await model_support(image, model)
    if not support.get("usable", True):
        return {"success": False, "error": support.get("reason", "")}

    with tempfile.TemporaryDirectory(prefix="gem5_workload_") as workdir:
        Path(workdir, "workload.c").write_text(code, encoding="utf-8")
        options = f" --options='{arguments}'" if arguments else ""
        script = (
            f"gcc {safe_flags} -o workload workload.c -lm 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            + (
                f"objdump -d workload | grep -qE '\\b({SCALAR_FMA_MNEMONICS})\\b' "
                "&& exit 92; "
                if model in MODELS_WITHOUT_SCALAR_FMA
                else ""
            )
            + f"{GEM5_BINARY} --outdir=m5out {GEM5_SE_CONFIG} "
            f"--cmd=./workload{options} --cpu-type={model} --caches --l2cache "
            + "".join(f"-P '{o}' " for o in overrides)
            + "> gem5.log 2>&1 || { tail -25 gem5.log >&2; exit 91; }; "
            "tail -5 gem5.log"
        )
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
                memory="4096m",
                cpus="2",
            )
        except TimeoutError:
            return {
                "success": False,
                "error": (
                    f"Simulation timed out after {timeout_seconds}s. An "
                    "out-of-order model runs on the order of 100k instructions "
                    "a second; shrink the workload's input."
                ),
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Docker is not available to this process",
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"simulate_c_workload failed: {exc}")
            return {"success": False, "error": f"Simulation failed: {exc}"}

        if returncode == 92:
            return {
                "success": False,
                "error": (
                    f"The compiled binary contains a scalar fused multiply-add, "
                    f"and the {model} model in this gem5 build has no functional "
                    "unit for it (its pool declares SimdFloatMultAcc but not "
                    "FloatMultAcc), so the simulation would hang rather than "
                    "fail. Either compile with -ffp-contract=off, or use a model "
                    "that implements it: O3CPU, O3_ARM_v7a_3 or HPI."
                ),
            }

        if returncode == 90:
            from app.services.agent_compiler_sandbox import explain_compiler_failure

            return {
                "success": False,
                "error": explain_compiler_failure(stderr),
                "compiler_stderr": stderr[:MAX_OUTPUT_CHARS],
            }
        if returncode == 91:
            return {
                "success": False,
                "error": explain_gem5_failure(stderr, overrides),
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }
        if returncode != 0:
            return {
                "success": False,
                "error": f"Simulation failed with exit code {returncode}",
                "stderr": stderr[:MAX_OUTPUT_CHARS],
            }

        stats_path = Path(workdir, "m5out", "stats.txt")
        if not stats_path.exists():
            return {
                "success": False,
                "error": "gem5 produced no stats.txt",
                "stdout": stdout[:MAX_OUTPUT_CHARS],
            }
        with stats_path.open() as handle:
            stats = gem5_stats.parse(handle)

    summary = gem5_stats.summarize(stats)
    subject = (label or "").strip() or "workload"
    source = f"gem5 {model}"

    return {
        "success": True,
        "data": {
            "image": image,
            "subject": subject,
            "cpu_type": model,
            "param_overrides": overrides,
            "flags": safe_flags,
            **summary,
            "measurement_source": source,
            "note": (
                "Simulated, not run on hardware: these are the modelled core's "
                "cycles. Compare cycles rather than sim_seconds, which also "
                "depends on the clock the configuration assigned."
            ),
        },
        "findings": [
            {
                "type": "simulated_measurement",
                "subject": subject,
                "title": (
                    f"{subject} @ {model}: {summary['cycles']} cycles, "
                    f"{summary['instructions']} instructions, IPC {summary['ipc']}"
                ),
                "cpu_type": model,
                "param_overrides": overrides,
                "cycles": summary["cycles"],
                "instructions": summary["instructions"],
                "ipc": summary["ipc"],
                "measurement_source": source,
            }
        ],
    }


# --- counters over time ---------------------------------------------------

#: The m5 dump-and-reset pseudo-op, as a magic instruction rather than a call.
#:
#: gem5 normally supplies these through util/m5, which this image does not
#: carry -- the build was stripped to keep the sandbox to 574 MB. On AArch64
#: the ops are a single instruction word, `0xff000110 | (func << 16)`, and
#: DUMP_RESET_STATS is func 0x42, so the library is not needed.
#:
#: x0 and x1 must be zeroed, and this is the whole trap. The op reads them as
#: (delay, period): a non-zero delay SCHEDULES the dump that many ticks ahead
#: rather than taking it now, so whatever the surrounding loop happened to
#: leave in x0 became a delay and the sample never fired before the program
#: exited. A simple probe appeared to work because those registers happened to
#: hold zero; a heavier workload emitted the instruction, executed it six
#: times, and produced exactly one dump. The failure is silent -- gem5 reports
#: nothing, and the result reads as a program that never sampled.
M5_SAMPLE_MACRO = (
    "/* Injected: take one counter sample. gem5 m5 DUMP_RESET_STATS. */\n"
    '#define M5_SAMPLE() __asm__ __volatile__("mov x0, #0\\n\\t'
    'mov x1, #0\\n\\t.inst 0xff420110" ::: "x0", "x1", "memory")\n'
)

#: Physical register file sizes SMT needs on this build, and the reason they
#: are here rather than left to the caller.
#:
#: O3CPU's defaults are sized for one thread, and running two panics in
#: `cpu.cc` with "Not enough physical registers". The panics come one at a
#: time -- fix numPhysVecPredRegs and it panics on numPhysMatRegs, fix that and
#: it runs -- so a caller discovering this pays for a full simulator startup
#: per register class. The values are generous rather than tuned: they are a
#: structural requirement for the run to start, not a microarchitectural
#: claim, and a study that varies them is studying the register file.
SMT_REGISTER_OVERRIDES = (
    "system.cpu[0].numPhysVecPredRegs=128",
    "system.cpu[0].numPhysMatRegs=32",
    "system.cpu[0].numPhysIntRegs=512",
    "system.cpu[0].numPhysFloatRegs=512",
    "system.cpu[0].numPhysVecRegs=512",
    "system.cpu[0].numPhysCCRegs=512",
)

#: A trace of one interval is a total, not a trace. Below this a predictor
#: study has nothing to learn from and should say so rather than proceed.
MIN_USEFUL_INTERVALS = 4


#: The cheap model. No timing at all, so its cycles are meaningless and its
#: instruction counts are exact -- which is the half a structural check needs.
PREFLIGHT_CPU = "AtomicSimpleCPU"

#: A design that cannot finish here has no prospect under an out-of-order
#: model, which is roughly two orders of magnitude slower. Timing out is
#: therefore an answer rather than an accident.
PREFLIGHT_TIMEOUT_SECONDS = 420


async def measure_structure(
    code: str,
    *,
    flags: str = DEFAULT_FLAGS,
    language: str = "c",
    extra_files: Optional[Dict[str, str]] = None,
    include_dirs: Optional[Sequence[str]] = None,
    image: str = DEFAULT_IMAGE,
) -> Dict[str, Any]:
    """Instructions per interval, taken in the cheapest model there is.

    The point is not to measure the workload but to find out whether it will
    support the measurement -- how many intervals it yields, whether the work
    is constant, whether it splits into regimes, and what the real run will
    cost. All of that is in the instruction counts, and instruction counts do
    not need a timing model.
    """
    result = await sample_counters(
        code=code,
        flags=flags,
        cpu_type=PREFLIGHT_CPU,
        language=language,
        extra_files=extra_files,
        include_dirs=include_dirs,
        image=image,
        max_counters=8,
        timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS,
        preflight=False,
    )
    if not result.get("success"):
        return {"measured": False, "error": result.get("error")}

    series = (result.get("data") or {}).get("series") or {}
    for name in ("simInsts", "system.cpu.commitStats0.numInsts"):
        counts = series.get(name)
        if isinstance(counts, list) and counts:
            return {"measured": True, "counter": name, "instructions": counts}
    return {
        "measured": False,
        "error": (
            "the cheap run produced no instruction counter, so the design "
            "could not be checked before the expensive run"
        ),
    }


async def sample_counters(
    *,
    code: str,
    flags: str = DEFAULT_FLAGS,
    cpu_type: str = DEFAULT_CPU,
    param_overrides: Optional[Sequence[str]] = None,
    run_args: str = "",
    label: str = "",
    max_counters: int = 60,
    language: str = "c",
    extra_files: Optional[Dict[str, str]] = None,
    include_dirs: Optional[Sequence[str]] = None,
    co_runner: str = "",
    image: str = DEFAULT_IMAGE,
    preflight: bool = True,
    intends_alternating_phases: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run a workload and return its hardware counters sampled over time.

    The workload calls `M5_SAMPLE()` wherever it wants a sample taken; the
    macro is injected, so the program does not have to declare it. Each call
    dumps every counter and resets them, so an interval holds what happened
    since the previous call -- which is the shape a hardware predictor reads,
    and not the shape a run total has.

    Only counters that actually move are returned. gem5 emits several hundred,
    most of them clock periods and configured sizes that are identical in every
    interval, and a counter that never changes cannot predict anything that
    does.

    `co_runner` runs a second program on the same core under SMT, which is the
    arrangement an SMT scheduling hint is about: two threads competing, and a
    predictor deciding which to favour. gem5 then reports progress per thread
    (`commitStats0`/`commitStats1`), so "will this thread make progress" is
    answerable. Only the primary program samples; its intervals cover both
    threads' activity.

    `language`, `extra_files` and `include_dirs` mirror `profile_c_workload`,
    because the corpora worth studying are not single files: Godot's core/math
    is C++ with a 25-header closure, and without these this tool can only be
    pointed at workloads written for it. A study run entirely on workloads
    written for the study is a study about the study.
    """
    blocked = _check_arguments(code)
    if blocked:
        return blocked
    if "M5_SAMPLE" not in code:
        return {
            "success": False,
            "error": (
                "The program never calls M5_SAMPLE(), so there is nothing to "
                "sample and this would return one total rather than a trace. "
                "Call M5_SAMPLE() at each point you want counters read -- "
                "typically once per outer-loop iteration or phase. The macro "
                "is injected for you; do not define it."
            ),
        }

    language = str(language or "c").strip().lower()
    if language not in ("c", "c++", "cpp", "cxx"):
        return {
            "success": False,
            "error": f"language must be 'c' or 'c++', got {language!r}",
        }
    is_cpp = language != "c"
    compiler, source_name = "gcc", "workload.c"
    if is_cpp:
        # Asked of the image, not asserted about it. Checked before the real
        # compile because the bare failure reports "g++: not found" -- an error
        # about a missing binary, when the situation may be that this image
        # cannot build C++ at all and no wording of the request will change it.
        support = await cpp_support(image)
        if not support["supported"]:
            return {
                "success": False,
                "error": (
                    f"The gem5 image ({image}) cannot build a static C++ "
                    "binary, so a C++ corpus cannot be counter-sampled on it. "
                    + (
                        f"The probe reported: {support['detail']} "
                        if support.get("detail")
                        else ""
                    )
                    + "gcc is present and C works. profile_c_workload runs C++ "
                    "because it uses a different image. Fixing this means "
                    "adding g++ and libstdc++-static to the gem5 image -- and "
                    "this refusal lifts on its own once they are there. Until "
                    "then, sample counters on a C workload, and do not read a "
                    "C result as standing in for the C++ corpus."
                ),
                "cpp_probe": support,
            }
        compiler, source_name = "g++", "workload.cc"

    model = resolve_cpu_type(cpu_type or DEFAULT_CPU)
    if model not in CPU_TYPES:
        return {
            "success": False,
            "error": (
                f"Unknown cpu_type {cpu_type!r}. Available: " + ", ".join(CPU_TYPES)
            ),
        }
    overrides = [str(x).strip() for x in (param_overrides or []) if str(x).strip()]
    if len(overrides) > MAX_PARAM_OVERRIDES:
        return {
            "success": False,
            "error": (
                f"{len(overrides)} parameter overrides exceeds the limit of "
                f"{MAX_PARAM_OVERRIDES}"
            ),
        }
    for override in overrides:
        if not SAFE_PARAM.match(override):
            return {
                "success": False,
                "error": (
                    f"parameter override {override!r} is not of the form "
                    "system.<path>=<value>."
                ),
            }

    safe_flags = (flags or DEFAULT_FLAGS).strip()
    if not SAFE_FLAGS.match(safe_flags):
        return {
            "success": False,
            "error": f"flags contain unsupported characters: {flags!r}",
        }
    if "-static" not in safe_flags.split():
        # Syscall-emulation mode has no dynamic loader.
        safe_flags = f"{safe_flags} -static"
    if is_cpp and "-std=" not in safe_flags:
        # The compiler defaults below C++17 and a corpus that needs it fails
        # with a static_assert naming neither the flag nor the caller.
        safe_flags = f"{safe_flags} -std=c++17"
    arguments = (run_args or "").strip()
    if not SAFE_RUN_ARGS.match(arguments):
        return {
            "success": False,
            "error": f"run_args contain unsupported characters: {run_args!r}",
        }

    blocked = _check_staged_paths(extra_files, include_dirs)
    if blocked:
        return blocked

    blocked = _check_environment(image)
    if blocked:
        return blocked

    support = await model_support(image, str(cpu_type or DEFAULT_CPU))
    if not support.get("usable", True):
        return {"success": False, "error": support.get("reason", "")}

    # Measure the design in the cheap model before paying for the real one.
    # After argument validation, never before it: rejecting an unknown
    # cpu_type is free, and a caller who mistyped one should be told that
    # rather than told about their workload after a simulation.
    # Deliberately default-on and not a separate tool: a run does not know its
    # design is unfit, so it will not think to ask. Skipped when this IS the
    # cheap pass, and when a co-runner makes the structure a property of two
    # programs interleaving rather than of this one.
    if preflight and str(cpu_type or DEFAULT_CPU) != PREFLIGHT_CPU and not co_runner:
        from app.services import agent_experiment_preflight

        structure = await measure_structure(
            code,
            flags=flags,
            language=language,
            extra_files=extra_files,
            include_dirs=include_dirs,
            image=image,
        )
        if structure.get("measured"):
            verdict = agent_experiment_preflight.judge(
                structure["instructions"],
                timeout_seconds=timeout_seconds,
                intends_alternating_phases=intends_alternating_phases,
            )
            if not verdict["fit"]:
                return {
                    "success": False,
                    "error": agent_experiment_preflight.refusal(verdict),
                    "preflight": verdict,
                }
            preflight_verdict = verdict
        else:
            # Could not check is not the same as checked and sound, and it is
            # not a reason to refuse either: the expensive run may still work.
            preflight_verdict = {"measured": False, "why": structure.get("error")}
    else:
        preflight_verdict = None

    source = M5_SAMPLE_MACRO + str(code)

    with tempfile.TemporaryDirectory(prefix="gem5_sample_") as workdir:
        Path(workdir, source_name).write_text(source, encoding="utf-8")

        root = Path(workdir).resolve()
        for name, content in (extra_files or {}).items():
            # Resolved and checked to stay under the work directory: these
            # names come from a caller, and a "header" called ../../etc is not
            # a header.
            candidate = (root / str(name)).resolve()
            if not str(candidate).startswith(str(root) + "/"):
                return {
                    "success": False,
                    "error": f"extra_files path escapes the workspace: {name!r}",
                }
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(str(content), encoding="utf-8")

        include_flags = ""
        for directory in include_dirs or []:
            resolved = (root / str(directory)).resolve()
            if not str(resolved).startswith(str(root)):
                return {
                    "success": False,
                    "error": f"include_dirs path escapes the workspace: {directory!r}",
                }
            include_flags += f"-I{Path(directory).as_posix()} "

        smt = bool(str(co_runner or "").strip())
        if smt:
            Path(workdir, "co_runner.c").write_text(str(co_runner), encoding="utf-8")
            overrides = list(SMT_REGISTER_OVERRIDES) + list(overrides)

        options = f" --options='{arguments}'" if arguments else ""
        cmd = "'./workload;./co_runner'" if smt else "./workload"
        smt_flags = " --smt -n 1" if smt else ""
        co_build = (
            "gcc -O2 -static -o co_runner co_runner.c -lm 2>co_err.txt || "
            "{ cat co_err.txt >&2; exit 93; }; "
            if smt
            else ""
        )
        script = (
            f"{compiler} {safe_flags} {include_flags}-o workload {source_name} "
            "-lm 2>compile_err.txt || "
            "{ cat compile_err.txt >&2; exit 90; }; "
            + co_build
            + f"{GEM5_BINARY} --outdir=m5out {GEM5_SE_CONFIG} "
            f"--cmd={cmd}{options}{smt_flags} --cpu-type={model} --caches --l2cache "
            + "".join(f"-P '{o}' " for o in overrides)
            + "> gem5.log 2>&1 || { tail -25 gem5.log >&2; exit 91; }; "
            "tail -3 gem5.log"
        )
        try:
            returncode, stdout, stderr = await agent_sandbox_runtime.run_in_sandbox(
                script,
                workdir,
                image=image,
                timeout_seconds=timeout_seconds,
                memory="4096m",
                cpus="2",
            )
        except TimeoutError:
            return {
                "success": False,
                "error": (
                    f"Sampling timed out after {timeout_seconds}s. An "
                    "out-of-order model runs on the order of 100k instructions "
                    "a second; shrink the workload or take fewer samples."
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"sample_counters failed: {exc}")
            return {"success": False, "error": f"Counter sampling failed: {exc}"}

        if returncode == 90:
            return {"success": False, "error": f"Compilation failed: {stderr[:800]}"}
        if returncode == 93:
            return {
                "success": False,
                "error": f"The co-runner failed to compile: {stderr[:600]}",
            }
        if returncode != 0:
            return {
                "success": False,
                "error": explain_gem5_failure(stderr, overrides),
            }

        stats_path = Path(workdir, "m5out", "stats.txt")
        if not stats_path.exists():
            return {"success": False, "error": "gem5 produced no stats.txt"}
        with stats_path.open() as handle:
            intervals = gem5_stats.parse_intervals(handle)

    names = gem5_stats.varying_counters(intervals, limit=max_counters)
    series = gem5_stats.as_series(intervals, names)

    # Per-thread IPC, because "will this thread make progress" is a rate and
    # neither of its two counters answers it alone. A thread doing identical
    # work each interval commits a constant instruction count whatever the
    # contention -- measured here as entropy 0.0, a target with nothing to
    # predict -- while the cycles it took to do so vary a great deal. Under
    # SMT the cycle count is shared by both threads, so it is not thread
    # progress either. The ratio is.
    for thread in (0, 1):
        insts = [
            i.get(f"system.cpu.commitStats{thread}.numInsts", 0.0) for i in intervals
        ]
        cycles = [i.get("system.cpu.numCycles", 0.0) for i in intervals]
        if any(insts) and all(c > 0 for c in cycles):
            series[f"derived.thread{thread}_ipc"] = [
                round(n / c, 6) for n, c in zip(insts, cycles)
            ]
    enough = len(intervals) >= MIN_USEFUL_INTERVALS

    # A co-runner that finishes early leaves the rest of the trace running
    # solo. Those intervals are not SMT and must not be read as contention --
    # the counters look calm because nothing is competing, not because the
    # predictor would have found it calm.
    # A trace that changes regime part way through is two experiments, and the
    # break is not detected by asking whether the co-runner is present -- it
    # was present throughout the run that motivated this, and merely finished
    # initialising at interval 104.
    regime = find_regime_change(cycles)
    regime_warning = (
        (
            f"the trace changes regime at interval {regime['at_interval']}: "
            f"median cycles per interval go "
            f"{regime['median_cycles_before']:,.0f} -> "
            f"{regime['median_cycles_after']:,.0f} ({regime['ratio']}x). "
            "Intervals either side describe different machines, so a "
            "predictability estimate taken across the break largely measures "
            "the break. Study one side, or lengthen the run until the opening "
            "regime is a negligible fraction of it."
        )
        if regime
        else ""
    )

    co_active = 0
    smt_warning = ""
    if smt:
        thread1 = [i.get("system.cpu.commitStats1.numInsts", 0.0) for i in intervals]
        co_active = sum(1 for v in thread1 if v and v > 0)
        if co_active < len(intervals):
            smt_warning = (
                f"the co-runner was active for {co_active} of {len(intervals)} "
                "intervals; the rest ran solo and are not SMT. Lengthen the "
                "co-runner or shorten the primary, and do not read the solo "
                "intervals as contention."
            )

    return {
        "success": True,
        "data": {
            "cpu_type": model,
            "label": str(label or ""),
            "smt": smt,
            "co_runner_active_intervals": co_active if smt else None,
            "regime_change": regime,
            "preflight": preflight_verdict,
            "intervals": len(intervals),
            "counters_varying": len(names),
            "counters": names,
            "series": series,
            "stdout": str(stdout)[:2000],
            "note": (
                "Each interval holds the counts since the previous M5_SAMPLE(). "
                "Only counters that move across the trace are returned; the "
                "rest are constants that cannot predict anything. "
                + (
                    ""
                    if enough
                    else f"WARNING: {len(intervals)} interval(s) is a total, "
                    "not a trace -- add more M5_SAMPLE() calls before drawing "
                    "any conclusion about predictability."
                )
                + (f" WARNING: {smt_warning}" if smt_warning else "")
                + (f" WARNING: {regime_warning}" if regime_warning else "")
            ),
        },
        "findings": [
            {
                "type": "counter_trace",
                "subject": str(label or "workload"),
                "title": (
                    f"{label or 'workload'} @ {model}: {len(intervals)} intervals, "
                    f"{len(names)} counters that vary"
                ),
                "cpu_type": model,
                "intervals": len(intervals),
                "counters_varying": len(names),
                "usable_as_trace": enough,
                "smt": smt,
                "co_runner_active_intervals": co_active if smt else None,
                "regime_change": regime,
            }
        ],
    }
