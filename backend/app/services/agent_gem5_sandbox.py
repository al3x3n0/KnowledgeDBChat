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


def _preflight(code: str, image: str) -> Optional[Dict[str, Any]]:
    if not (code or "").strip():
        return {"error": "code is required"}
    if len(code) > MAX_CODE_CHARS:
        return {"error": f"code exceeds {MAX_CODE_CHARS} characters"}
    if not agent_sandbox_runtime.execution_enabled():
        return {
            "error": (
                "Sandboxed execution is disabled on this server "
                "(ENABLE_UNSAFE_CODE_EXECUTION is false)."
            )
        }
    if image not in agent_sandbox_runtime.allowed_images():
        return {
            "error": (
                f"Image {image} is not allowlisted. Allowed: "
                f"{', '.join(agent_sandbox_runtime.allowed_images()) or 'none'}"
            )
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
    blocked = _preflight("int main(void){return 0;}", image)
    if blocked:
        return blocked

    model = str(cpu_type or DEFAULT_CPU).strip()
    if model not in CPU_TYPES:
        return {
            "error": (
                f"Unknown cpu_type {cpu_type!r}. Available: "
                + ", ".join(f"{name} ({why})" for name, why in CPU_TYPES.items())
            )
        }

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
                "error": f"Reading the model's configuration timed out after {timeout_seconds}s"
            }
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"describe_model_parameters failed: {exc}")
            return {"error": f"Could not read the model configuration: {exc}"}

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
    blocked = _preflight(code, image)
    if blocked:
        return blocked

    model = str(cpu_type or DEFAULT_CPU).strip()
    if model not in CPU_TYPES:
        return {
            "error": (
                f"Unknown cpu_type {cpu_type!r}. Available: "
                + ", ".join(f"{name} ({why})" for name, why in CPU_TYPES.items())
            )
        }
    overrides = [str(p).strip() for p in (param_overrides or []) if str(p).strip()]
    if len(overrides) > MAX_PARAM_OVERRIDES:
        return {
            "error": (
                f"{len(overrides)} parameter overrides exceeds the limit of "
                f"{MAX_PARAM_OVERRIDES}"
            )
        }
    for override in overrides:
        if not SAFE_PARAM.match(override):
            return {
                "error": (
                    f"parameter override {override!r} is not of the form "
                    "system.<path>=<value>. The path must start at `system`, "
                    "index vector members as FUList[3].opList[4], and contain "
                    "no shell metacharacters."
                )
            }
    safe_flags = (flags or DEFAULT_FLAGS).strip()
    if not SAFE_FLAGS.match(safe_flags):
        return {"error": f"flags contain unsupported characters: {flags!r}"}
    if "-static" not in safe_flags.split():
        # Syscall-emulation mode has no dynamic loader; a dynamically linked
        # binary fails inside the simulator with an error about the interpreter
        # that says nothing about how to fix it.
        safe_flags = f"{safe_flags} -static"
    arguments = (run_args or "").strip()
    if not SAFE_RUN_ARGS.match(arguments):
        return {"error": f"run_args contain unsupported characters: {run_args!r}"}

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
                "error": (
                    f"Simulation timed out after {timeout_seconds}s. An "
                    "out-of-order model runs on the order of 100k instructions "
                    "a second; shrink the workload's input."
                )
            }
        except FileNotFoundError:
            return {"error": "Docker is not available to this process"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"simulate_c_workload failed: {exc}")
            return {"error": f"Simulation failed: {exc}"}

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
